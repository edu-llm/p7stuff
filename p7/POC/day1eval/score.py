"""Score generated tutor responses with an LLM judge (gpt-5.6-sol via PromptLens).

Reads a ``generate.py`` output file, asks the judge to rate each generated
response on the 8 MRBench dimensions, attaches the judgment to each record,
computes aggregate scores, and writes ``<input>_scored.json``.

    export PROMPTLENS_API_KEY=...            # or put it in day1eval/.env
    python score.py --in outputs/olmo_V1.json
    python score.py --in outputs/olmo_V1.json --limit 10 --workers 8
    python score.py --self-test              # 1 tiny call to check connectivity

The API key is read from the environment / .env and never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from llm_client import LLMClientError, chat_completion
from scoring import DIMENSIONS, aggregate, build_judge_messages, format_summary, parse_judgment
from stats import bootstrap_ci, fmt_ci


def _load_dotenv() -> None:
    """Load a .env if python-dotenv is available (optional, non-overriding).

    Checks day1eval/.env, the repo-root .env (parent of day1eval), and the cwd,
    so it works whether you run from day1eval/ or the project root.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (
        os.path.join(config.ROOT, ".env"),               # day1eval/.env
        os.path.join(os.path.dirname(config.ROOT), ".env"),  # repo-root/.env
        os.path.join(os.getcwd(), ".env"),               # cwd
    ):
        if os.path.exists(path):
            load_dotenv(path, override=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-as-a-judge scoring for MRBench tutor responses.")
    ap.add_argument("--in", dest="in_path", default="",
                    help="generate.py output json (default: newest in outputs/).")
    ap.add_argument("--out", default="", help="Scored output path (default: <in>_scored.json).")
    ap.add_argument("--model", default=config.JUDGE_MODEL, help="Judge model id.")
    ap.add_argument("--limit", type=int, default=0, help="Cap #records (0 = all).")
    ap.add_argument("--workers", type=int, default=config.JUDGE_WORKERS)
    ap.add_argument("--max-tokens", type=int, default=config.JUDGE_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=config.JUDGE_TEMPERATURE)
    ap.add_argument("--json-mode", action="store_true",
                    help="Send response_format=json_object (only if the gateway supports it).")
    ap.add_argument("--self-test", action="store_true", help="One tiny call, then exit.")
    return ap.parse_args()


def _newest_output() -> str:
    if not os.path.isdir(config.OUTPUT_DIR):
        return ""
    cands = [
        os.path.join(config.OUTPUT_DIR, f)
        for f in os.listdir(config.OUTPUT_DIR)
        if f.endswith(".json") and not f.endswith("_scored.json")
    ]
    return max(cands, key=os.path.getmtime) if cands else ""


def _judge_one(record: dict, args: argparse.Namespace) -> dict:
    """Return the record augmented with a 'judgment' (or 'judge_error')."""
    messages = build_judge_messages(
        record.get("conversation_history", ""),
        record.get("generated_response", ""),
        record.get("Ground_Truth_Solution", ""),
    )
    rf = {"type": "json_object"} if args.json_mode else None
    try:
        text = chat_completion(
            messages,
            args.model,
            gateway_url=config.JUDGE_GATEWAY_URL,
            api_key_env=config.JUDGE_API_KEY_ENV,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            response_format=rf,
            max_retries=config.JUDGE_MAX_RETRIES,
        )
        record = dict(record)
        record["judgment"] = parse_judgment(text)
    except (LLMClientError, ValueError) as exc:
        record = dict(record)
        record["judgment"] = None
        record["judge_error"] = str(exc)
    return record


def _bootstrap_cis(judgments: list[dict]) -> dict[str, list[float] | None]:
    """Per-dimension (and overall) 95% bootstrap CIs over the judged dialogues.

    Returns {dim_key: [lo, hi], ..., "_overall": [lo, hi]}. A dimension's score
    is None where the judge output was invalid; bootstrap_ci drops those.
    """
    out: dict[str, list[float] | None] = {}
    for d in DIMENSIONS:
        per_dialogue = [d.score.get(j.get(d.key)) for j in judgments]
        _, lo, hi = bootstrap_ci(per_dialogue, seed=0)
        out[d.key] = [round(lo, 4), round(hi, 4)] if lo is not None else None
    overall_per_dialogue = []
    for j in judgments:
        vals = [d.score[j[d.key]] for d in DIMENSIONS if j.get(d.key) in d.score]
        overall_per_dialogue.append(sum(vals) / len(vals) if vals else None)
    _, lo, hi = bootstrap_ci(overall_per_dialogue, seed=0)
    out["_overall"] = [round(lo, 4), round(hi, 4)] if lo is not None else None
    return out


def self_test(args: argparse.Namespace) -> int:
    print(f"[score] self-test -> {args.model} @ {config.JUDGE_GATEWAY_URL}")
    try:
        text = chat_completion(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            args.model,
            gateway_url=config.JUDGE_GATEWAY_URL,
            api_key_env=config.JUDGE_API_KEY_ENV,
            max_tokens=config.JUDGE_MAX_TOKENS,
            temperature=args.temperature,
            max_retries=config.JUDGE_MAX_RETRIES,
        )
    except LLMClientError as exc:
        print(f"[score] FAILED: {exc}")
        return 1
    print(f"[score] OK. Model replied: {text.strip()[:120]!r}")
    return 0


def main() -> int:
    _load_dotenv()
    args = parse_args()

    if not os.environ.get(config.JUDGE_API_KEY_ENV):
        print(f"[score] WARNING: {config.JUDGE_API_KEY_ENV} not set "
              f"(export it or add to day1eval/.env).", file=sys.stderr)

    if args.self_test:
        return self_test(args)

    in_path = args.in_path or _newest_output()
    if not in_path or not os.path.exists(in_path):
        print(f"[score] no input file (got {in_path!r}). Run generate.py first "
              f"or pass --in.", file=sys.stderr)
        return 2

    with open(in_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    records = data.get("records", data if isinstance(data, list) else [])
    if args.limit:
        records = records[:args.limit]
    print(f"[score] judging {len(records)} responses from {in_path} with {args.model}")

    scored: list[dict | None] = [None] * len(records)
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_judge_one, rec, args): i for i, rec in enumerate(records)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            rec = fut.result()
            scored[i] = rec
            if rec.get("judgment") is None:
                errors += 1
            done += 1
            if done % 10 == 0 or done == len(records):
                print(f"[score]   {done}/{len(records)} (errors: {errors})")

    elapsed = time.time() - t0
    judgments = [r["judgment"] for r in scored if r and r.get("judgment")]
    summary = aggregate([j for j in judgments])
    ci = _bootstrap_cis(judgments)
    for key, bounds in ci.items():
        if key in summary and isinstance(summary[key], dict):
            summary[key]["ci95"] = bounds
    summary["_overall_ci95"] = ci.get("_overall")

    out_path = args.out or (os.path.splitext(in_path)[0] + "_scored.json")
    result = {
        "meta": {
            **data.get("meta", {}),
            "judge_model": args.model,
            "judged": len(records),
            "judge_errors": errors,
            "judge_elapsed_sec": round(elapsed, 2),
        },
        "summary": summary,
        "records": scored,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    title = (f"\n=== Judge: {args.model} | {len(judgments)}/{len(records)} scored "
             f"({errors} errors) in {elapsed:.1f}s ===")
    print(format_summary(summary, title))

    print("\n95% bootstrap CIs (mean [lo, hi]):")
    for d in DIMENSIONS:
        s = summary.get(d.key, {})
        mean = s.get("mean_score")
        bounds = s.get("ci95")
        if mean is None or not bounds:
            print(f"  {d.key:<26} —")
        else:
            print(f"  {d.key:<26} {fmt_ci(mean, bounds[0], bounds[1])}")
    ov, ovb = summary.get("_overall_mean_score"), summary.get("_overall_ci95")
    if ov is not None and ovb:
        print(f"  {'OVERALL':<26} {fmt_ci(ov, ovb[0], ovb[1])}")

    print(f"\n[score] wrote {out_path}")
    if errors:
        print(f"[score] NOTE: {errors} records failed to judge "
              f"(see 'judge_error' fields).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
