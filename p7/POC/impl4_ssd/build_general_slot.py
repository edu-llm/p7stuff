#!/usr/bin/env python
"""Step 2b (PLAN §8.2) — build one arm's replay slot.

Pipeline: shared SuperNI pool -> generation under the arm's sampling config (§4)
-> degeneracy filter (§4) -> optional B2 gate (§9) -> token-budget subsample (§5)
-> ``runs/<arm>/general_slot.jsonl`` + the ``general_slot`` manifest section.

What each arm gets:

    A1   7,496 Tülu-3 gold                  <- defines the token budget everyone matches
    A2   7,496 SuperNI *gold*               <- prompt shift, or self-generation?
    A3   7,496 SuperNI SSD at T1            <- the intervention (= Block T's T1)
    A4   3,748 Tülu gold + 3,748 SSD at T1
    T2/T3/T4  7,496 SSD at that sampling config
    B2   7,496 SSD at T1, gated against SuperNI gold

Because ``T`` and ``ρ`` change output length and degeneracy rate, token matching and
the degeneracy filter are recomputed per arm — a ``T`` comparison across arms with
different realized token weights is not a ``T`` comparison.

Run A1 first: it writes ``data/tulu_reference.json``, the token total every other
arm is matched to.

Usage:
    python build_general_slot.py --arm A1
    python build_general_slot.py --arm A3 --backend vllm
    python build_general_slot.py --arm T4 --backend hf --max_tokens 384
    python build_general_slot.py --arm B2 --poc          # tiny end-to-end rehearsal
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from impl4 import chat, degeneracy, gate, generate, manifest, mixing, superni, tulu
from impl4.config import (
    ALL_ARMS,
    ARM_CHOICES,
    BASE_MODEL,
    MAX_LEN,
    OVERGENERATE,
    SEED,
    resolve_arm,
    slot_sizes,
)
from impl4.paths import (
    SUPERNI_GOLD_REFERENCE,
    SUPERNI_POOL,
    TULU_REFERENCE,
    run_dir,
)

SUPERNI_SOURCE_ID = f"allenai/natural-instructions@{superni.SUPERNI_COMMIT}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES,
                   help=f"One of {', '.join(ALL_ARMS)} (T1 is an alias of A3).")
    p.add_argument("--pool", default=str(SUPERNI_POOL))
    p.add_argument("--runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--batch_size", type=int, default=32,
                   help="HF backend only: max rows per generation batch.")
    p.add_argument("--max_batch_tokens", type=int, default=32768,
                   help="HF backend only: max padded tokens per generation batch. Attention "
                        "memory grows as rows x length^2, so a row cap alone lets a batch of "
                        "long prompts OOM; this bounds it. Raise on a big GPU for speed.")
    p.add_argument("--max_prompt_tokens", type=int, default=MAX_LEN - 128,
                   help="Drop pool prompts longer than this. A prompt near max_len leaves no "
                        "room for the assistant turn, so the record trains on nothing and is "
                        "discarded anyway — this just avoids generating it first.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM only.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--poc", action="store_true", help="63-block smoke slot instead of 937.")

    p.add_argument("--max_tokens", type=int, default=0,
                   help="0 = auto-calibrate against the Tulu reference mean target length.")
    p.add_argument("--calibrate_n", type=int, default=256,
                   help="Pilot prompts per calibration round.")
    p.add_argument("--calibrate_rounds", type=int, default=3)
    p.add_argument("--overgenerate", type=float, default=OVERGENERATE)
    p.add_argument("--token_reference", choices=["a1", "superni_gold"], default="a1",
                   help="Which token budget to match this arm's replay slot to. 'a1' is "
                        "PLAN §5's literal instruction. 'superni_gold' matches A2's "
                        "realized total instead, which makes the A2<->A3 paired control "
                        "exact — see load_reference() for why A2 cannot be matched to A1.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def _load_ref_file(path, built_by: str, poc: bool, n_expected: int) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path}.\nBuild {built_by} first — it defines the token budget this "
            f"arm is matched to:\n"
            f"    python build_general_slot.py --arm {built_by}{' --poc' if poc else ''}"
        )
    ref = json.loads(path.read_text(encoding="utf-8"))
    if bool(ref.get("poc")) != poc or ref.get("n") != n_expected:
        raise SystemExit(
            f"{path} was built at n={ref.get('n')} (poc={ref.get('poc')}) but this arm needs "
            f"n={n_expected} (poc={poc}). Silently rescaling would fake the token match. "
            f"Rebuild it at the same setting:\n"
            f"    python build_general_slot.py --arm {built_by}"
            f"{' --poc' if poc else ''} --force"
        )
    return ref


def load_reference(which: str, poc: bool, n_expected: int) -> dict:
    """The token budget this arm's replay slot is matched to (PLAN §5).

    ``a1`` — A1's Tülu-3 slot, the plan's literal instruction and the default.

    ``superni_gold`` — A2's realized total instead. An escape hatch, not the
    recommended path. At the measured numbers (A1 = 600,173 label tokens over 7,496
    examples; SuperNI gold at ``--min_gold_words 20`` reaches 174k-617k for a subset
    of that size) A2 *can* be matched to A1 exactly, so this is unnecessary. It exists
    for the case where a differently-filtered pool cannot bracket A1's total: PLAN §5
    calls A2↔A3 the "clean paired control" and says "A1 remains the external
    reference", so matching the SSD arms to A2 makes *that* pair exact at the cost of
    the A1 comparison. Whichever you pick is recorded in the manifest; do not mix the
    two within one comparison.
    """
    if which == "a1":
        return _load_ref_file(TULU_REFERENCE, "A1", poc, n_expected)
    if which == "superni_gold":
        return _load_ref_file(SUPERNI_GOLD_REFERENCE, "A2", poc, n_expected)
    raise ValueError(f"unknown --token_reference {which!r} (a1|superni_gold)")


def superni_record(item: dict, assistant: str, kind: str, sampling=None,
                   max_tokens=None, gate_info=None, n_tries=None) -> dict:
    sc = sampling
    return {
        "messages": [
            {"role": "user", "content": superni.user_message(item)},
            {"role": "assistant", "content": assistant},
        ],
        "problem_id": None,
        "dialogue_id": item.get("instance_id"),
        "answer": item.get("gold"),
        "source": SUPERNI_SOURCE_ID,
        # `kind` is the coarse stream label Impl 3's tagging uses ("pedagogy" / "general"),
        # so a train file from either project reads the same way. Impl 4's own provenance —
        # which replay source this row came from — moves to `replay_kind`.
        "kind": "general",
        "replay_kind": kind,
        "superni_task_id": item["superni_task_id"],
        "sample_T": sc.temperature if sc else None,
        "sample_top_k": sc.top_k if sc else None,
        "sample_top_p": sc.top_p if sc else None,
        "sample_max_tokens": max_tokens,
        "gate_passed": None if gate_info is None else gate_info[0],
        "gate_how": None if gate_info is None else gate_info[1],
        "gate_rouge_l": None if gate_info is None else round(gate_info[2], 4),
        "n_tries": n_tries,
    }


def filter_by_prompt_tokens(pool, tokenizer, max_prompt_tokens: int, log=print):
    """Drop pool prompts too long to leave room for a target at ``MAX_LEN``.

    This is not a new restriction — it is :func:`drop_unlabelled` moved earlier. A prompt
    longer than ``MAX_LEN`` consumes the whole sequence, ``make_tokenize_fn`` truncates the
    assistant turn away, and the record trains on nothing, so it was always going to be
    discarded. Doing it before generation instead of after means we do not pay to sample
    ~1.4k tokens of output for a record that gets thrown away, and it removes the
    batch-memory blowup that a 6k-token prompt causes.

    Applied to the *shared* pool, so A2 (gold) and the SSD arms still draw from an
    identical prompt set at identical counts — the paired control PLAN §5 depends on.

    The drop is task-correlated (long-passage summarization and contract QA lose the most),
    which shifts the replay slot's task mix. The retained/dropped counts per task are
    returned and recorded in the manifest so that shift is visible rather than implicit.
    """
    kept, dropped = [], Counter()
    for item in pool:
        n = len(chat.generation_prompt_ids(
            tokenizer, [{"role": "user", "content": superni.user_message(item)}]))
        if n <= max_prompt_tokens:
            kept.append(item)
        else:
            dropped[item["superni_task_id"]] += 1
    n_drop = sum(dropped.values())
    if n_drop:
        log(f"  prompt-length filter (<= {max_prompt_tokens} tokens, so a target fits within "
            f"max_len={MAX_LEN}): kept {len(kept)}/{len(pool)}, dropped {n_drop}")
        for task, c in dropped.most_common(5):
            log(f"    -{c:>5}  {task}")
        if len(dropped) > 5:
            log(f"    ... and {len(dropped) - 5} more tasks")
        gone = {t for t in dropped} - {i["superni_task_id"] for i in kept}
        if gone:
            log(f"    WARNING: {len(gone)} task(s) removed entirely: {sorted(gone)}")
    return kept, {"max_prompt_tokens": max_prompt_tokens, "n_before": len(pool),
                  "n_kept": len(kept), "n_dropped": n_drop,
                  "dropped_by_task": dict(dropped),
                  "n_tasks_after": len({i["superni_task_id"] for i in kept})}


def drop_unlabelled(records, counts, what: str, log=print):
    """Remove records that contribute zero unmasked label tokens.

    A SuperNI prompt (task definition + a long passage) can exceed ``max_len`` on its
    own, in which case ``make_tokenize_fn`` truncates the assistant turn away entirely.
    Such a record trains on nothing, yet still occupies one of the 8 general slots in
    every block — so it silently dilutes the replay stream. Dropping them here is the
    only place it can be done without breaking the §6 block layout downstream.
    """
    keep = [i for i, c in enumerate(counts) if c > 0]
    dropped = len(counts) - len(keep)
    if dropped:
        log(f"  dropped {dropped}/{len(counts)} {what} candidates with 0 label tokens "
            f"(prompt alone exceeds max_len={MAX_LEN})")
    return [records[i] for i in keep], [counts[i] for i in keep], dropped


def gen_round(items, sampling, max_tokens, engine, args, seed, log=print) -> list[str]:
    msgs = [[{"role": "user", "content": superni.user_message(it)}] for it in items]
    res = generate.generate_targets(
        msgs, sampling, max_tokens, model_id=args.base_model,
        backend=args.backend, seed=seed, batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization, engine=engine, log=log,
        max_batch_tokens=args.max_batch_tokens,
    )
    return res.texts


def calibrate_max_tokens(items, sampling, engine, args, tok_fn, target_mean, log=print):
    """Tune ``max_tokens`` so mean realized target length ~= the Tülu slot's (PLAN §4).

    ``max_tokens`` is a *cap*, so this can only raise the mean while the cap binds.
    When two rounds in a row fail to move the mean, the natural length is below the
    target and we stop and report the shortfall rather than pretend to have matched.
    """
    cur = max(64, int(round(target_mean * 1.2)))
    trace = []
    prev_mean = None
    # Calibrate on the *tail* of the pool: build_ssd draws from the head, so the pilot
    # generations are thrown away without wasting prompts that end up in the slot.
    pilot = items[-args.calibrate_n:] if len(items) > args.calibrate_n else list(items)
    for r in range(args.calibrate_rounds):
        texts = gen_round(pilot, sampling, cur, engine, args, args.seed + 7919 * (r + 1), log)
        kept = [t for t in texts if not degeneracy.is_degenerate(t)]
        if not kept:
            log(f"  calibration round {r}: every pilot output was degenerate at "
                f"max_tokens={cur}; doubling and retrying")
            trace.append({"round": r, "max_tokens": cur, "mean_label_tokens": 0.0,
                          "kept": 0, "of": len(texts)})
            cur *= 2
            continue
        counts = [
            chat.label_token_count(tok_fn, {"messages": [
                {"role": "user", "content": superni.user_message(it)},
                {"role": "assistant", "content": t},
            ]})
            for it, t in zip(pilot, texts) if not degeneracy.is_degenerate(t)
        ]
        mean = sum(counts) / len(counts)
        trace.append({"round": r, "max_tokens": cur, "mean_label_tokens": round(mean, 1),
                      "kept": len(kept), "of": len(texts)})
        log(f"  calibration round {r}: max_tokens={cur} -> mean label tokens {mean:.1f} "
            f"(target {target_mean:.1f})")
        if abs(mean - target_mean) / target_mean <= 0.05:
            break
        if prev_mean is not None and abs(mean - prev_mean) < 0.02 * target_mean:
            log("  mean stopped responding to the cap — natural output length is the "
                "binding constraint, not max_tokens. Stopping calibration.")
            break
        prev_mean = mean
        cur = max(32, min(MAX_LEN, int(round(cur * target_mean / mean))))
    return cur, trace


# ---------------------------------------------------------------------------
def build_ssd(items, sampling, n_target, target_tokens, engine, args, tok_fn,
              gated: bool, max_rounds: int = 24, log=print):
    """Generate, degeneracy-filter, optionally gate, then token-match to ``n_target``.

    Prompts are tracked by pool index. Two distinct retry mechanisms, deliberately
    kept separate:

    * **degeneracy** (all arms) — a degenerate output is dropped and the *prompt* is
      replaced from the pool. That is just over-generation catching up; it is not
      quality gating.
    * **the B2 gate** (``gated=True`` only) — a failing output is *resampled* on the
      same prompt up to ``gate.MAX_TRIES`` times, and only then is the prompt dropped
      and another drawn. We never fall back to gold: doing so would reinject
      off-policy targets exactly where we are trying to remove them (PLAN §9).
    """
    n_candidates = min(len(items), int(math.ceil(n_target * args.overgenerate)))
    log(f"Generating {n_candidates} candidates for a {n_target}-example slot "
        f"(over-generation x{args.overgenerate}, pool has {len(items)})")

    accepted: list[tuple[int, str, tuple | None, int]] = []
    drop_reasons: Counter = Counter()
    gate_fail_final = 0
    tries: dict[int, int] = {}
    cursor = min(len(items), n_candidates)
    active = list(range(cursor))
    for i in active:
        tries[i] = 0
    round_no = 0
    total_generated = 0

    while active and round_no < max_rounds:
        round_no += 1
        texts = gen_round([items[i] for i in active], sampling, args.max_tokens,
                          engine, args, args.seed + 104729 * round_no, log)
        total_generated += len(texts)
        retry: list[int] = []
        for i, text in zip(active, texts):
            tries[i] += 1
            reason = degeneracy.degeneracy_reason(text)
            if reason:
                drop_reasons[reason] += 1
                if gated and tries[i] < gate.MAX_TRIES:
                    retry.append(i)
                continue
            if gated:
                info = gate.gate_result(text, items[i]["gold"])
                if not info[0]:
                    drop_reasons["gate_fail"] += 1
                    if tries[i] < gate.MAX_TRIES:
                        retry.append(i)
                    else:
                        gate_fail_final += 1
                    continue
                accepted.append((i, text, info, tries[i]))
            else:
                accepted.append((i, text, None, tries[i]))

        if len(accepted) >= n_candidates:
            break
        short = n_candidates - len(accepted) - len(retry)
        replacements: list[int] = []
        if short > 0 and cursor < len(items):
            replacements = list(range(cursor, min(len(items), cursor + short)))
            cursor += len(replacements)
            for i in replacements:
                tries[i] = 0
        active = retry + replacements
        if active:
            log(f"  round {round_no}: accepted {len(accepted)}/{n_candidates}, "
                f"resampling {len(retry)}, drawing {len(replacements)} fresh prompts")

    if len(accepted) < n_target:
        raise SystemExit(
            f"only {len(accepted)} usable generations for a {n_target}-example slot. "
            f"Raise --overgenerate, enlarge the prompt pool (build_prompt_pool.py "
            f"--n_prompts), or relax the arm. Drop reasons: {dict(drop_reasons)}"
        )

    records = [
        superni_record(items[i], text, "general_ssd", sampling=sampling,
                       max_tokens=args.max_tokens, gate_info=info, n_tries=nt)
        for i, text, info, nt in accepted
    ]
    counts = chat.label_token_counts(tok_fn, records)
    records, counts, n_unlabelled = drop_unlabelled(records, counts, "SSD", log=log)
    if len(records) < n_target:
        raise SystemExit(
            f"only {len(records)} SSD candidates carry any label tokens (need {n_target}). "
            f"Raise --overgenerate or shorten the prompts.")
    keep, tstats = mixing.token_matched_select(counts, n_target, target_tokens,
                                               seed=args.seed)
    selected = [records[i] for i in keep]
    sel_counts = [counts[i] for i in keep]

    stats = {
        "n_generated": total_generated,
        "n_dropped_zero_label_tokens": n_unlabelled,
        "n_candidates_requested": n_candidates,
        "n_accepted": len(accepted),
        "n_rounds": round_no,
        "degeneracy_drops": dict(drop_reasons),
        "degeneracy_drop_rate": round(
            sum(v for k, v in drop_reasons.items() if k != "gate_fail") / max(1, total_generated), 4),
        "gate_drop_rate": (round(drop_reasons.get("gate_fail", 0) / max(1, total_generated), 4)
                           if gated else None),
        "gate_prompts_dropped_after_max_tries": gate_fail_final if gated else None,
        "gate_max_tries": gate.MAX_TRIES if gated else None,
        "gate_threshold": gate.GATE_THRESHOLD if gated else None,
        "token_match": tstats,
        "mean_label_tokens": round(sum(sel_counts) / max(1, len(sel_counts)), 1),
        "mean_output_words": round(
            sum(len(r["messages"][1]["content"].split()) for r in selected) / max(1, len(selected)), 1),
    }
    return selected, stats


def build_superni_gold(items, n_target, target_tokens, args, tok_fn):
    """SuperNI gold as the replay target (A2).

    ``target_tokens=None`` means "take a natural seeded sample and let the realized
    total stand" — used when A2 is itself the token reference, where preserving the
    round-robin breadth matters more than chasing a total the data cannot reach.
    """
    n_candidates = min(len(items), int(math.ceil(n_target * args.overgenerate)))
    records = [superni_record(it, it["gold"], "general_gold_superni")
               for it in items[:n_candidates]]
    counts = chat.label_token_counts(tok_fn, records)
    records, counts, n_unlabelled = drop_unlabelled(records, counts, "SuperNI-gold")
    if len(records) < n_target:
        raise SystemExit(
            f"only {len(records)} SuperNI-gold candidates carry any label tokens "
            f"(need {n_target}). Lower --min_gold_words or enlarge the pool.")
    keep, tstats = mixing.token_matched_select(counts, n_target, target_tokens, seed=args.seed)
    selected = [records[i] for i in keep]
    sel_counts = [counts[i] for i in keep]
    return selected, {
        "n_candidates": n_candidates,
        "n_dropped_zero_label_tokens": n_unlabelled,
        "token_match": tstats,
        "total_label_tokens": sum(sel_counts),
        "mean_label_tokens": round(sum(sel_counts) / max(1, len(sel_counts)), 1),
        "mean_output_words": round(
            sum(len(r["messages"][1]["content"].split()) for r in selected) / max(1, len(selected)), 1),
    }


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    _, n_gen = slot_sizes(args.poc)
    n_ssd = int(round(n_gen * arm.sigma))
    n_gold = n_gen - n_ssd

    out_dir = run_dir(arm.name, args.runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_path = out_dir / "general_slot.jsonl"
    if slot_path.exists() and not args.force:
        print(f"{slot_path} already present. Use --force to rebuild.")
        return

    manifest.init(out_dir, arm, poc=args.poc)
    print("=" * 74)
    print(f"Arm {arm.name} (block {arm.block}) | sigma={arm.sigma} delta=0 | "
          f"gated={arm.gated} | slot={n_gen} = {n_gold} gold + {n_ssd} ssd")
    if arm.aliases:
        print(f"  aliases: {', '.join(arm.aliases)} (same run, referenced from both blocks)")
    print("=" * 74)

    # Resolve the token reference *before* touching the model, so a missing A1 build
    # fails with an actionable message rather than a tokenizer download.
    #
    # A1 always defines the Tülu budget. A2 defines the SuperNI-gold budget only when
    # that is the regime we asked for; under the default it tries (and, per the
    # docstring on load_reference, fails) to match A1, and records the shortfall.
    self_defining = (arm.name == "A1"
                     or (arm.name == "A2" and args.token_reference == "superni_gold"))
    reference = None if self_defining else load_reference(args.token_reference,
                                                          args.poc, n_gen)

    tokenizer = chat.load_tokenizer(args.base_model)
    tok_fn = chat.make_tokenize_fn(tokenizer, MAX_LEN)

    # --- the gold half -----------------------------------------------------
    parts: list[dict] = []
    section: dict = {
        "n_target": n_gen, "n_gold": n_gold, "n_ssd": n_ssd,
        "sigma": arm.sigma, "max_len": MAX_LEN,
        "token_reference": "self" if self_defining else args.token_reference,
    }

    if n_gold and arm.gold_source == "tulu":
        # Over-request, then drop the conversations whose prompt consumes the whole
        # max_len budget and leaves no assistant turn, then trim back to n_gold.
        #
        # The SuperNI paths have always done this (drop_unlabelled); the Tulu path did
        # not, so A1's slot could contain records that train on nothing while still
        # occupying one of the 8 general slots in a block — diluting the replay stream.
        # mix_and_order.py refuses such a mix, which is how this surfaced: 34 of 504
        # Tulu conversations hit it at max_len=1024.
        #
        # Trimming *after* the filter keeps A4's prefix property intact: both arms filter
        # the same seed-deterministic sequence the same way, so A4's half is still exactly
        # the first half of A1's.
        want = int(math.ceil(n_gold * args.overgenerate))
        print(f"Loading {want} Tulu-3 gold conversations for a {n_gold}-example slot ...")
        gold = tulu.load_tulu_slot(want, args.seed)
        gold_counts = chat.label_token_counts(tok_fn, gold)
        gold, gold_counts, n_gold_unlabelled = drop_unlabelled(gold, gold_counts, "Tulu-gold")
        if len(gold) < n_gold:
            raise SystemExit(
                f"only {len(gold)} Tulu conversations carry any label tokens at "
                f"max_len={MAX_LEN} (need {n_gold}). Raise --overgenerate.")
        gold, gold_counts = gold[:n_gold], gold_counts[:n_gold]
        parts += gold
        section["gold"] = {
            "source": tulu.TULU_ID,
            "kind": tulu.KIND,
            "n": len(gold),
            "n_requested": want,
            "n_dropped_zero_label_tokens": n_gold_unlabelled,
            "total_label_tokens": sum(gold_counts),
            "mean_label_tokens": round(sum(gold_counts) / max(1, len(gold)), 1),
            "note": ("Seed-deterministic ordering, so A4's Tulu half is exactly the first "
                     "half of A1's slot."),
        }
        if arm.name == "A1":
            ref = {
                "arm": "A1",
                "n": len(gold),
                "total_label_tokens": sum(gold_counts),
                "mean_label_tokens": sum(gold_counts) / max(1, len(gold)),
                "max_len": MAX_LEN,
                "base_model": args.base_model,
                "seed": args.seed,
                "poc": args.poc,
            }
            TULU_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
            TULU_REFERENCE.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote token reference -> {TULU_REFERENCE}: "
                  f"{ref['total_label_tokens']} label tokens over {ref['n']} examples "
                  f"(mean {ref['mean_label_tokens']:.1f})")
            reference = ref

    # --- the SuperNI half(s) ----------------------------------------------
    need_pool = (arm.gold_source == "superni") or n_ssd > 0
    pool: list[dict] = []
    if need_pool:
        pool_path = Path(args.pool)
        if not pool_path.exists():
            raise SystemExit(
                f"missing {pool_path}. Build the shared pool first:\n"
                f"    python build_prompt_pool.py [--superni_dir /path/to/natural-instructions]"
            )
        pool = manifest.read_jsonl(pool_path)
        print(f"SuperNI pool: {len(pool)} prompts from "
              f"{len({p['superni_task_id'] for p in pool})} tasks")
        n_raw = len(pool)
        pool, lenstats = filter_by_prompt_tokens(pool, tokenizer, args.max_prompt_tokens)
        if len(pool) < n_gen:
            raise SystemExit(
                f"only {len(pool)} prompts survive the <= {args.max_prompt_tokens}-token "
                f"filter, need at least {n_gen} (plus over-generation). Raise "
                f"--max_prompt_tokens (at the cost of records that train on nothing) or "
                f"enlarge the pool with build_prompt_pool.py --n_prompts.")
        section["prompt_pool"] = {
            "path": str(pool_path), "n_raw": n_raw, "n": len(pool),
            "n_tasks": len({p["superni_task_id"] for p in pool}),
            "source": SUPERNI_SOURCE_ID,
            "prompt_length_filter": lenstats,
        }

    if n_gold and arm.gold_source == "superni":
        # A2 has no reference of its own: it takes the longest-available subset it can
        # (target = +inf via a very large budget), which is the most it could ever match
        # A1 by. The realized total then *becomes* the superni_gold reference.
        target = (reference["total_label_tokens"] * (n_gold / reference["n"])
                  if reference else None)
        gold, gstats = build_superni_gold(
            pool, n_gold, int(round(target)) if target else None, args, tok_fn)
        parts += gold
        gstats.update({"source": SUPERNI_SOURCE_ID, "kind": "general_gold_superni",
                       "n": len(gold)})
        section["gold"] = gstats
        print(f"SuperNI gold slot: {len(gold)} examples, mean {gstats['mean_label_tokens']} "
              f"label tokens, total {gstats['total_label_tokens']}")
        if arm.name == "A2":
            ref = {
                "arm": "A2",
                "n": len(gold),
                "total_label_tokens": gstats["total_label_tokens"],
                "mean_label_tokens": gstats["total_label_tokens"] / max(1, len(gold)),
                "max_len": MAX_LEN, "base_model": args.base_model,
                "seed": args.seed, "poc": args.poc,
                "note": ("SuperNI gold is far shorter than Tulu-3 gold even after the "
                         "PLAN §3 >=30-word filter, so this total is the ceiling for a "
                         "7,496-example SuperNI gold slot. Pass --token_reference "
                         "superni_gold on the SSD arms to make the A2<->A3 pair exact."),
            }
            SUPERNI_GOLD_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
            SUPERNI_GOLD_REFERENCE.write_text(json.dumps(ref, indent=2) + "\n",
                                              encoding="utf-8")
            print(f"Wrote SuperNI-gold reference -> {SUPERNI_GOLD_REFERENCE}: "
                  f"{ref['total_label_tokens']} label tokens "
                  f"(mean {ref['mean_label_tokens']:.1f})")
            if reference is None:
                reference = ref

    if n_ssd:
        sampling = arm.sampling_config
        assert sampling is not None
        engine = generate.build_engine(args.backend, args.base_model, args.seed,
                                       args.gpu_memory_utilization)
        args.backend = engine.backend

        gold_tokens = section.get("gold", {}).get("total_label_tokens")
        if gold_tokens is None:
            gold_tokens = sum(chat.label_token_counts(tok_fn, parts)) if parts else 0
        ssd_target = int(round(reference["total_label_tokens"] - gold_tokens))
        target_mean = ssd_target / max(1, n_ssd)

        if args.max_tokens <= 0:
            print(f"Calibrating max_tokens toward a {target_mean:.1f}-label-token mean ...")
            args.max_tokens, trace = calibrate_max_tokens(
                pool, sampling, engine, args, tok_fn, target_mean)
            section["max_tokens_calibration"] = {"trace": trace, "chosen": args.max_tokens,
                                                 "target_mean_label_tokens": round(target_mean, 1)}
            print(f"  chose max_tokens={args.max_tokens}")
        else:
            section["max_tokens_calibration"] = {"trace": None, "chosen": args.max_tokens,
                                                 "target_mean_label_tokens": round(target_mean, 1),
                                                 "note": "supplied via --max_tokens"}

        ssd, sstats = build_ssd(pool, sampling, n_ssd, ssd_target, engine, args, tok_fn,
                                gated=arm.gated)
        parts += ssd
        sstats.update({"kind": "general_ssd", "n": len(ssd),
                       "sampling": sampling.as_dict(), "max_tokens": args.max_tokens,
                       "backend": engine.backend, "isolates": sampling.isolates})
        section["ssd"] = sstats
        print(f"SSD slot: {len(ssd)} examples, mean {sstats['mean_label_tokens']} label "
              f"tokens, token ratio {sstats['token_match']['ratio_to_target']}")

    # --- write + report -----------------------------------------------------
    assert len(parts) == n_gen, f"slot is {len(parts)}, expected {n_gen}"
    for r in parts:
        assert all(m["role"] != "system" for m in r["messages"]), \
            "replay records must be SI-free (PLAN §11 check 3)"

    counts = chat.label_token_counts(tok_fn, parts)
    total = sum(counts)
    ref_total = reference["total_label_tokens"] if reference else total
    ratio = total / ref_total if ref_total else None
    section.update({
        "n_written": len(parts),
        "total_label_tokens": total,
        "mean_label_tokens": round(total / max(1, len(parts)), 1),
        "reference_total_label_tokens": ref_total,
        "token_ratio_to_A1": round(ratio, 4) if ratio else None,
        "example_ratio_to_A1": round(len(parts) / max(1, reference["n"]), 4) if reference else 1.0,
        "within_token_tolerance": abs(total - ref_total) / ref_total <= 0.05 if ref_total else None,
        "kinds": dict(Counter(r["kind"] for r in parts)),
        "replay_kinds": dict(Counter(r.get("replay_kind") for r in parts)),
    })

    manifest.write_jsonl(slot_path, parts)
    manifest.merge(out_dir, "general_slot", section)
    print(f"\nWrote {len(parts)} replay examples -> {slot_path}")
    if ratio:
        print(f"  label tokens {total} vs A1 reference {ref_total} (ratio {ratio:.4f})")
    if ratio and abs(ratio - 1.0) > 0.05:
        print(f"\n  WARNING: token ratio {ratio:.4f} is outside PLAN §5's +/-5% tolerance.")
        print(f"  Under token-mean loss normalisation this arm's replay pressure per step "
              f"differs from the reference's, so it is a real confound — not a rounding "
              f"issue. Record it and say so when the arms are compared.")
        if arm.name == "A2" and args.token_reference == "a1":
            print("  For A2 this usually means the prompt pool cannot bracket A1's total. "
                  "At --min_gold_words 20 it can (reachable 174k-617k vs a 600k target), "
                  "so first check --min_gold_words and the pool size. Options:\n"
                  "    (a) re-tune --min_gold_words so the pool's reachable range brackets "
                  "the target — `build_prompt_pool.py --scan_only` shows the trade;\n"
                  "    (b) rebuild the SSD arms with --token_reference superni_gold, which "
                  "makes the A2<->A3 paired control exact and leaves A1 as the external "
                  "reference (PLAN §5's own framing);\n"
                  "    (c) keep this and lean on the §6 block layout, which holds the "
                  "replay stream at 25% of every step length-independently — but only if "
                  "probe_loss_norm.py reports micro_batch_mean.")
    if arm.name == "A2" and args.token_reference == "superni_gold":
        print("  A2 is the token reference for this build. Pass --token_reference "
              "superni_gold to every SSD arm too, or the arms will not be comparable.")
    print(f"  manifest -> {out_dir / manifest.MANIFEST_NAME}")


if __name__ == "__main__":
    main()
