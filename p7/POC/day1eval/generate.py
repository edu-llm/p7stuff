"""Run a model as a *tutor* over MRBench and save its generated responses.

Example (on an EC2 GPU box, after ``pip install -r requirements.txt``):

    python generate.py --model olmo --dataset V1
    python generate.py --model qwen --dataset V1 --thinking
    python generate.py --model olmo-instruct --limit 20 --out outputs/olmo_it_smoke.json

Output is a JSON list; each record carries the source dialogue metadata plus the
model's ``generated_response`` so it can be scored / annotated downstream (e.g.
merged back into the MRBench ``anno_llm_responses`` schema).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import config
from config import EngineConfig, GenConfig, MODELS
from data import load_dialogues
from prompts import PROMPT_CONDITIONS, build_messages, render_prompt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MRBench tutor-response generation (vLLM).")
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(MODELS),
                    help="Model key from the registry in config.py.")
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=list(config.DATASETS),
                    help="MRBench split to run over.")
    ap.add_argument("--limit", type=int, default=0, help="Cap #dialogues (0 = all).")
    ap.add_argument("--out", default="", help="Output json path (auto-named if omitted).")

    # Generation
    ap.add_argument("--temperature", type=float, default=GenConfig.temperature)
    ap.add_argument("--top-p", type=float, default=GenConfig.top_p)
    ap.add_argument("--max-tokens", type=int, default=GenConfig.max_tokens)
    ap.add_argument("--seed", type=int, default=GenConfig.seed)

    # Prompting
    sol = ap.add_mutually_exclusive_group()
    sol.add_argument("--include-solution", dest="include_solution", action="store_true",
                     help="Give the tutor the reference solution as context (default).")
    sol.add_argument("--no-solution", dest="include_solution", action="store_false",
                     help="Withhold the reference solution from the tutor.")
    ap.set_defaults(include_solution=config.INCLUDE_SOLUTION)
    ap.add_argument("--condition", default="baseline", choices=list(PROMPT_CONDITIONS),
                    help="Tutor system-prompt condition (baseline vs pedagogical).")
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen3 only: keep <think> reasoning traces on.")

    # Engine
    ap.add_argument("--tp", type=int, default=EngineConfig.tensor_parallel_size,
                    help="tensor_parallel_size (#GPUs).")
    ap.add_argument("--max-model-len", type=int, default=EngineConfig.max_model_len)
    ap.add_argument("--gpu-mem-util", type=float, default=EngineConfig.gpu_memory_utilization)
    ap.add_argument("--dtype", default=EngineConfig.dtype)
    ap.add_argument("--enforce-eager", action="store_true", default=EngineConfig.enforce_eager)

    ap.add_argument("--list-models", action="store_true", help="Print the registry and exit.")
    return ap.parse_args()


def _auto_out_path(model_key: str, dataset: str, condition: str) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset}_{condition}.json")


def main() -> None:
    args = parse_args()

    if args.list_models:
        for key, spec in MODELS.items():
            default = " (default)" if key == config.DEFAULT_MODEL else ""
            print(f"{key:15s} {spec.model_id}{default}\n{'':15s} {spec.notes}")
        return

    spec = MODELS[args.model]
    out_path = args.out or _auto_out_path(args.model, args.dataset, args.condition)
    system_prompt = PROMPT_CONDITIONS[args.condition]

    # Per-model chat-template kwargs, with the Qwen3 thinking override.
    template_kwargs = dict(spec.chat_template_kwargs)
    if "enable_thinking" in template_kwargs and args.thinking:
        template_kwargs["enable_thinking"] = True

    # ---- Load data -------------------------------------------------------- #
    dialogues = load_dialogues(args.dataset, limit=args.limit)
    print(f"[gen] {len(dialogues)} dialogues | model={spec.model_id} | out={out_path}")

    # ---- Heavy imports here so --list-models / --help work without a GPU --- #
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True)
    has_template = spec.has_chat_template
    if has_template is None:
        has_template = bool(getattr(tokenizer, "chat_template", None))

    # ---- Build prompts ---------------------------------------------------- #
    prompts: list[str] = []
    for d in dialogues:
        messages = build_messages(d, include_solution=args.include_solution,
                                  system_prompt=system_prompt)
        prompts.append(render_prompt(tokenizer, messages, template_kwargs, has_template))

    # ---- Engine + sampling ------------------------------------------------ #
    llm = LLM(
        model=spec.model_id,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
        trust_remote_code=EngineConfig.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=GenConfig().stop,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - t0

    # vLLM may reorder internally but returns results aligned to input order.
    records = []
    for d, prompt, out in zip(dialogues, prompts, outputs):
        text = out.outputs[0].text.strip() if out.outputs else ""
        records.append({
            "conversation_id": d.conversation_id,
            "Data": d.data,
            "Split": d.split,
            "Topic": d.topic,
            "conversation_history": d.raw_history,
            "Ground_Truth_Solution": d.ground_truth_solution,
            "generated_response": text,
            "prompt": prompt,
        })

    result = {
        "meta": {
            "model_key": args.model,
            "model_id": spec.model_id,
            "dataset": args.dataset,
            "condition": args.condition,
            "num_dialogues": len(records),
            "include_solution": args.include_solution,
            "thinking": template_kwargs.get("enable_thinking", None),
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
            "elapsed_sec": round(elapsed, 2),
            "python": platform.python_version(),
        },
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"[gen] wrote {len(records)} responses to {out_path} in {elapsed:.1f}s "
          f"({len(records) / elapsed:.1f} resp/s)")
    # Show one sample so the run is glanceable in the terminal.
    if records:
        print("\n--- sample ---")
        print("last student:", dialogues[0].last_student_turn[:200])
        print("tutor (gen):", records[0]["generated_response"][:300])


if __name__ == "__main__":
    sys.exit(main())
