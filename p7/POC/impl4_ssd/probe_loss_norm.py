#!/usr/bin/env python
"""PLAN §5 — "write a 2-step probe" for which loss normalisation the Trainer actually uses.

Why this matters. ``transformers>=4.48`` normalises the loss by
``num_items_in_batch`` = total unmasked label tokens across the whole accumulation
group, which makes **stream weight token-proportional, not example-proportional**.
At a fixed 75/25 *example* ratio, arms with different target lengths would then get
different replay pressure per step and the comparison would be silently invalid.

But that fix depends on ``Trainer.model_accepts_loss_kwargs``, resolved by inspecting
the forward signature, and it can fall back to per-micro-batch mean when the model is
PEFT-wrapped. So we measure rather than assume.

Method. Build one accumulation group of ``grad_accum`` micro-batches over two
synthetic streams with deliberately lopsided target lengths (short ≈ 4 label tokens,
long ≈ 120). Compute both candidate reference values by hand from per-token
cross-entropy on the *unmodified* model, then run exactly one optimizer step at
``lr=0`` (so the weights cannot move between the two measurements) and compare the
logged loss against each candidate.

Two details that would otherwise make the comparison meaningless: the probe uses the
same ``SequentialSampler`` as training (a ``RandomSampler`` would scatter short and
long targets across micro-batches, erasing the lopsidedness the probe depends on),
and LoRA dropout is forced to 0 so the two measurements see the same function.

    token_mean       = sum(all token losses) / sum(all label tokens)
    micro_batch_mean = mean over micro-batches of (that micro-batch's token mean)

Usage:
    python probe_loss_norm.py                        # real setup: OLMo-2-1B + LoRA
    python probe_loss_norm.py --arm A3               # also merge into that manifest
    python probe_loss_norm.py --model hf-internal-testing/tiny-random-OlmoForCausalLM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from impl4 import manifest
from impl4.config import (
    ARM_CHOICES,
    BASE_MODEL,
    GRAD_ACCUM,
    LORA_ALPHA,
    LORA_R,
    PER_DEVICE_BATCH,
    SEED,
    resolve_arm,
)
from impl4.paths import run_dir
from impl4.trainer import loss_capture_callback, sequential_trainer_cls

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--arm", default=None, choices=ARM_CHOICES,
                   help="Merge the result into runs/<arm>/manifest.json.")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--out", default=None, help="Also write the raw result here as JSON.")
    p.add_argument("--per_device_batch", type=int, default=PER_DEVICE_BATCH)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--short_tokens", type=int, default=4)
    p.add_argument("--long_tokens", type=int, default=120)
    p.add_argument("--prompt_tokens", type=int, default=16)
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def synth_examples(vocab: int, n: int, prompt_len: int, target_len: int, rng,
                   mode: str = "random") -> list[dict]:
    """A micro-batch with a fixed number of unmasked label tokens.

    ``mode`` controls how *predictable* the targets are, which is what actually
    separates the two candidate normalisations:

        token_mean       = sum(token losses) / sum(label tokens)     [count-weighted]
        micro_batch_mean = mean over micro-batches of their token means  [unweighted]

    Those two are equal whenever every micro-batch has the same mean loss — so a probe
    built from uniformly random ids cannot discriminate at all, no matter how lopsided
    the *lengths* are. Pairing a low-loss stream (``repeat``: a short cycling pattern a
    pretrained model predicts easily) with a high-loss one (``random``) is what makes
    the two values diverge.
    """
    out = []
    for _ in range(n):
        total = prompt_len + target_len
        if mode == "repeat":
            cycle = [int(rng.integers(5, vocab - 5)) for _ in range(4)]
            ids = [cycle[i % 4] for i in range(total)]
        else:
            ids = [int(rng.integers(5, vocab - 5)) for _ in range(total)]
        labels = [IGNORE] * prompt_len + ids[prompt_len:]
        out.append({"input_ids": ids, "labels": labels,
                    "attention_mask": [1] * len(ids)})
    return out


def collate(batch, pad_id):
    width = max(len(x["input_ids"]) for x in batch)
    ii, ll, aa = [], [], []
    for x in batch:
        n = width - len(x["input_ids"])
        ii.append(x["input_ids"] + [pad_id] * n)
        ll.append(x["labels"] + [IGNORE] * n)
        aa.append(x["attention_mask"] + [0] * n)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(aa)}


@torch.no_grad()
def reference_values(model, micro_batches, device) -> dict:
    """Hand-compute both candidate normalisations from per-token cross-entropy."""
    import torch.nn.functional as F

    tot_loss, tot_tokens, per_mb = 0.0, 0, []
    for mb in micro_batches:
        batch = {k: v.to(device) for k, v in mb.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = batch["labels"][:, 1:].reshape(-1)
        mask = shift_labels != IGNORE
        losses = F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="sum")
        n = int(mask.sum())
        tot_loss += float(losses)
        tot_tokens += n
        per_mb.append(float(losses) / max(1, n))
    return {
        "token_mean": tot_loss / max(1, tot_tokens),
        "micro_batch_mean": sum(per_mb) / len(per_mb),
        "total_label_tokens": tot_tokens,
        "per_micro_batch_token_means": [round(x, 6) for x in per_mb],
    }


def main():
    args = parse_args()
    import numpy as np
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    vocab = int(model.config.vocab_size)

    peft_wrapped = False
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                # dropout 0, not the training 0.05: the hand-computed reference and the
                # Trainer's logged loss must see the same deterministic function.
                r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ))
            peft_wrapped = True
        except Exception as e:
            print(f"WARNING: could not PEFT-wrap ({e}); probing the bare model instead.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # One accumulation group: the first half short *and* easy, the second half long
    # *and* hard. Lopsided in both token count and per-token loss, which is the case
    # where the two normalisations disagree the most.
    n_mb = args.grad_accum
    examples: list[dict] = []
    for i in range(n_mb):
        short = i < n_mb // 2
        examples += synth_examples(
            vocab, args.per_device_batch, args.prompt_tokens,
            args.short_tokens if short else args.long_tokens, rng,
            mode="repeat" if short else "random")
    micro_batches = [
        collate(examples[i * args.per_device_batch:(i + 1) * args.per_device_batch],
                tokenizer.pad_token_id)
        for i in range(n_mb)
    ]

    ref = reference_values(model, micro_batches, device)
    spread = abs(ref["token_mean"] - ref["micro_batch_mean"])
    print(f"reference token_mean       = {ref['token_mean']:.6f}")
    print(f"reference micro_batch_mean = {ref['micro_batch_mean']:.6f}  (spread {spread:.6f})")
    if spread < 1e-4:
        print("WARNING: the two candidates are nearly identical, so this probe cannot "
              "discriminate. Widen --short_tokens/--long_tokens.")

    ds = Dataset.from_list(examples)
    logged: list[float] = []

    targs = TrainingArguments(
        output_dir="/tmp/impl4_loss_probe",
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=1,
        learning_rate=0.0,            # weights cannot move between the two measurements
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=args.seed,
        dataloader_drop_last=True,
        bf16=False, fp16=False,
    )
    trainer = sequential_trainer_cls()(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        callbacks=[loss_capture_callback(logged)],
    )
    model.train()
    trainer.train()

    if not logged:
        raise SystemExit("Trainer logged no loss — cannot determine the normalisation.")
    observed = logged[0]
    d_token = abs(observed - ref["token_mean"])
    d_micro = abs(observed - ref["micro_batch_mean"])
    verdict = ("token_mean" if d_token < d_micro else "micro_batch_mean")
    confident = spread > 1e-4 and min(d_token, d_micro) < 0.25 * spread

    result = {
        "verdict": verdict if confident else "inconclusive",
        "observed_logged_loss": observed,
        "reference": ref,
        "abs_diff_to_token_mean": d_token,
        "abs_diff_to_micro_batch_mean": d_micro,
        "candidate_spread": spread,
        "peft_wrapped": peft_wrapped,
        "model": args.model,
        "model_accepts_loss_kwargs": bool(
            getattr(trainer, "model_accepts_loss_kwargs", False)),
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "short_tokens": args.short_tokens,
        "long_tokens": args.long_tokens,
        "transformers": manifest.environment()["transformers"],
        "device": device,
        "interpretation": {
            "token_mean": "Stream weight is TOKEN-proportional. Token-matching the replay "
                          "slot to A1 (PLAN §5) is the binding requirement.",
            "micro_batch_mean": "Stream weight is EXAMPLE-proportional per micro-batch. The "
                                "24/8 block layout (PLAN §6) is the binding requirement and "
                                "gives the replay stream exactly 25% of every step.",
            "inconclusive": "Probe could not discriminate. Widen the token spread and re-run; "
                            "until then both mitigations stay in place (they cost nothing).",
        }[verdict if confident else "inconclusive"],
    }

    print(f"\nobserved logged loss       = {observed:.6f}")
    print(f"  |obs - token_mean|       = {d_token:.6f}")
    print(f"  |obs - micro_batch_mean| = {d_micro:.6f}")
    print(f"VERDICT: {result['verdict']}")
    print(f"  {result['interpretation']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    if args.arm:
        arm = resolve_arm(args.arm)
        d = run_dir(arm.name, args.runs_root)
        manifest.merge(d, "loss_normalization", result)
        print(f"Merged into {d / manifest.MANIFEST_NAME}")


if __name__ == "__main__":
    main()
