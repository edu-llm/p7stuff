#!/usr/bin/env python
"""Step 4 (PLAN §8.4) — the Impl 4 trainer.

Rather than editing ``ORCD-SFT/train_sft.py`` in place, this script *imports* it and
reuses ``make_tokenize_fn`` and ``load_model_and_tokenizer`` unchanged. That is the
strongest available guarantee that §2.2 (per-dialogue SI), §2.4 (assistant-only
masking) and §2.6 (LoRA r=16/α=32, cosine, warmup 0.03, LR 2e-4, 1 epoch,
max_len=1024) are untouched — they are literally the same code objects, and Impl 2 /
``curve_run`` stay reproducible.

Four things differ from Impl 2, all required by the plan:

1. ``SequentialSampler`` instead of the ``Trainer`` default ``RandomSampler``, so the
   24-pedagogy/8-general block layout written by ``mix_and_order.py`` survives into
   the micro-batches (PLAN §6).
2. ``dataloader_drop_last=True``, which keeps block alignment at the tail.
3. A ``TrainerCallback`` that saves a PEFT adapter at the dense early grid
   {5,10,20,40,80,160,320,480,640,800,937} (PLAN §7). Forgetting is concentrated in
   the first ~20 steps, which a uniform ``save_steps`` grid misses entirely. This is
   separate from HF's own ``save_strategy``, which stays coarse and exists only so a
   6h-partition timeout can resume.
4. ``--arm``, for output naming and manifest bookkeeping.

Usage:
    python train_sft_impl4.py --arm A3
    python train_sft_impl4.py --arm A3 --poc
    python train_sft_impl4.py --arm A3 --resume auto
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from impl4 import manifest
from impl4.chat import impl2_trainer_module
from impl4.trainer import checkpoint_grid_callback, sequential_trainer_cls
from impl4.config import (
    ALL_ARMS,
    ARM_CHOICES,
    BASE_MODEL,
    GRAD_ACCUM,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_LEN,
    NUM_EPOCHS,
    PER_DEVICE_BATCH,
    SEED,
    WARMUP_RATIO,
    checkpoint_grid,
    priority_checkpoints,
    resolve_arm,
)
from impl4.paths import run_dir

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES,
                   help=f"One of {', '.join(ALL_ARMS)} (T1 is an alias of A3).")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--data_dir", default=None, help="Defaults to runs/<arm>/.")
    p.add_argument("--output_dir", default=None, help="Defaults to runs/<arm>/.")
    p.add_argument("--base_model", default=BASE_MODEL)

    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)

    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--full_finetune", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=LORA_R)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)

    p.add_argument("--per_device_batch", type=int, default=PER_DEVICE_BATCH)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--no_grad_checkpointing", dest="grad_checkpointing",
                   action="store_false", default=True)

    p.add_argument("--eval_cap", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=300,
                   help="HF's own checkpoints — for RESUME ONLY. The eval grid is the "
                        "callback's job; don't conflate them.")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=20)

    p.add_argument("--poc", action="store_true",
                   help="Smoke run against a --poc mix (63 blocks) with the POC grid.")
    p.add_argument("--resume", default=None, help="Checkpoint path, or 'auto'.")
    return p.parse_args()


# ---------------------------------------------------------------------------
def build_ordered_datasets(args, block: int):
    """Load the ordered mix **without reshuffling or capping** — order is the point."""
    from datasets import Dataset

    ref = impl2_trainer_module()
    need = {k: os.path.join(args.data_dir, f"socrateach_sft_{k}.jsonl")
            for k in ("train", "val", "test")}
    missing = [p for p in need.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing prepared data files: " + ", ".join(missing) +
            f"\nRun: python mix_and_order.py --arm {args.arm}"
        )

    train_recs = ref.load_records(need["train"])
    eval_recs = ref.load_records(need["val"])
    if len(train_recs) % block:
        raise ValueError(
            f"train file has {len(train_recs)} examples, not a whole number of "
            f"{block}-example blocks — the §6 layout would be broken")

    kinds: dict[str, int] = {}
    for e in train_recs:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
    print(f"Loaded '{args.data_dir}': train={len(train_recs)} {kinds} | val={len(eval_recs)}")

    train_ds = Dataset.from_list([{"messages": e["messages"]} for e in train_recs])
    eval_ds = Dataset.from_list([{"messages": e["messages"]} for e in eval_recs])
    return train_ds, eval_ds, kinds


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    args.arm = arm.name
    out = Path(args.output_dir) if args.output_dir else run_dir(arm.name, args.runs_root)
    args.output_dir = str(out)
    args.data_dir = args.data_dir or str(out)
    out.mkdir(parents=True, exist_ok=True)

    block = args.per_device_batch * args.grad_accum
    grid = checkpoint_grid(args.poc)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 74)
    print(f"Impl 4 arm {arm.name} (block {arm.block}) | sigma={arm.sigma} delta=0 | "
          f"gated={arm.gated}")
    print(f"  sampling: {arm.sampling_config.as_dict() if arm.sampling_config else 'n/a'}")
    print(f"  out={out}  data={args.data_dir}")
    print(f"  lora={args.use_lora} lr={args.learning_rate} "
          f"per_device={args.per_device_batch} grad_accum={args.grad_accum} "
          f"(block={block}) epochs={args.num_epochs} max_len={args.max_len}")
    print(f"  checkpoint grid: {list(grid)}")
    print("=" * 74)

    ref = impl2_trainer_module()
    train_ds, eval_ds, kinds = build_ordered_datasets(args, block)
    model, tokenizer, bf16, fp16 = ref.load_model_and_tokenizer(args)

    tok_fn = ref.make_tokenize_fn(tokenizer, args.max_len)
    train_tok = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="tok train")
    eval_tok = eval_ds.map(tok_fn, remove_columns=eval_ds.column_names, desc="tok eval")

    # A `filter` here would silently break block alignment, so assert instead of filtering.
    n_empty, lens = 0, []
    for row in train_tok:
        lens.append(len(row["input_ids"]))
        if all(t == IGNORE for t in row["labels"]):
            n_empty += 1
    if n_empty:
        raise ValueError(
            f"{n_empty} train examples have no unmasked labels; dropping them would break "
            f"the 24/8 block layout. Fix them in the mix, not here.")
    eval_tok = eval_tok.filter(lambda x: any(t != IGNORE for t in x["labels"]))
    if len(eval_tok) > args.eval_cap:
        eval_tok = eval_tok.shuffle(seed=args.seed).select(range(args.eval_cap))

    n_steps = len(train_tok) // block
    print(f"train={len(train_tok)} eval={len(eval_tok)} | tokens mean {np.mean(lens):.0f} "
          f"p95 {int(np.percentile(lens, 95))} max {max(lens)} | optimizer steps {n_steps}")
    off_grid = [s for s in grid if s > n_steps]
    if off_grid:
        print(f"NOTE: grid points {off_grid} exceed the {n_steps} steps this run will take "
              f"and will not be written.")

    from transformers import TrainingArguments

    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        pad = tokenizer.pad_token_id
        ii, ll, aa = [], [], []
        for x in batch:
            n = maxlen - len(x["input_ids"])
            ii.append(x["input_ids"] + [pad] * n)
            ll.append(x["labels"] + [IGNORE] * n)
            aa.append(x["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(aa)}

    # `group_by_length` reorders the dataset by length and would destroy the §6 block
    # layout. It defaults to False, and transformers 5.x removed the argument outright,
    # so pass it only where it exists rather than crashing on newer versions. Either
    # way the SequentialSampler override below is what actually guarantees the order.
    extra_ta = {}
    if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
        extra_ta["group_by_length"] = False

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,          # resume only
        save_total_limit=args.save_total_limit,
        bf16=bf16, fp16=fp16,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        dataloader_drop_last=True,           # PLAN §6 — keeps block alignment at the tail
        **extra_ta,
    )

    cb, helper = checkpoint_grid_callback(args.output_dir, grid)
    trainer = sequential_trainer_cls()(
        model=model, args=train_args,
        train_dataset=train_tok, eval_dataset=eval_tok, data_collator=collate,
        callbacks=[cb],
    )

    from torch.utils.data import SequentialSampler
    sampler = trainer._get_train_sampler(train_tok)
    assert isinstance(sampler, SequentialSampler), \
        "train sampler is not SequentialSampler — the §6 block layout would be shuffled away"

    resume = args.resume
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        print(f"Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Report what is on disk, not just what this process wrote: after a resume (the
    # normal path under the 6h ORCD partition limit) the recorder only knows about the
    # steps it replayed, and the manifest would under-report the grid.
    on_disk = {int(p.name.split("-")[1]) for p in out.glob("ckpt-*")
               if p.is_dir() and any(p.glob("adapter_model*"))}
    saved = sorted(on_disk | set(helper.saved))
    index = {
        "arm": arm.name,
        "checkpoint_grid": list(grid),
        "checkpoints_saved": saved,
        "checkpoints_written_this_run": sorted(set(helper.saved)),
        "priority_checkpoints": list(priority_checkpoints(arm, args.poc)),
        "steps": n_steps,
        "adapter_dirs": {str(s): f"ckpt-{s}" for s in saved},
    }
    (out / "checkpoint_index.json").write_text(json.dumps(index, indent=2) + "\n",
                                               encoding="utf-8")
    manifest.merge(out, "training", {
        **index,
        "base_model": args.base_model,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "enabled": args.use_lora},
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "num_epochs": args.num_epochs,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": block,
        "max_len": args.max_len,
        "seed": args.seed,
        "sampler": "SequentialSampler",
        "dataloader_drop_last": True,
        "group_by_length": False,
        "train_kinds": kinds,
        "n_train": len(train_tok),
    })

    print(f"\nSaved adapter + tokenizer to {out}")
    print(f"Grid checkpoints on disk: {saved}"
          + (f" (this run wrote {sorted(set(helper.saved))})"
             if set(helper.saved) != set(saved) else ""))
    missing = [s for s in grid if s <= n_steps and s not in saved]
    if missing:
        print(f"WARNING: grid steps {missing} were NOT written — re-running to recover a "
              f"checkpoint costs far more than disk, so investigate before handing off.")


if __name__ == "__main__":
    main()
