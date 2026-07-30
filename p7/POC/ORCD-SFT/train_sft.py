#!/usr/bin/env python
"""SFT OLMo-2-1B(-Instruct) into a step-level Socratic math tutor on MIT ORCD.

This is a direct, cluster-friendly port of ``olmo2_1b_sft_colab.ipynb`` (cells 4-16):
load the prepared JSONL files -> LoRA -> assistant-only loss masking -> Trainer -> save.
Test-result generation (notebook cells 18-20) lives in ``generate_test_results.py``.

Method (LearnLM-style, see REPORT.md): per-dialogue System Instructions on the pedagogy
data + SI-free general "replay" data, already mixed into socrateach_sft_train.jsonl.

Only real change vs. the base-model run: ``--base_model`` defaults to the Instruct
checkpoint and the output dir / results file are tagged accordingly.

Example (full run, 1x L40S/A100):
    python train_sft.py --start_from instruct
Smoke test:
    python train_sft.py --start_from instruct --poc
Resume an interrupted run (e.g. after the 6h partition limit):
    python train_sft.py --start_from instruct --resume auto
"""
import argparse
import json
import os
import random

import numpy as np
import torch

IGNORE = -100

_MODELS = {
    "base": "allenai/OLMo-2-0425-1B",
    "instruct": "allenai/OLMo-2-0425-1B-Instruct",
}
# Chat template source (OLMo-2 Tulu-style: <|system|>/<|user|>/<|assistant|>, BOS=EOS).
TEMPLATE_SRC = "allenai/OLMo-2-0425-1B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start_from", choices=["base", "instruct"], default="instruct",
                   help="Which OLMo-2 checkpoint to fine-tune. 'instruct' is the ORCD run.")
    p.add_argument("--base_model", default=None, help="Override the HF model id (defaults from --start_from).")
    p.add_argument("--data_dir", default="data", help="Dir with socrateach_sft_{train,val,test}.jsonl.")
    p.add_argument("--output_dir", default=None, help="Where to save the adapter (default: olmo2-1b-socratic-tutor-<start_from>).")

    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--num_epochs", type=float, default=1.0)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--full_finetune", dest="use_lora", action="store_false",
                   help="Full fine-tune instead of LoRA (needs a bigger GPU).")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization. Defaults follow the handoff (effective batch 32, checkpointing ON).
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=None, help="Default 2e-4 (LoRA) / 1e-5 (full).")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--no_grad_checkpointing", dest="grad_checkpointing", action="store_false", default=True,
                   help="Turn OFF gradient checkpointing (only safe on >40GB GPUs).")

    # Eval / logging / checkpointing
    p.add_argument("--eval_cap", type=int, default=200, help="Max examples for the in-loop eval loss.")
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=2)

    # Data sizing. Full run uses everything in the file (already the final 30k mix).
    p.add_argument("--train_total", type=int, default=0,
                   help="Cap on train examples (0 = use the whole file, which is already the 30k mix).")
    p.add_argument("--poc", action="store_true", help="Quick smoke test: small train cap + 1 epoch.")

    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint, or 'auto' to resume from the latest one in output_dir.")
    return p.parse_args()


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_datasets(args):
    from datasets import Dataset

    need = {k: os.path.join(args.data_dir, f"socrateach_sft_{k}.jsonl") for k in ["train", "val", "test"]}
    missing = [p for p in need.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing prepared data files: " + ", ".join(missing) +
            "\nTransfer them to the cluster (scp/rsync) or regenerate with prepare_socrateach_sft.py."
        )

    train_recs = load_records(need["train"])
    eval_recs = load_records(need["val"])

    # The train file is an already-shuffled pedagogy+general mix; a prefix preserves the
    # 75/25 ratio. Only cap when explicitly requested (or in POC).
    cap = args.train_total
    if args.poc and cap == 0:
        cap = 4000
    if cap and len(train_recs) > cap:
        random.Random(args.seed).shuffle(train_recs)
        train_recs = train_recs[:cap]

    kinds = {}
    for e in train_recs:
        k = e.get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1

    train_ds = Dataset.from_list([{"messages": e["messages"]} for e in train_recs])
    eval_ds = Dataset.from_list([{"messages": e["messages"]} for e in eval_recs])
    print(f"Loaded data from '{args.data_dir}/': train={len(train_ds)} {kinds} | eval(val)={len(eval_ds)}")
    return train_ds, eval_ds


def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    print(f"base_model={args.base_model} | bf16={bf16} fp16={fp16}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.chat_template is None:  # Instruct already has it; base does not.
        tokenizer.chat_template = AutoTokenizer.from_pretrained(TEMPLATE_SRC).chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model.config.use_cache = False  # required with gradient checkpointing

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    return model, tokenizer, bf16, fp16


def make_tokenize_fn(tokenizer, max_len):
    """Assistant-only loss masking: labels=-100 on system/user and the <|assistant|> header;
    loss only on assistant content + EOS. Mirrors the notebook exactly."""
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"]

    def enc(s):
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    def tokenize_conversation(example):
        ids = [tokenizer.bos_token_id]
        labels = [IGNORE]
        for m in example["messages"]:
            role, content = m["role"], m["content"]
            if role == "assistant":
                head = enc("<|assistant|>\n")
                body = enc(content) + [tokenizer.eos_token_id]
                ids += head + body + nl
                labels += [IGNORE] * len(head) + body + [IGNORE] * len(nl)
            else:
                tag = "<|system|>\n" if role == "system" else "<|user|>\n"
                seg = enc(tag + content + "\n")
                ids += seg
                labels += [IGNORE] * len(seg)
        return {
            "input_ids": ids[:max_len],
            "labels": labels[:max_len],
            "attention_mask": [1] * len(ids[:max_len]),
        }

    return tokenize_conversation


def main():
    args = parse_args()
    if args.base_model is None:
        args.base_model = _MODELS[args.start_from]
    if args.output_dir is None:
        args.output_dir = f"olmo2-1b-socratic-tutor-{args.start_from}"
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.use_lora else 1e-5
    if args.poc:
        args.num_epochs = min(args.num_epochs, 1.0)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"SFT run: start_from={args.start_from} -> output_dir={args.output_dir}")
    print(f"lora={args.use_lora} lr={args.learning_rate} per_device={args.per_device_batch} "
          f"grad_accum={args.grad_accum} epochs={args.num_epochs} max_len={args.max_len} "
          f"grad_ckpt={args.grad_checkpointing}")
    print("=" * 70)

    train_ds, eval_ds = build_datasets(args)
    model, tokenizer, bf16, fp16 = load_model_and_tokenizer(args)

    tok_fn = make_tokenize_fn(tokenizer, args.max_len)
    train_tok = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="tok train")
    eval_tok = eval_ds.map(tok_fn, remove_columns=eval_ds.column_names, desc="tok eval")
    train_tok = train_tok.filter(lambda x: any(t != IGNORE for t in x["labels"]))
    eval_tok = eval_tok.filter(lambda x: any(t != IGNORE for t in x["labels"]))
    if len(eval_tok) > args.eval_cap:  # keep in-loop eval cheap
        eval_tok = eval_tok.shuffle(seed=args.seed).select(range(args.eval_cap))

    lens = [len(x) for x in train_tok["input_ids"]]
    print(f"train={len(train_tok)} eval={len(eval_tok)} | tokens mean {np.mean(lens):.0f} "
          f"p95 {int(np.percentile(lens, 95))} max {max(lens)}")

    from transformers import Trainer, TrainingArguments

    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        pad = tokenizer.pad_token_id
        ii, ll, aa = [], [], []
        for x in batch:
            n = maxlen - len(x["input_ids"])
            ii.append(x["input_ids"] + [pad] * n)
            ll.append(x["labels"] + [IGNORE] * n)
            aa.append(x["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ii),
            "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(aa),
        }

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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=train_tok, eval_dataset=eval_tok, data_collator=collate,
    )

    resume = args.resume
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        print(f"Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter/model + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
