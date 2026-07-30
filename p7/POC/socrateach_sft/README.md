---
license: odc-by
language:
- en
task_categories:
- text-generation
tags:
- socratic
- tutoring
- education
- math
- sft
- olmo
pretty_name: SocraTeach SFT (Socratic Math Tutor)
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/socrateach_sft_train.jsonl
  - split: validation
    path: data/socrateach_sft_val.jsonl
  - split: test
    path: data/socrateach_sft_test.jsonl
---

# SocraTeach SFT — Socratic Math Tutor

Supervised fine-tuning (SFT) data for turning `allenai/OLMo-2-0425-1B` into a
step-by-step **Socratic math tutor** that guides students one nudge at a time and
never hands over the final answer. Built for the P7 "tutor layer" work.

> **Data mix:** the `train` split is **75% Socratic pedagogy** (22,500 examples) +
> **25% general instructions** (7,500 examples) mixed in from
> [`allenai/tulu-3-sft-olmo-2-mixture-0225`](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225).
> This 25% general "replay" slice is included LearnLM-style / Tülu-3 co-training to
> preserve the base model's general ability (avoid catastrophic forgetting) while
> learning the tutoring behavior.

## Splits

| Split | Records | Composition |
|---|---|---|
| `train` | 30,000 | 22,500 Socratic pedagogy (75%) + 7,500 general Tülu instructions (25%) |
| `validation` | 1,724 | Socratic pedagogy |
| `test` | 1,743 | Socratic pedagogy |

Splits are **grouped by problem**, so no problem leaks across train/validation/test.

## Format

Each record is a chat conversation in the standard `messages` format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a supportive math mentor ..."},
    {"role": "user", "content": "A bulk warehouse is offering 48 cans ..."},
    {"role": "assistant", "content": "How much does each can cost ...?"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

- **System message**: a per-dialogue System Instruction describing the pedagogy that
  dialogue actually practices (Socratic step-by-step always; mistake-handling /
  explanation / pacing / closing moves added only when the dialogue exhibits them).
- **General replay** examples make up exactly **25% of the train split** (7,500 of
  30,000). They are System-Instruction-free, English-only math/code/reasoning
  examples drawn from `allenai/tulu-3-sft-olmo-2-mixture-0225`, mixed in
  LearnLM-style to preserve general ability during co-training.

Recommended loss masking for SFT: compute loss on assistant tokens (+ EOS) only.

## Sources

- Pedagogy dialogues: [`ulises-c/SocraTeach_Multi`](https://huggingface.co/datasets/ulises-c/SocraTeach_Multi)
- General replay data: [`allenai/tulu-3-sft-olmo-2-mixture-0225`](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225)

## Loading

```python
from datasets import load_dataset

ds = load_dataset("meric533/socrateach-sft")
print(ds)
print(ds["train"][0]["messages"])
```
