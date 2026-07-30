# General-instruction eval — base Instruct vs SFT (forgetting check)

Does the SFT model still handle **general, non-pedagogy instructions** as well as the base
Instruct model? (The co-training was meant to prevent regression here.) No API key needed —
inference runs in Colab, scoring is done by blind subagent judges offline.

## Files

| File | What it is |
|------|------------|
| `general_prompts.jsonl` | 36 general prompts across 12 categories (QA, math, code, summarization, reasoning, creative, rewriting, extraction, how-to, explanation, constraints, planning). |
| `general_eval_colab.ipynb` | Colab: loads base + your SFT adapter, generates for both (no system prompt), saves `general_eval_results.jsonl`. Run-all. |
| `judge_build.py` | Turns results into **blind pairwise** batches (base/SFT anonymized as X/Y). |
| `judge_aggregate.py` | Maps blind verdicts back and reports SFT win-rate + mean quality. |

## Workflow

1. **Colab:** open `general_eval_colab.ipynb`, GPU runtime, confirm `SFT_MODEL` points at your
   adapter checkpoint (default: the Drive path `.../checkpoint-923`). Run All. Download
   `general_eval_results.jsonl` (also backed up to Drive).
2. **Send `general_eval_results.jsonl` back** to the assistant.
3. **Offline scoring (assistant):**
   ```bash
   python judge_build.py general_eval_results.jsonl 4   # -> judge_batch_*.json + judge_key.json
   # assistant spawns 4 judge subagents; each writes judge_out_<k>.json
   python judge_aggregate.py                             # -> SFT win-rate + quality
   ```

## What "good" looks like

SFT should be **at parity** with base on general tasks — SFT win-rate ≈ 0.5 (ties count 0.5).
A win-rate materially below ~0.45 means SFT degraded general ability (forgetting); at/above 0.5
means the co-training preserved (or improved) it.

## Judge rubric (per pair)

Which response better **follows the instruction** and is more **correct, helpful, and
appropriately formatted**? Verdict `X` / `Y` / `tie`, plus an absolute 0–10 quality score for
each and a one-line rationale. Judges see the two responses **blind** (randomized order), so
there is no position or identity bias toward either model.
