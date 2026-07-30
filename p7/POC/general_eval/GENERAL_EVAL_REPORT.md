# General-Instruction Evaluation — Catastrophic-Forgetting Check

**Question.** Did pedagogical SFT (on SocraTeach + Tulu-3 co-training mix) damage the model's
*general* instruction-following ability?

**Models.** `allenai/OLMo-2-0425-1B-Instruct` (base) vs the LoRA-SFT tutor adapter on top of it (sft).
Both generated **without any system prompt** (`NO_SYSTEM_PROMPT=True`), so this measures default,
out-of-the-box behavior — the regime where forgetting would show up.

**Prompts.** 36 non-pedagogy prompts across 12 categories (factual QA, math, code, summarization,
reasoning, creative, rewriting, extraction, how-to, explanation, format constraints, planning).

## Method (MT-Bench protocol, blind, no API key)

Judging done by 4 spawned LLM-judge subagents over 144 blind tasks. Model identities and
presentation order were stripped from every task and only re-mapped at aggregation.

1. **Single-answer grading** — each response rated **1–10** in isolation (helpfulness, relevance,
   accuracy, depth, creativity, detail). 72 tasks (2 responses × 36).
2. **Pairwise, position-swap controlled** — each pair judged in **both** orders; a model "wins" a
   prompt only if it wins in *both* orderings, otherwise it's a tie (this neutralizes position bias
   and is the conservative MT-Bench convention). 72 tasks (2 orders × 36). Reported as an
   AlpacaEval-style win-rate with ties = 0.5.

## Results

| Metric | base | SFT |
|---|---|---|
| Single-answer mean (1–10) | **6.75** | **6.75** |
| Pairwise win-rate vs base (ties=0.5) | 0.50 (ref) | **0.472** |
| Pairwise record (win/tie/loss for SFT) | — | 9 / 16 / 11 |
| Mean response length (words) | 74 | 33 |

**Per-category single-answer (base → sft):**

| Category | base → sft | | Category | base → sft |
|---|---|---|---|---|
| code | 9.7 → 8.7 | | howto | 6.7 → 6.7 |
| constraint | 4.3 → 4.3 | | math_solve | 8.3 → 8.3 |
| creative | 5.3 → **6.3** | | planning | 6.3 → 6.0 |
| explanation | 8.0 → 7.7 | | reasoning | 4.0 → 4.3 |
| extraction | 5.3 → **4.3** | | rewriting | 6.0 → **7.7** |
| factual_qa | 9.3 → 9.3 | | summarization | 7.7 → 7.3 |

## Interpretation

- **No catastrophic forgetting.** Single-answer quality is *identical* (6.75 vs 6.75) and the
  pairwise win-rate (0.472) is within noise of parity (0.50) at n=36. The co-training mix (25%
  general Tulu-3 data with no system instruction) did its job: general instruction-following
  survived the pedagogical fine-tune.
- **SFT is much more concise** (33 vs 74 words) yet loses no quality — consistent with the terse,
  one-idea-per-turn tutoring style bleeding into general answers. This is a *style* shift, not a
  *capability* loss.
- **Category movement is small and mixed.** Slight dips on `code`/`extraction` (where verbosity /
  full enumeration helps) are offset by gains on `creative`/`rewriting`. Nothing collapses.
- The low absolute scores on `constraint` and `reasoning` (~4/10) are shared by *both* models —
  they reflect the 1B base model's ceiling, not anything SFT did.

## Limitations

- n=36, single judge model family, one generation per prompt (greedy/default decoding).
- Single-answer grading and pairwise were scored by the same subagents; treat both as directional,
  corroborating signals rather than independent measurements.
- No-system-prompt only; behavior *with* a general assistant SI was not separately tested here.

## Files

- `general_prompts.jsonl` — the 36 prompts.
- `general_eval_colab.ipynb` — Colab inference (base + SFT) → `general_eval_results.jsonl`.
- `judge_build.py` / `judge_aggregate.py` — MT-Bench task builder + aggregator.
- `judge_batch_*.json` (blind inputs) · `judge_out_*.json` (subagent verdicts) · `judge_key.json`
  (identity map) · `judge_summary.json` (machine-readable summary).
