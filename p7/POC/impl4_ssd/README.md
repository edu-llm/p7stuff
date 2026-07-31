# Impl 4 — Self-distilled replay (team overview)
## What changes

| | Impl 2 | Impl 4 |
|---|---|---|
| 75% pedagogy (SocraTeach + system instruction) | gold | **unchanged** |
| 25% replay (no system instruction) | Tülu-3 gold | **model's own output on SuperNI prompts** |
| LoRA / masking / LR / 1 epoch | — | **unchanged** |

**Set expectations:** we're changing 25% of the data, on prompts where we weren't trying to
change behavior anyway. 

We'll also run a control with just SuperNI gold replacing Tulu-3 gold.

## Steps

1. **Get the prompt pool.** Super-NaturalInstructions, English training tasks only.
2. **Clean it.** Drop anything sourced from BIG-Bench, GSM8K, MATH, or AIME — the eval team
   grades BBH, and BBH is part of BIG-Bench. Then n-gram check against their actual eval
   prompts. Contaminating the evals would invalidate the whole experiment.
3. **Filter to long-answer tasks.** Keep tasks whose gold answers average ≥30 words. Reason in
   "the control" below.
4. **Generate.** One sample per prompt from `OLMo-2-0425-1B-Instruct`, at each of the four
   sampling settings we're comparing (see "Two things worth knowing"). No filtering for quality
   (only drop empty/gibberish-repetition outputs).
5. **Mix and order.** 75/25, matched on *token* count (not just example count), pre-ordered so
   every optimizer step contains both streams rather than getting them randomly.
6. **Train and save.**
7. **Hand off** checkpoints + data + manifest.

**We're testing the sampling temperature, not assuming it.** Two settings are defensible and
we don't know which wins:

- **Hold temperature at 1.0, no truncation.** Then the target *is* what the model would say,
  the gradient goes to ~zero, and the replay slot adds almost no drift. Maximum anchoring.
- **Tune it like the SSD paper did** (Apple, arXiv:2604.01193 — hotter sampling plus top-k/top-p
  truncation). 2 reasons this might win: (1) an anchor that exerts *no* force isn't an
  anchor — a fully inert 25% means the pedagogy data dominates every step; (2) a 1B model at
  temperature 1.0 with no truncation generates a lot of junk, and truncation removes it;

So we sweep four settings (see the runs table). If the truncated arms forget *less* while
sitting at *higher* KL, that's a bigger finding than the one we're chasing — it would mean KL
isn't really what causes forgetting.

**Save checkpoints early and densely.** Almost all the forgetting happens in the first ~20
steps — math drops 20% → 11% by step 20. A regular every-100-steps grid would miss the entire
effect.

## The control

Swapping Tülu → SuperNI changes two things at once: the *prompts* and the *targets*. So we run
SuperNI with its own **gold** answers too. That isolates "self-generated vs. gold" from "we
changed datasets."

This is also why step 4 filters to long-answer tasks: SuperNI's gold answers are often a single
word, so a gold arm would carry almost no training weight and wouldn't be a fair control.
Filtering makes the gold and self-generated arms directly comparable.

## Runs

| Arm | Replay data | Question it answers |
|---|---|---|
| `A1` | Tülu-3 gold | baseline — this is Impl 2 **DONE** | 
| `A2` | SuperNI gold | is it the prompts or the self-generation? |
| `A3` | SuperNI self-generated | **the actual idea** |
| `A4` | half and half | how much do you need? |
| `T2` | self-generated, temp 1.0 **+ truncation** | does truncation alone help? |
| `T3` | self-generated, temp 1.3 + truncation | trend, or a peak in the middle? |
| `T4` | self-generated, temp 1.6 + truncation | does the paper's tuned setting beat 1.0? |
| `B2` | self-generated, quality-gated | does checking the output help or hurt? |

`A3` doubles as the temperature grid's first point, so that's 8 runs, ~5.5 GPU-hours. `A1` has
to be re-run — the existing `curve_run` checkpoints are at different steps and won't pair up.

`T2` is the one not to cut. Without it, if `T4` wins we can't tell whether it was the higher
temperature or just the truncation cleaning up junk.
