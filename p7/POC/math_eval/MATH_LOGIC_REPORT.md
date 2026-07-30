# Math + Logic Eval — Results (frontier benchmarks, final-answer accuracy)

**Models.** `allenai/OLMo-2-0425-1B-Instruct` (base) vs the pedagogical LoRA-SFT adapter (sft).
**Protocol.** Greedy, pass@1, answers in `\boxed{}`. Grading = verifiable final-answer accuracy
(deterministic for int/MC; LLM-as-verifier for MATH-500 symbolic equivalence). Two arms:
**no-SI** (no system prompt) and **direct-SI** (both models given "solve the problem and give the
final answer inside `\boxed{}`").

## Headline — three arms (where the "give the answer" directive lives)

| Benchmark | nosi base | nosi SFT | directsi base | directsi SFT | userinstr base | userinstr SFT |
|---|---|---|---|---|---|---|
| GSM8K | 47% | 20% | 60% | 7% | 60% | 20% |
| MATH-500 (lvl5) | 12% | 4% | 8% | 0% | 16% | 4% |
| BBH logical-deduction | 13% | 27% | 33% | 13% | 20% | 0% |
| AIME 2024 | 7% | 0% | 7% | 0% | 0% | 7% |
| **Overall** | **19%** | **11%** | **24%** | **4%** | **23%** | **7%** |

## The key test: system-channel vs user-channel directive

Definitions (consistent across arms): **hand-back** = response ends with a question (Socratic
deflection, no answer). **committed** = states a declarative final answer (grader extracted an answer
AND the response does not end in a question) — "is the answer related to 10?" does NOT count.
**acc when committed** = correct ÷ committed.

| arm | model | hand-back | commit | overall | acc when committed |
|---|---|---|---|---|---|
| nosi | base | 0% | 70% | 19% | 27% |
| nosi | sft | 23% | 53% | 11% | 22% |
| directsi | base | 0% | 93% | 24% | 26% |
| directsi | sft | **36%** | 41% | 4% | 10% |
| userinstr | base | 0% | 94% | 23% | 24% |
| userinstr | sft | **20%** | 51% | 7% | 11% |

**Controlled comparison = directsi vs userinstr** (identical directive text, same 1024-token budget,
same boxing hint — only the *channel* differs; `nosi` used earlier wording + 640 tokens so it is not
perfectly comparable):

1. **Channel shortcut confirmed.** The SFT hand-back rate is highest when the directive is a system
   prompt (**36%**) and lowest when it rides in the question (**20%**); base never hands back (0%).
   Moving the directive out of the system channel cuts Socratic deflection nearly in half. Cause:
   100% of pedagogy training rows had a system prompt, 0% of general rows did, so the *presence* of a
   system message cues tutor-mode regardless of its content.
2. **Real residual skill loss.** Whenever SFT does commit a declarative answer, it is right only
   ~10–22% vs base's ~24–27% — roughly half — and its overall accuracy (4–11%) stays far below base
   (19–24%) in every arm. So the fine-tune both (a) gated answering behind the system channel AND
   (b) genuinely weakened the underlying math/logic.

**Answer to "is internal skill worse, separate from not-giving-answers?"** Both. Routing the
instruction around the system prompt reduces the Socratic deflection (much of the behavior is
channel-triggered), but a genuine computation regression remains: even on committed answers SFT is
about half as accurate as base.

## Accuracy (no-SI arm detail)

| Benchmark | difficulty | base | SFT | Δ |
|---|---|---|---|---|
| GSM8K | easy | **7/15 (47%)** | 3/15 (20%) | **−27** |
| MATH-500 (lvl 5) | hard | 3/25 (12%) | 1/25 (4%) | −8 |
| BBH logical-deduction | hard (logic) | 2/15 (13%) | **4/15 (27%)** | **+14** |
| AIME 2024 | very hard | 1/15 (7%) | 0/15 (0%) | −7 |
| **Overall** | | **13/70 (19%)** | **8/70 (11%)** | **−8** |

## This is the forgetting you were looking for

Unlike the general-instruction eval (which came out at parity), **math is where the tutoring SFT
visibly hurts** — overall accuracy drops 19% → 11%, and GSM8K (the one math benchmark a 1B can
actually do) falls by more than half (47% → 20%). Two mechanisms, both traceable to what the model
was trained to do:

1. **The tutor stops committing to a final answer.** The SFT objective was literally "guide, don't
   reveal the answer." On a "solve it and box the answer" benchmark that backfires: SFT produces
   **no final-answer marker on 39/70 items vs 29/70 for base**, and is **more terse** (median 143 vs
   262 words) — it explains a step or two and trails off without landing the number. That's the
   Socratic style bleeding into a setting that demands a committed answer.
2. **Degraded multi-step arithmetic.** Even when it does commit, reasoning is shakier. Example
   (GSM8K, base right / SFT wrong):

```text
Q: A tub of ice cream ($13) is now $11. Milk is discounted $0.50. How much do you SAVE
   buying 2 tubs and 4 packets of milk?   (gold = $6)
base: savings per tub = 13-11 = $2; 2 tubs -> $4; milk 4 x $0.50 -> $2; total $6.   ✓
sft : computes total COST ($70), invents a "discount" of $26, answers $26.          ✗
```
The SFT model misreads *save* as *spend* and loses the thread — a reasoning regression, not just a
formatting one.

## The one place SFT *helps*: logic

BBH logical-deduction went **up** (13% → 27%). These are constraint-satisfaction word puzzles
("X is left of Y, Z is rightmost, who's third?"). The tutor training — read the setup carefully,
work one relation at a time — happens to suit step-by-step deductive puzzles, even as it hurts
numeric computation. So it's not blanket forgetting; it's a **skill re-weighting** away from
arithmetic and toward careful stepwise deduction.

## Caveats

- **1B ceiling + hard sets:** absolute scores on MATH-500-lvl5 and AIME are low for *both* models
  (single digits) — expected even for much larger models; the interpretable signal is GSM8K + the
  base-vs-SFT delta, not the absolute AIME number.
- **Token budget:** `GEN_MAX_NEW=640`. Some long CoTs (esp. base, max ~530 words) approach the cap;
  a higher budget might lift both models slightly, but SFT's *under-committing* is a behavior, not
  only a truncation artifact (it's terser, not longer-and-cut-off).
- n is modest (15–25/benchmark), single greedy sample. Directional, not a leaderboard number.

## Takeaway

Pairing this with the earlier evals gives the complete story:
- **Pedagogy** (`llm_judge/`): SFT+SI wins — the intended skill improved.
- **General instructions** (`GENERAL_EVAL_REPORT.md`): parity — no broad forgetting.
- **Math/logic (here):** a real **trade-off** — arithmetic problem-solving degrades (the model
  learned *not* to just give answers), while stepwise logical deduction improves. This is the
  classic capability trade-off of narrow SFT, and the co-training mix softened but did not fully
  prevent it in the math-computation direction.

## Files
`math_logic_prompts.jsonl` (test set) · `math_logic_results.jsonl` (model outputs) ·
`math_logic_graded.json` (per-item) · `verifier_out_*.json` (MATH equivalence) ·
`grade_math_logic.py` · `build_verify_batches.py` · `MATH_LOGIC_README.md` (how to reproduce).
