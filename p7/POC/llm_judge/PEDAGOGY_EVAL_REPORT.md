# Pedagogy eval — SFT OLMo-2-1B-Instruct as a Socratic tutor (LLM-as-judge)

Offline, blind, subagent-based evaluation of the four experimental setups on the held-out
SocraTeach test problems. No external API — the judges are LLM subagents.

## Setups (the 2×2)

| Code | Setup | Model | System instruction |
|------|-------|-------|--------------------|
| A | `A_raw_noSI` | base OLMo-2-1B-Instruct | none |
| B | `B_raw_SI`   | base OLMo-2-1B-Instruct | pedagogical SI |
| C | `C_sft_noSI` | SFT (LoRA)              | none |
| D | `D_sft_SI`   | SFT (LoRA)              | pedagogical SI |

Inputs: `test_results_instruct.jsonl` — 16 unique held-out problems (deduplicated), first tutor
turn per problem. 4 setups × 16 = 64 responses.

## Method (bias controls)

- **Blind.** For each problem the 4 responses are shuffled and relabeled `R1–R4`; the judge
  never sees which setup produced which response. The `rid → setup` map is kept separate
  (`judge_key.json`) and is not shown to judges. Verified: no `raw/sft/noSI/SI/setup` strings
  appear in the judge inputs; the `R#` slot is randomized across setups.
- **No gold leakage.** The judges are **not** shown the dataset's gold tutor turn (it is close
  to the SFT training data). They see only the problem, the correct numeric answer (reference,
  to detect answer-revealing/correctness), and the four candidate responses.
- **Held-out data.** Test problems come from the grouped train/val/test split (split *by
  problem*), so no test problem appears in training.
- **Judges.** 4 subagents, one per batch of 4 problems, scoring independently.

## Rubric (built only from the two agreed sources — not authored ad hoc)

Six dimensions are the **published MRBench** rubric (verbatim, from the teammate's
`tutor-eval-suite`, cell A2); two are from **Joe's P7 rubric** (VanLehn step-level guidance;
Mayer cognitive load). The two MRBench *mistake* dimensions
(`Mistake_Identification`, `Mistake_Location`) are **excluded** because these are opening turns —
the student has not answered yet, so there is no mistake to identify. Each dimension is scored
0 / 0.5 / 1.

| Dimension | Source | 1.0 means |
|-----------|--------|-----------|
| Revealing_of_the_Answer | MRBench | does NOT reveal the final answer |
| Providing_Guidance | MRBench | correct & relevant guidance (hint/question/example) |
| Actionability | MRBench | clear what the student should do next |
| Coherence | MRBench | coherent & consistent with the problem |
| Tutor_Tone | MRBench | encouraging |
| Humanlikeness | MRBench | natural / not robotic |
| Step_Level_Guidance | Joe (VanLehn) | hints at the next single step, not the full solution |
| Load_Aware_Formatting | Joe (Mayer) | brief, one idea, low cognitive load |

OVERALL = unweighted mean of the 8 dimensions (0–1).

## Results (0–1, higher is better; n = 16 per setup)

| Setup | NoReveal | Guidance | Action | Coher | Tone | Human | StepLvl | LoadFmt | **OVERALL** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A — raw, no SI | 0.06 | 0.84 | 0.12 | 0.84 | 0.50 | 0.56 | 0.00 | 0.12 | **0.38** |
| B — raw, +SI   | 0.75 | 0.69 | 0.62 | 0.94 | 0.53 | 0.88 | 0.56 | 0.72 | **0.71** |
| C — SFT, no SI | 0.25 | 0.91 | 0.34 | 0.91 | 0.56 | 0.72 | 0.19 | 0.31 | **0.52** |
| D — SFT, +SI   | 1.00 | 0.66 | 0.88 | 0.94 | 0.50 | 1.00 | 0.78 | 1.00 | **0.84** |

**Ranking: D (0.84) > B (0.71) > C (0.52) > A (0.38).**

## Interpretation

- **The system instruction is the behavioral switch, and SFT amplifies it.** OVERALL rises
  A→B (0.38→0.71, add SI to raw) and C→D (0.52→0.84, add SI to SFT). SFT adds on top of the SI
  (B→D: +0.13) and shifts no-SI behavior only mildly (A→C: +0.14).
- **D is the deployment config.** It maxes the pedagogical dimensions: **never reveals the
  answer (1.00)**, best **step-level guidance (0.78)**, **load-aware formatting (1.00)**, high
  **actionability (0.88)** and **humanlikeness (1.00)**.
- **Gating holds.** Without an SI the models act like solvers: A and C reveal the answer often
  (NoReveal 0.06 / 0.25) and score low on step-level guidance — i.e. pedagogy is tied to the SI,
  not baked in unconditionally (the co-training goal).
- **Caveat on `Providing_Guidance`.** This dimension is *highest for the solvers* (A 0.84,
  C 0.91) because it rewards correct, relevant content regardless of whether it is Socratic — a
  full worked solution is "guidance." Read it alongside `Step_Level_Guidance` and
  `Revealing_of_the_Answer`, which capture the *Socratic* quality and cleanly favor D.
- **`Tutor_Tone` did not differentiate** (~0.5 everywhere): the short one-line questions read as
  neutral rather than warm, so tone is a wash across setups.

## Limitations

- **Small sample:** n = 16 problems, first tutor turn only. Treat as a directional signal; no
  confidence intervals computed here.
- **Single judge model family** (subagents share one model). Blind + a source-grounded rubric
  reduce bias, but this is not a human-validated (Cohen's κ) judgment. The teammate's
  `tutor-eval-suite` provides the κ-validated version and should be run for the headline number.
- **Benchmark familiarity:** GSM8K/MAWPS are public and may be in the base model's pretraining;
  this affects all four setups equally, so it does not bias the relative comparison.

## Artifacts (in `llm_judge/`)

- `build_batches.py` — builds blind, gold-free batches from `../test_results_instruct.jsonl`.
- `judge_batch_*.json` — exact inputs shown to the judges (no setup labels, no gold turn).
- `judge_out_*.json` — per-response labels + one-line rationales from the judges.
- `judge_key.json` — `rid → setup` map (kept from judges).
- `aggregate.py` — maps labels to scores and produces the table.
- `judge_summary.json` — machine-readable summary.
