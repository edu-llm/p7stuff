# Math + Logic Eval (frontier benchmarks)

Math/logic evaluation of `base` (OLMo-2-1B-**Instruct**) vs your pedagogical **SFT** adapter, using
**real items from frontier benchmarks** and the **standard rubric for these tasks: verifiable
final-answer accuracy** (not a hand-written prompt set or a subjective LLM rubric).

## Test set — `math_logic_prompts.jsonl` (70 items)

| Source | n | Category | Difficulty | Rubric |
|---|---|---|---|---|
| **GSM8K** (`openai/gsm8k`) | 15 | math | easy (anchor) | integer exact-match |
| **MATH-500** level 5 (`HuggingFaceH4/MATH-500`) | 25 | math | hard (competition) | symbolic equivalence |
| **BBH logical-deduction-7** (`lukaemon/bbh`) | 15 | logic | hard | multiple-choice letter |
| **AIME 2024** (`Maxwell-Jia/AIME_2024`) | 15 | math | very hard (olympiad) | integer 0–999 exact-match |

MATH-500 is the OpenAI/o1 competition-math subset; BBH is "Big-Bench Hard"; AIME is the olympiad set
used in current frontier reports. Regenerate deterministically: `python build_math_logic_set.py` (seed=7).

## Three arms (set in the notebook via `ARM`)

The "give the answer" directive is applied identically to both models; the arms differ only in
**which channel** carries it:

1. **`nosi`** → `math_logic_results_nosi.jsonl`. No system prompt; only a minimal "put answer in
   `\boxed{}`" hint in the question. Out-of-the-box behavior.
2. **`directsi`** → `math_logic_results_directsi.jsonl`. Directive is a **system prompt**. NOTE: the
   SFT model learned "system message present ⇒ be Socratic" (100% of pedagogy training rows had a
   system prompt, 0% of general rows did), so this arm *triggers* hand-back and backfires.
3. **`userinstr`** → `math_logic_results_userinstr.jsonl`. Directive rides **inside the question**
   ("Solve it and give the final answer. Put your final answer in `\boxed{}`") — the in-distribution
   channel — to sidestep the system-prompt shortcut.

All arms: greedy pass@1, answers in `\boxed{}`, `GEN_MAX_NEW=1024`.

## Grading = final-answer accuracy (two stages)

1. `grade_math_logic.py <results.jsonl>` — deterministic: integers (GSM8K/AIME) and boxed MC letters
   (BBH) are objective; MATH-500 answers are LaTeX-normalized (sympy-checked if installed). Output
   filenames are **tagged** from the results name (`nosi` / `directsi`) so arms never clobber.
2. MATH-500 answers that aren't a clean auto-match go to a blind **LLM-as-verifier** subagent pass
   (simple-evals/Minerva method) so equivalent-but-differently-written answers (`12\pi` vs `12*pi`)
   aren't under-counted.

## Run it (per arm)

1. Open `math_logic_eval_colab.ipynb` in Colab (GPU), set `SFT_MODEL` and `USE_DIRECT_SI`,
   **Run All** → download the results jsonl, send it back.
2. `python grade_math_logic.py math_logic_results_<arm>.jsonl` → accuracy table; may emit
   `needs_verify_<arm>.json`.
3. If so: `python build_verify_batches.py needs_verify_<arm>.json` → spawn verifier subagents writing
   `verifier_out_<arm>_*.json` → `python grade_math_logic.py math_logic_results_<arm>.jsonl --with-verify`.

## Results so far

- **no-SI** (graded): base **19%** overall vs SFT **11%** — SFT drops on math (GSM8K 47→20) but gains
  on logic (BBH 13→27). See `MATH_LOGIC_REPORT.md`.
- **direct-SI**: the first run (`math_logic_results_directsi_v1_truncated.jsonl`) is **kept only as a
  record** — base was truncated before boxing (768-token cap) and the SI held a literal `(X)`
  placeholder, so base numbers were invalid. The notebook is now fixed (concise SI, no placeholder,
  1024 tokens); re-run to produce a clean `math_logic_results_directsi.jsonl`. Key qualitative result
  already visible: **the SFT model ignores the "give the answer" SI and still asks a question on
  ~55% of math items** (boxed an answer only 3/55) — the Socratic behavior overrides explicit
  instructions.

## Files
- `build_math_logic_set.py` — pulls the 70 benchmark items → `math_logic_prompts.jsonl`.
- `math_logic_eval_colab.ipynb` — Colab inference (both arms) → results jsonl.
- `grade_math_logic.py` — tagged, deterministic + verifier-aware accuracy grader.
- `build_verify_batches.py` — blind verifier tasks for MATH-500 equivalence.
- `math_logic_results_nosi.jsonl`, `math_logic_graded_nosi.json`, `verifier_out_nosi_*.json` — no-SI arm.
- `MATH_LOGIC_REPORT.md` — no-SI findings writeup.
