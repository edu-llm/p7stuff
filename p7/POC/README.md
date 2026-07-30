# P7 — The Tutor Layer ("Rosenshine at the Interface")

Turning an instruction-following model (`allenai/OLMo-2-0425-1B`) into a step-level
Socratic tutor at the inference/post-training layer, and studying the
**KL ↔ forgetting** trade-off it induces.

Full spec, recipes, and measured results are in **[`p7_PRD.md`](p7_PRD.md)**. This
README is just the map of the directory.

## Implementations

1. **Prompting** — inference-time behavior via a system instruction (the verbatim
   prompt is in `p7_PRD.md` §1).
2. **SI-conditioned SFT** — fine-tune on pedagogically-labeled conversations, with a
   per-dialogue System Instruction and a 25% general co-training (replay) mix.
   2.5. **Low-KL / forgetting-aware SFT (SDFT)** — same as (2) but reduces the
   math/logic forgetting it causes by lowering new-task KL from the base model.
3. **RLHF** — not implemented yet (recipe not worked out).

Cross-cutting: every result is reported **both** with and without the pedagogy
System Instruction (SI/no-SI gating), and every training run saves ≳10 checkpoints
that are each evaluated *and* KL-measured.

## Directory map

| Path | What it is |
|------|-----------|
| `p7_PRD.md` | The product/requirements doc — the spec + appendices with our measured results |
| `socrateach_sft/` | SFT data prep (`prepare_socrateach_sft.py`), Colab training notebooks, `GUIDE.md`, `REPORT.md` |
| `ORCD-SFT/` | SFT training on the MIT ORCD Slurm cluster (`train_sft.py`, `run_sft.sbatch`, `ORCD_RUN.md`) |
| `llm_judge/` | Pedagogy quality eval — blind LLM-as-judge, 8-dim rubric (`PEDAGOGY_EVAL_REPORT.md`) |
| `math_eval/` | Old-task retention — math/logic final-answer accuracy (`MATH_LOGIC_REPORT.md`) |
| `general_eval/` | General instruction-following — MT-Bench-style, no system prompt (`GENERAL_EVAL_REPORT.md`) |
| `tutor-eval-suite/` | Additional tutor-eval scaffolding |
| `kl_analysis/` | Forward-KL demo / setup (`kl_forgetting_demo_colab.ipynb`) |
| `curve_run/` | The KL↔forgetting POC: per-checkpoint KL, math grading, pedagogy judging, figures (`Report_KL_POC.md`) |
| `joe_rubric.txt`, `test_results_instruct.jsonl` | Rubric text and an instruct-model test-results sample |
| `EDULLM-NEW-WORKSPACE-SETUP.md` | Bootstrap guide for reconstructing the ORCD/GitHub/W&B workspace |

## Headline result (KL POC — RL's Razor)

Across SFT checkpoints, old-task (math) forgetting tracks new-task forward KL
`KL(π₀‖π)` with **Pearson r ≈ −0.94** (linear R² ≈ 0.88), in both SI and no-SI
conditions. This is the empirical basis for optimizing new-task KL in Impl 2.5.
Data and figures: [`curve_run/Report_KL_POC.md`](curve_run/Report_KL_POC.md).

## Data

The SocraTeach SFT dataset is published on Hugging Face:
**[`meric533/socrateach-sft`](https://huggingface.co/datasets/meric533/socrateach-sft)**.

The large training split (`socrateach_sft_train.jsonl`, ~73 MB) is **not committed**
— pull it from HF or regenerate it with `socrateach_sft/prepare_socrateach_sft.py`.
The smaller `*_val.jsonl` / `*_test.jsonl` splits are included so the evals run
out of the box.
