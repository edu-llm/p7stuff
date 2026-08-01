# ATLAS CAT — adaptive checkpoint ability probe

Serve a checkpoint with vLLM, run an adaptive ARC-Challenge test that picks 8–40 items until the
standard error of the ability estimate drops below a stop threshold, and write `theta ± SE` per
checkpoint to S3.

**This runs alongside the existing evals — it replaces none of them.** `llm_judge/`,
`math_eval/`, `general_eval/` and `day1eval/` are all live and unchanged. CAT adds one signal
they do not produce: an ability estimate *with an error bar*, cheap enough to run on every
checkpoint of a grid without a human in the loop.

## What each eval answers

| Eval | Question | Automated? |
|---|---|---|
| `atlas_cat/` (this) | old-task ability along the trajectory, **with uncertainty** — is the change bigger than measurement noise? | **fully** — one GPU, no judges |
| `math_eval/` | math/logic final-answer accuracy, and the **format-failure vs wrong-answer** split | Colab inference, then offline grading + a verifier pass |
| `general_eval/` | general instruction-following parity vs base (MT-Bench, position-swap controlled) | Colab inference, then judge subagents |
| `llm_judge/` | **pedagogy** — blind 8-dimension Socratic rubric across the 4 SI/no-SI cells | judge subagents |
| `day1eval/` | standalone day-1 eval package | its own `run.sh` |

CAT is the only one of these that measures nothing about pedagogy, formatting, or
instruction-following. It is a science-QA multiple-choice ability probe and nothing more — which
is exactly why it does not stand in for the other four. Read it as the retention axis, and read
`llm_judge/` for whether the checkpoint is still a tutor.

Two things CAT gives you that the others do not:

* **An error bar.** `se` makes "did this checkpoint actually move?" answerable. The fixed prompt
  sets report point accuracies with no uncertainty, so a 19% → 11% drop on 70 items has no stated
  precision.
* **Per-checkpoint cost low enough for a grid.** No judge subagents, no Colab step, so all 11
  points of an Impl 4 arm can be scored unattended.

And the corresponding limitation: each checkpoint sees a *different* subset of items, so compare
`theta`, never raw accuracy. `n_items == 40` with `se` still above the stop means the estimate
never converged and should not be compared at all.

## Run it

```bash
# one arm's priority grid (Block S: all 11 points; Blocks T/G: {20,160,937})
./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 \
  --s3-out s3://YOUR_BUCKET/atlas_cat --priority

# every ckpt-* on disk
./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 --s3-out s3://YOUR_BUCKET/atlas_cat

# specific steps, tighter stop
./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 --s3-out s3://YOUR_BUCKET/atlas_cat \
  --steps 20,160,937 -- --se-stop 0.25
```

`--dry-run` prints the plan and touches nothing — no GPU, no S3. Results per checkpoint:

```
s3://YOUR_BUCKET/atlas_cat/<arm>-step<N>/
  atlas_arc_results.json    # theta, se, pirt_accuracy, n_items, selected_question_ids
  pipeline_provenance.json
  worker.log
  _READY                    # written last
```

The mechanics (conversion, vLLM, the olmo-eval invocation, all flags) live in the skill:
`.claude/skills/add-cat-evals/SKILL.md` and its `CONVERSION.md`. `score_checkpoints.sh` is a
sweep loop over that runner — it reads `manifest.json` for the arm name and base model,
`checkpoint_index.json` for `priority_checkpoints`, skips checkpoints that already have a
`_READY` in S3, and keeps going when one checkpoint fails.

p7 checkpoints are **LoRA adapters** (~25 MB), so each one is merged onto
`allenai/OLMo-2-0425-1B-Instruct` before serving and the merged copy is deleted after its CAT.
That is why the sweep is sequential: see CONVERSION.md on transient disk.

## Running the full picture on one arm

CAT is automated end to end; the other three have a manual step, so this is a sequence, not one
command:

1. **CAT** (unattended): `./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 --priority
   --s3-out …` → `theta ± SE` per checkpoint.
2. **Math/logic**: `math_eval/math_logic_eval_colab.ipynb` per arm → `grade_math_logic.py` →
   optional verifier pass. Gives the format-failure vs wrong-answer split CAT cannot.
3. **General parity**: `general_eval/general_eval_colab.ipynb` → `judge_build.py` → judge
   subagents → `judge_aggregate.py`.
4. **Pedagogy**: `llm_judge/build_batches.py` → judge subagents → `aggregate.py`.

Steps 2–4 are per-checkpoint-expensive (inference plus judging), which is the practical argument
for using CAT's dense grid to *locate* where the interesting movement is, then spending the
judged evals on those few checkpoints rather than all 11.

## A note on the KL–forgetting plane

`Report_KL_POC.md`'s r ≈ −0.94 was measured with math accuracy as the forgetting axis. `theta` is
a different axis on a different scale, so a CAT-based version of that plot is a **new**
measurement, not a continuation of the old one. Keep them separate; do not mix a CAT theta into a
series of math-accuracy points.

## Requirements

One GPU (~24 GB for a 1B merged model), `olmo-eval` installed with vLLM extras, `peft` for the
adapter merge, and AWS credentials with `s3:PutObject` on the results prefix. Full list in the
skill's SKILL.md.

**Not verified end to end here.** `olmo-eval` and the `atlas_arc` item bank are not vendored in
this repo, so the CAT step has only been exercised through `--dry-run`. The first real run should
be a single checkpoint, not a full sweep.
