# Impl 4 — Self-distilled replay for low-KL pedagogy SFT (implementation plan)

Agent-facing build spec. This is PRD §2.5 renumbered as "Impl 4" and **re-scoped**: the
intervention targets the *general/replay* stream, not the pedagogy targets.

**Scope: build data, train, save checkpoints. Nothing else.**
Evaluation is owned by another team. Do **not** implement, run, or modify:
`llm_judge/`, `math_eval/`, `general_eval/`, `curve_run/analysis/`,
`ORCD-SFT/generate_test_results.py`, PRD §3 (the 2×2), or any KL / judge / grading code.
Our deliverable is checkpoints + data + a manifest.

---

## 1. Hypothesis and what actually changes

Impl 2 forgets math, and forgetting tracks new-task forward KL `KL(π₀‖π)`
(r ≈ −0.94, `curve_run/Report_KL_POC.md`). Impl 2's 25% general replay stream uses
**Tülu-3 gold**, which is *not* on-distribution for `allenai/OLMo-2-0425-1B-Instruct` —
that checkpoint went through DPO/RLVR after the Tülu SFT stage. So the replay stream, whose
only job is to *not* move the model, is itself paying KL.

Impl 4 replaces gold replay with **π₀'s own outputs** on a broad general-domain prompt pool
(Super-NaturalInstructions), making the replay slot a near-zero-KL anchor.

| | Impl 2 | Impl 4 |
|---|---|---|
| Pedagogy stream (75%) | SocraTeach gold, per-dialogue SI | **unchanged** |
| Replay stream (25%) | Tülu-3 gold, SI-free | **self-generated from SuperNI prompts**, SI-free |
| Everything else (§2.2, §2.4, §2.6) | — | **unchanged** |

**Pedagogy targets are NOT self-distilled.** δ = 0 throughout. There is no tutor-turn
rewriting, no teacher-forced generation over SocraTeach, no pedagogy quality gate.

There are two swept axes:
- **σ** — fraction of the replay slot that is self-generated (§9, block S)
- **the sampling config** used to generate it — hold `T=1.0` untruncated, or tune it like
  SSD (§2, block T). This is an open question, not a settled default.

Expectation to hold onto: this can only remove the *incidental* KL. The KL of installing
Socratic behavior on pedagogy prompts is unchanged and unavoidable. We are changing the
source of 25% of the examples, on prompts where we weren't trying to change behavior — so
a **modest** effect is the honest prediction. That is precisely why the A2 control exists.

## 2. The sampling configuration — a primary axis, not a default

Two defensible settings for the anchor stream, and **we do not know which wins**: hold
`T_train = 1.0` with no truncation, or tune it the way SSD did. This is a primary experimental
axis alongside σ, not an ablation.

From *Embarrassingly Simple Self-Distillation* (Apple, arXiv:2604.01193), Eq. 4:

```
L = −log KeptMass_θ          (support compression, via ρ_train)
  + (1−T)·H_{1/T}(p_θ|S)     (within-support reshaping, via T_train)
  + T·KL(q ‖ p_θ,T|S)        (alignment to the base model)
```

The three terms do different jobs, and only the third anchors to π₀:

- `ρ_train` (truncation) drives **support compression** — strips diffuse tail mass
- `T_train` drives **within-support reshaping** — flattens or sharpens the retained head
- the KL term ties that reshaping back to π₀

**Case for holding `T=1.0`, no truncation.** At `T=1` with vacuous `ρ`, the first two terms
vanish and the target *is* π₀'s own distribution — the expected gradient goes to zero (their
Eq. 9). The paper's degenerate "no learning signal" case is the maximal-anchor case for us.
The replay slot then adds ≈ zero KL, which is the entire premise of Impl 4.

**Case for tuning it like SSD.** Three arguments we can't dismiss:

1. *An anchor that exerts no force is not an anchor.* A genuinely zero-gradient stream is
   inert — 25% of every step contributes nothing, so the pedagogy stream dominates the update
   and drift *per step* could be **higher** than with gold replay, which at least pulls
   somewhere. The best anchor may need a small but nonzero signal.
2. *`T=1.0` unrestricted on a 1B model produces a lot of junk.* Training on degenerate text
   teaches degenerate formatting. Truncation removes the tail at generation time.
3. *Support compression may be exactly the repair we need.* The paper's mechanism is that
   truncation suppresses distractor tails hardest at **lock** positions — where one
   continuation is correct and the rest are noise. "Put the final answer in `\boxed{}`" is a
   lock. Impl 2's math regression is substantially a *formatting* failure (PRD Appendix A: the
   model learned not to commit answers). A config that sharpens locks could reduce measured
   forgetting even while raising KL.

That third point makes this axis a second test of the same underlying question as σ: **is KL
the mediator of forgetting, or only a correlate?** If truncated arms show lower forgetting at
higher KL, that is a larger result than the one we set out to get.

Evidence on the risk side, weighted honestly: Table 5 shows SSD's out-of-domain damage
concentrated at small scale and landing on math — Llama-3.1-8B AIME'24 4.7 → 0.3,
Qwen3-4B-Instruct 61.3 → 55.0, 30B models flat. Their diagnosis of the Llama collapse:
*"the model frequently fails to output a final numerical answer and instead produces a code
block"* — a format failure, the same mode Impl 2 already has. But this transfers only weakly:
in the paper SSD was the *entire* training signal on code prompts, whereas ours is 25% replay
on general prompts. Treat it as a reason to keep `T=1.0` in the grid, not as a prediction.

### 2.1 The grid

Separate the two knobs, since they do different things. Truncation is held at the paper's
`top-k=20, top-p=0.8` wherever it is on, so `T` is the only thing moving within the truncated
set.

| Arm | `T_train` | `ρ_train` | Isolates |
|---|---|---|---|
| `T1` | 1.0 | none | pure anchor — no reshaping, no compression |
| `T2` | 1.0 | k=20, p=0.8 | support compression alone |
| `T3` | 1.3 | k=20, p=0.8 | + moderate reshaping |
| `T4` | 1.6 | k=20, p=0.8 | the paper's Qwen3-Instruct recommended point |

`T2` is the load-bearing arm: it separates "truncation cleaned up the junk" from "hot sampling
reshaped the distribution." Without it a win at `T4` is uninterpretable. `T3` supplies one
interior point so a monotone trend is distinguishable from a peak.

### 2.2 We do not control `T_eval` — this has to be coordinated

The paper's central finding is that the two temperatures **compose**: `T_eff = T_train × T_eval`
governs performance, with a broad peak near `T_eff ≈ 1.2` in the untruncated regime, and the
optimum shifting higher under more aggressive truncation. We set `T_train`; the eval team sets
`T_eval`. So this whole grid gets read at whichever single `T_eval` slice they decode at — the
regime where the composition structure is *least* visible.

Required of the handoff (§10): ask them to **hold `T_eval` fixed across all arms** and record
the value. If they have budget, ask for the `T` arms at 2–3 values of `T_eval` so `T_eff` is
identifiable. Without a fixed `T_eval`, this comparison is confounded by their decoding choice
and should not be reported.

## 3. Prompt pool: Super-NaturalInstructions

**Verify the source before building and record which you used in the manifest.**
Official: GitHub `allenai/natural-instructions` — `tasks/*.json` (1,616 tasks) +
`splits/default/train_tasks.txt`. Likely HF mirror: `Muennighoff/natural-instructions`
(unverified — the web search for this errored out; confirm it exists and has the task-id
field before depending on it).

Filters, applied in this order:

1. **English training tasks only** — use `splits/default/train_tasks.txt`. Hold out
   `test_tasks.txt` entirely; ship it unused (§8) in case the eval team wants a
   general-prompt KL axis.
2. **Contamination exclusion (mandatory).** Drop any task whose `Source` field references
   BIG-Bench, GSM8K, MATH, or AIME. `math_eval/` grades BBH-logical-deduction, and BBH ⊂
   BIG-Bench — a task-name/source filter alone is not enough, so also run a 13-gram overlap
   check against `math_eval/math_logic_prompts.jsonl` (70 prompts) and
   `general_eval/general_prompts.jsonl`, dropping any instance that hits.
   Note: `day1eval/decontam.py` referenced in earlier notes **no longer exists in the tree** —
   write the n-gram check fresh, ~30 lines.
   Overlap with Tülu-3 is fine and even desirable (Tülu contains FLAN v2, which contains
   Natural Instructions v2 — that makes SuperNI *more* in-distribution for π₀). Overlap with
   the eval sets is fatal.
3. **Gold output length filter (≥ 30 whitespace words, mean over the task's instances).**
   This is what makes the A2 control valid — see §5. Filter on *length*, not domain;
   report the retained task count and a `Categories` histogram so we can show domain
   breadth was preserved.
4. Sample instances round-robin across retained tasks so no single task dominates.

## 4. Generation

Use vLLM (add to `setup_orcd_env.sh`; a 1B model over ~7.5k prompts is a few minutes).
Fallback: batched HF `generate`, ~20–30 min on the L40S.

**Invariant, and the single most important one in this document:** sample under *exactly* the
chat template, prompt formatting, and assistant header used at training time. If generation
formatting differs from training formatting, the targets are not on-policy w.r.t. π₀ and the
entire premise is void. Concretely: build prompts with
`tokenizer.apply_chat_template(msgs, add_generation_prompt=True)` and pass
`prompt_token_ids` to vLLM — do not hand vLLM a raw string and let it re-template. The
training-side header is `<|assistant|>\n` (`ORCD-SFT/train_sft.py:171`).

- Model: `allenai/OLMo-2-0425-1B-Instruct` (π₀, frozen).
- Message shape: `[{"role": "user", "content": task_definition + "\n\n" + instance_input}]`.
  **No system message.** The SuperNI task definition goes in the *user* turn, never the
  system slot — the Impl 2 contract is "system message present ⇔ tutor mode", and putting a
  non-pedagogy system message in that slot would redefine the SI switch and change what eval
  cells B/C/D mean mid-experiment.
- **Sampling params come from the arm config** (`T_train`, `top_k`, `top_p` per §2.1), not from
  a hardcoded default. `build_general_slot.py` takes them as CLI flags and records them.
- `max_tokens`: tune so mean target length ≈ the Tülu-3 gold general slot's mean target
  length (measure that first; expect ~300–500). This is what makes A1↔A3 token-matched.
- `N=1` per prompt. The paper shows one sample suffices.

Because `T` and `ρ` change output length and degeneracy rate, **token-matching (§5) and the
degeneracy filter must be recomputed per arm** — expect the truncated arms to produce shorter,
cleaner text and therefore need a different `max_tokens` / subsample to hit the same token
budget. Record realized mean output length and drop rate for every arm; a `T` comparison across
arms with different realized token weights is not a `T` comparison.

**Degeneracy filter only — no quality gating** (except run B2). Exact rules, do not improvise:
drop if stripped output is empty; drop if < 3 whitespace tokens; drop if the whole output is
a single line under 8 characters; drop if any 10-gram repeats more than 4 times. Over-generate
~15% so the filter doesn't shrink the slot below target.

Anchor staleness is *correct here*: π₀ is frozen, so by step 900 the data is far from θ_t.
We are anchoring to π₀, not doing on-policy distillation. Do not "fix" this by regenerating
mid-run, and do not describe Impl 4 as on-policy self-distillation.

## 5. Mixing: token-matched, and why

`setup_orcd_env.sh` pins `transformers>=4.48.0`, which includes the gradient-accumulation
loss fix: loss is normalized by `num_items_in_batch` = total unmasked label tokens across the
whole accumulation group. **Stream weight is therefore token-proportional, not
example-proportional.** At a fixed 75/25 *example* ratio, arms with different target lengths
get different replay pressure per step — which would silently invalidate the comparison.

However, that fix depends on `Trainer.model_accepts_loss_kwargs`, which is resolved by
inspecting the forward signature and **can silently fall back to per-micro-batch mean when
the model is PEFT-wrapped**. So:

**Task: write a 2-step probe** that trains on two synthetic streams of known, very different
target lengths and checks whether the reported loss matches token-mean over the accumulation
group or mean-of-micro-batch-means. Record the answer in the manifest.

Then satisfy **both** normalizations, since they cost nothing together:

- **Token-match by construction** (correct under token-mean): pick generation `max_tokens`
  and subsample so each arm's replay slot has ≈ the same unmasked-label token total as A1's
  Tülu slot, at a fixed 7,496 examples. Tolerance ±5%. Log realized example *and* token
  ratios per arm.
- **Grouped block layout** (correct under micro-batch-mean, harmless otherwise): see §6.

The gold-output length filter in §3 is what makes this achievable for A2. Terse SuperNI
classification labels (5–20 target tokens) could only be token-matched by using ~10× more
examples, which would wreck the example balance. Filtering to long-form tasks means A2 and
A3 draw from an **identical** prompt pool at identical counts with comparable token totals —
a clean paired control. A1 remains the external reference.

## 6. Ordering and batching

`per_device_batch=8 × grad_accum=4` = **32 examples per optimizer step** (`run_sft.sbatch`).
Today `train_sft.py:268` uses the HF `Trainer` default `RandomSampler` over an
already-shuffled mix (`prepare_socrateach_sft.py:433`), so at φ=0.25 each step gets ~8 general
*in expectation* with sd ≈ 2.4 and occasional near-zero-general steps. The anchor should be a
**per-step constraint**, not an in-expectation one.

Pre-order the dataset into repeating blocks of 32: **24 pedagogy, then 8 general**. With a
`SequentialSampler` and `per_device_batch=8`, micro-batches are consecutive slices — positions
0–7, 8–15, 16–23 are pedagogy and 24–31 are the general micro-batch. Under micro-batch-mean
loss that gives the replay stream exactly 25% of every step's gradient, length-independently;
under token-mean it is merely good for padding efficiency. Shuffle *within* each stream pool
(seeded) before blocking so content is random while structure is fixed.

Arithmetic: 22,488 pedagogy + 7,496 general = 29,984 = **937 steps** at 937 blocks.
Pedagogy pool has 22,500, general 7,500 — both sufficient.

Required `TrainingArguments` / `Trainer` changes:
- override `_get_train_sampler` → `SequentialSampler`
- `dataloader_drop_last=True` (keeps block alignment at the tail)
- `group_by_length` must stay `False` — it reorders and would destroy the layout
- keep `seed=13`

## 7. Checkpointing

Forgetting is concentrated in the **first ~20 steps** (POC finding #3: math 20% → 11% by
step 20 while KL jumps to 0.33). A uniform `save_steps` grid misses the entire effect.

Save at: **5, 10, 20, 40, 80, 160, 320, 480, 640, 800, 937** (11 points). Step 0 is π₀ itself,
no checkpoint needed. Note `warmup_ratio=0.03` × 937 ≈ 28 steps, so the 5/10/20 points sit
*inside* warmup and are not on the cosine schedule proper — that is intentional (it is where
the damage happens), but flag it in the manifest so nobody misreads those points.

HF `Trainer` only supports a fixed interval, so add a `TrainerCallback` that on
`on_step_end` calls `model.save_pretrained(f"{out}/ckpt-{step}")` when `step` is in the grid.
**Adapter only** — a PEFT adapter here is ~25 MB (≈12M trainable params in bf16), whereas a
full trainer checkpoint also writes fp32 Adam state (~100 MB+). 11 × 6 runs ≈ 1.7 GB.

Keep HF's own `save_strategy="steps"` with a coarse `save_steps` (e.g. 300) and
`save_total_limit=2` *for resume only* — the ORCD partition caps at 6h and `train_sft.py`
already supports `--resume auto`. The eval grid is the callback's job; don't conflate them.
Save generously: re-running to recover a checkpoint we didn't save costs far more than disk.

## 8. Scripts to write

Build order. Pedagogy pool is built once and shared by all arms.

1. **Regenerate the pedagogy pool.** `ORCD-SFT/data/socrateach_sft_train.jsonl` is **absent**
   from the tree (only `_val` and `_test` are present). Run:
   ```
   python socrateach_sft/prepare_socrateach_sft.py --out_dir <pool> --seed 13 \
       --general_frac 0 --max_total 22500
   ```
   `--seed 13` is required — it reproduces Impl 2's problem-grouped split so val/test stay
   comparable. `--general_frac 0` yields pedagogy-only; Impl 4 owns the general slot.
2. **`build_general_slot.py`** (new) — SuperNI pull → filters (§3) → generation (§4) →
   degeneracy filter → token-budget subsample → per-arm JSONL. Emit records matching the
   existing schema (`messages`, `problem_id`, `dialogue_id`, `answer`, `source`, `kind`) with
   `kind` ∈ {`general_gold_tulu`, `general_gold_superni`, `general_ssd`} plus provenance
   fields: `superni_task_id`, `sample_T`, `sample_top_k`, `sample_top_p`, `gate_passed`.
3. **`mix_and_order.py`** (new) — takes the pedagogy pool + one or two general slots (for
   σ=0.5) → applies §6 block ordering → writes `socrateach_sft_train.jsonl` for that arm +
   `manifest.json`.
4. **Patch `ORCD-SFT/train_sft.py`** — sampler, `drop_last`, checkpoint-grid callback,
   `--arm` flag for output naming. Leave §2.2/§2.4/§2.6 logic untouched: the per-dialogue SI
   builder, the assistant-only masking in `make_tokenize_fn` (`train_sft.py:157`), LoRA
   `r=16/α=32`, cosine, `warmup_ratio=0.03`, LR 2e-4, 1 epoch, `max_len=1024`.
5. **`run_arm.sbatch`** — parameterized copy of `run_sft.sbatch` (1× L40S, `mit_normal_gpu`,
   4h, `per_device_batch=8 --grad_accum=4`).

## 9. Run matrix

Eight distinct runs, ~40 min each on the L40S (~5.5 GPU-hours). Only the replay slot differs;
everything else is Impl 2 unchanged.

**Block S — replay source.** Sampling held at the `T1` config so σ is the only thing moving.

| Arm | Replay slot | Question |
|---|---|---|
| `A1` | Tülu-3 gold (σ=0) | vanilla Impl 2 reference locus |
| `A2` | SuperNI **gold** | prompt shift, or self-generation? |
| `A3` | SuperNI SSD, σ=1, at `T1` | the intervention |
| `A4` | σ=0.5 — half Tülu gold, half SuperNI SSD | how much is needed? |

**Block T — sampling config.** All σ=1, all identical except `T_train` / `ρ_train` per §2.1.
`T1` **is the same run as `A3`** — train it once, reference it from both blocks.

| Arm | Config | Question |
|---|---|---|
| `T1` (=`A3`) | `T=1.0`, no truncation | pure anchor |
| `T2` | `T=1.0`, k=20, p=0.8 | does truncation alone help? |
| `T3` | `T=1.3`, k=20, p=0.8 | interior point — trend or peak? |
| `T4` | `T=1.6`, k=20, p=0.8 | does the paper's tuned config beat holding at 1.0? |

**Block G — gating.** `B2`: σ=1 at `T1`, gated against SuperNI gold.

A1 **cannot** be reused from `curve_run/` — those runs saved at 100-spaced and 20-spaced
steps and won't line up with the §7 grid for a paired comparison.

`A4` and `B2` are built at the `T1` config because something has to be chosen before Block T
resolves. If a truncated arm wins Block T, both should ideally be re-run at the winner (+2
runs). Note this in the manifest rather than silently pinning them to a losing config.

**Eval load.** Eight arms × 11 checkpoints = 88 points is a lot to ask of the eval team. Save
all 11 for every arm regardless — checkpoints are ~25 MB and re-running is expensive — but mark
a `priority_checkpoints` list in each manifest: **all 11 for Block S** (the curve comparison
needs the trajectory) and **{20, 160, 937} for Blocks T and G** (those need "where does this
arm land", not the full curve). Flag clearly that this is a deliberate coverage cap, not
complete coverage.

**B2's gate** (this is the *only* gating experiment; the pedagogy gate is gone with δ=0):
keep a sample if the normalized gold string appears as a substring of the output **or**
ROUGE-L F1 ≥ 0.3. On failure, **resample (up to 4 tries), do not fall back to gold** —
falling back reinjects off-policy targets exactly where we're trying to remove them. If a
prompt never passes, drop it and draw another so the slot count holds. Log the drop rate.

**Not sweeping `GENERAL_FRAC`.** Raising it above 0.25 would give the anchor more leverage but
also reduces pedagogy data, confounding "anchoring" with "less pedagogy pressure". Hold
φ = 0.25.

## 10. Deliverables

Per arm, under `impl4_ssd/runs/<arm>/`:
- `ckpt-{5,10,20,40,80,160,320,480,640,800,937}/` — PEFT adapters
- `socrateach_sft_train.jsonl` — the exact ordered training file
- `general_slot.jsonl` — the replay slot with full provenance
- `manifest.json` — σ, δ=0, `T_train` / `top_k` / `top_p` / `max_tokens`, realized mean output
  length, realized example **and** token ratios, gate drop rate, degeneracy drop rate,
  loss-normalization probe result, step count, checkpoint grid, `priority_checkpoints`,
  SuperNI source used, retained task count, seed, `transformers` version
- `train.log`

Shared, once:
- `superni_train_task_ids.txt` — tasks we trained on, so the eval team can keep their sets clean
- `superni_heldout_prompts.jsonl` — the untouched `test_tasks.txt` split, in case they want a
  general-prompt KL axis (POC finding #4 notes `kl_new_SI` and `kl_ped_noSI` were near-collinear
  and couldn't separate the gating hypothesis; a general-prompt KL is the natural third axis)

Asks of the eval team (the first is a hard requirement for Block T to mean anything):
- **Hold `T_eval` fixed across all arms and record the value** (§2.2). `T_train` and `T_eval`
  compose, so a `T_train` sweep read at a drifting `T_eval` is uninterpretable. If budget
  allows, evaluate the `T` arms at 2–3 `T_eval` values so `T_eff` is identifiable.
- Split the math metric into **format-failure rate vs. wrong-answer rate**. Right now a missing
  `\boxed{}` and a wrong number score identically, and per §2 those have opposite implications
  for whether Impl 4 worked.
- Plot every arm on the KL–forgetting plane against A1's locus, not endpoint deltas. If
  forgetting drops more than the pedagogy-KL drop predicts, points move **off** the Impl 2
  curve — which tests whether r ≈ −0.94 is a law or a within-run artifact.

## 11. Acceptance checks before any full run

1. Round-trip one generated general example through `make_tokenize_fn` and assert the
   unmasked label span decodes to exactly the assistant content + EOS.
2. Assert the generation-time prompt string is byte-identical to the training-time prefix for
   the same messages. This is the §4 invariant; a mismatch invalidates everything.
3. Assert no general record has a `system` message; assert every pedagogy record has one.
4. Run the loss-normalization probe (§5) and record the result.
5. Dump the first 3 blocks of the ordered train file and assert the 24/8 layout holds.
6. Assert 13-gram overlap with `math_logic_prompts.jsonl` and `general_prompts.jsonl` is zero.
7. `--poc` smoke run (cap ~2,000) end-to-end, confirming the callback writes adapters at the
   expected steps and that resume works.

## 12. Open

How many arms to train. The plan assumes **all eight**. Two smaller cuts, both coherent:

- **Four runs** — `A1`, `A2`, `A3`/`T1`, `T4`. Still answers both primary questions (does
  self-generated replay beat gold; does holding `T=1.0` beat tuning it), but loses `T2`, so a
  `T4` win can't be attributed to truncation vs. reshaping.
- **One run** — `A3` only, as a single deliverable checkpoint set compared against the existing
  `curve_run` Impl 2 numbers. Answers neither question rigorously; ~an afternoon.
