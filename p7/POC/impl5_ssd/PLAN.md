# Impl 5 — Self-distilled *pedagogy targets* for low-KL SFT (implementation plan)

Agent-facing build spec. This is PRD §2.5 taken **literally**: the intervention targets the
*pedagogy* targets, not the replay stream.

**Relationship to `impl4_ssd/`.** Impl 4 is PRD §2.5 *re-scoped* — it self-distills the
general/replay slot and explicitly pins δ = 0 ("pedagogy targets are NOT self-distilled",
`impl4_ssd/PLAN.md` §1). Impl 5 is the other half: δ is the swept axis and the replay slot goes
back to Impl 2's Tülu-3 gold. The two are orthogonal and composable; a δ=1 × σ=1 cell is the
natural follow-up and is out of scope here.

**Numbering.** The source spec is headed "5.x" but its Definition of Done says "(Impl 4)" — a
carry-over. This directory is Impl 5. The spec also references an "Impl 3" (token reweighting);
**no such implementation exists in this tree**. Do not cite it as if it did.

**Scope: build data, distill targets, train, save checkpoints. Nothing else.**
Evaluation is owned by another team. Do **not** implement, run, or modify:
`llm_judge/`, `math_eval/`, `general_eval/`, `curve_run/analysis/`,
`ORCD-SFT/generate_test_results.py`, PRD §3 (the 2×2), or any KL / judge / grading code.
The one exception is §4 Stage 4, which *calls* `day1eval/`'s judge client to calibrate a data
filter — that is a build-time gate on our own data, not an evaluation of a model.
Our deliverable is checkpoints + data + a manifest.

---

## 0. Measured facts to build against

Measured on `ORCD-SFT/data/socrateach_sft_val.jsonl` (1,724 examples / 513 problems, committed)
and taken from `impl4_ssd/RUNBOOK.md` §2b/§2d. Re-derive rather than trust — see §9.

| Fact | Value | Consequence |
|---|---|---|
| Tutor turns per dialogue | mean **5.30**, min 3, max 8 | 8 sequential rewriting rounds |
| Tutor turn length | **30.6** words mean, median 30, p90 52, max 156 | `max_tokens = 128`; no calibration loop |
| Pedagogy label tokens | ~224/example → **~5.0M** over 22,488 | |
| Tülu general slot (impl4-measured) | **600,173** tokens over 7,496 (mean 80.1) | |
| **General share of label tokens** | **~10.6%**, not 25% | The pedagogy stream carries **~89% of the gradient** under token-mean loss. Any drift in rewrite length moves the stream ratio far harder than it did in Impl 4. |
| Loss normalisation | `token_mean` (PEFT-wrapped, transformers 5.14.1) | Token-matching binds; example-matching does not. |
| Total rewrites | **119,288** generations for a 22,488-dialogue pool | one shared pass, ~30–60 min on an L40S with vLLM |
| Round sizes r=1…8 | 100 / 100 / 100 / 90 / 71 / 47 / 23 / 0.06 % of pool | batch across dialogues per round |
| **Gold turns stating the final answer** | **2.3%** of mid turns, **51.8%** of *final* turns | The answer-leak rule **must** be conditional on gold (§4 Stage 1). |

That last row is the most important number in this document. An unconditional
"does the rewrite state the answer?" rule would fall back to gold on over half of all final
turns — the highest-KL, most behavioural turns in the dataset — and Impl 5 would quietly
degenerate into Impl 2.

## 1. Hypothesis and what actually changes

Impl 2 forgets math, and forgetting tracks new-task forward KL `KL(π₀‖π)` (r ≈ −0.94,
`curve_run/Report_KL_POC.md`). RL's Razor: staying KL-minimal on the new task limits forgetting.
SFT on gold targets does not, because those targets sit far from π₀'s own distribution. SDFT
(Yang et al., 2024, arXiv:2402.13669) closes that gap by training on targets the model rewrote
into its own distribution.

| | Impl 2 | Impl 4 | Impl 5 |
|---|---|---|---|
| Pedagogy stream (75% ex / **~89% tokens**) | SocraTeach gold, per-dialogue SI | **unchanged** (δ = 0) | **self-distilled tutor turns**, δ swept 0→1 |
| Replay stream (25% ex / ~11% tokens) | Tülu-3 gold, SI-free | SuperNI self-generated (σ swept) | **unchanged — Tülu-3 gold** |
| Recipe | — | + `SequentialSampler` / 24-8 blocks | **stock Impl 2** + dense checkpoint grid (§6, §7) |

Impl 4's honest prediction was "modest" — it changed 25% of the examples on prompts where we
were not trying to change behaviour. **Impl 5's honest prediction is the opposite: a large
effect, of unknown sign on pedagogy.** It changes ~89% of the label tokens on exactly the
prompts where we *are* trying to change behaviour. The whole Definition of Done hinges on
*matched pedagogy quality*, which makes the pedagogy judge load-bearing rather than a
nice-to-have — see §4 Stage 4.

## 2. The interpretive frame (irreducible KL)

Some of Impl 2's new-task KL is the KL of **installing Socratic behaviour at all**. That part is
irreducible if the behaviour is to be installed. SDFT can only remove the stylistic/lexical part
— SocraTeach's phrasing versus π₀'s phrasing for the same pedagogical move.

So, if the gate genuinely preserves intent, what SDFT removes is *by construction* the
non-behavioural component. Read results through that:

- **KL drops, pedagogy holds** → the win. The removed KL was incidental.
- **KL drops, pedagogy falls** → the gate was too loose; we distilled away the behaviour, not
  the style. Not a win, even though the KL–forgetting plane will look great.
- **KL barely moves** → gold SocraTeach phrasing was already close to π₀, and the KL is
  behavioural. That is a real, publishable negative and it bounds what Impl 4 can achieve too.

This is why realised δ and the per-stage gate rates are **first-class manifest fields**, not
diagnostics. Without them the middle case is indistinguishable from the first.

## 3. The rewriting procedure

### 3.1 Sequential, prefix-consistent rewriting

For a dialogue with tutor turns `t₁…t_N` and student turns `s₁…s_{N-1}`, for round `r = 1…8`:
every dialogue with `N ≥ r` builds its distillation prompt from the **already-rewritten** prefix

```
[SI, u₀(problem), t̃₁, s₁, t̃₂, s₂, …, t̃_{r-1}, s_{r-1}]
```

samples `t̃_r` from π₀, gates it (§4), and on failure sets `t̃_r = t_r`.

Two properties this buys, and they are the reason the pass is sequential rather than one big
batch:

1. **The generation context equals the training context.** At training time the example contains
   `t̃₁…t̃_{r-1}`, so `t̃_r` must have been generated conditioned on those, not on gold. This is
   the multi-turn analog of impl4's §4 invariant.
2. **Fallbacks compose correctly.** A gold fallback at turn `r` does not abort the dialogue —
   later turns condition on the partly-gold prefix, which is exactly what training will see.

Rounds are batched across dialogues, so this is 8 vLLM calls of shrinking size (see §0), not
119,288 sequential ones.

### 3.2 Where the reference goes — the one invariant we cannot keep

This is the single place where impl4's §4 invariant **cannot** hold exactly. The distillation
prompt has to carry the gold turn as a reference, so it is strictly longer than the training
prefix. The same is true in the SDFT paper (its Fig. 3 distillation template differs from its
Fig. 10 training template); say so rather than pretending otherwise.

Minimise and *localise* the divergence: append the reference block to the **content of the last
user message** — the problem statement at `r = 1`, the student turn otherwise — rather than
inserting a new turn. That keeps role alternation identical to training and confines the
divergence to a suffix, which makes it mechanically checkable (§9 check 2).

Template, SDFT Fig. 3 "Using", adapted:

```
Write your next tutor message. A reference version of that message is given below —
use it as a guide for what to cover, but write it in your own words.

### Reference tutor message:
{gold t_r}

### Your tutor message:
```

**The per-dialogue SI stays in the system slot**, for both generation and training. Unlike
Impl 4 — which forbade a system message because the replay stream is SI-free — Impl 5's
pedagogy stream *is* the SI-conditioned stream. The Impl 2 contract ("system message present ⇔
tutor mode") is preserved. The SI is deterministic per `dialogue_id`
(`socrateach_sft/prepare_socrateach_sft.py:build_system_instruction`, md5-seeded), so it
reproduces without being stored separately.

### 3.3 Sampling configuration

Default: **`T = 1.0`, no truncation** — the maximal-anchor setting, where "the target is what π₀
would say" is literally true. Unlike Impl 4, sampling is a **secondary** axis here; δ is the
primary one, so pin a default and put the alternatives in Block R (§8).

State the tension rather than hiding it: `T = 1.0` untruncated on a 1B model over 119k turns
produces more junk than over Impl 4's 7,496, and junk becomes a gold fallback — so it **lowers
realised δ**. Record realised δ. If it comes in low, `R2` exists for exactly this.

### 3.4 Generation budget

`max_tokens = 128` (covers ~p99 of gold turn length at ~1.35 tok/word), `N = 1` per turn, stop
at EOS. No length-calibration loop: the problem impl4 §4 had to solve does not arise here,
because gold and rewrite are the same kind of object. Still record realised mean label tokens
per arm — §5 depends on it.

## 4. The pedagogy quality gate

Per spec: keep the rewrite only if it (a) does not reveal the final answer, (b) stays one step /
one idea, and (c) matches the gold turn's intent; **otherwise fall back to the original gold
turn.**

Note the deliberate difference from impl4's B2 gate, which resamples up to 4 times and *never*
falls back. There, falling back would have reinjected off-policy targets into the very slot we
were cleaning. Here, gold fallback **is** the spec and is the direct analog of SDFT Eq. 4 — the
cost is a lower realised δ, which we measure and report.

### Stage 0 — degeneracy

Reuse `impl4_ssd/impl4/degeneracy.py` verbatim: drop if empty, `< 3` whitespace tokens, a single
line under 8 characters, or any 10-gram repeated more than 4 times. Fall back to gold.

### Stage 1 — (a) answer leakage, deterministic and **conditional on gold**

The record carries `answer` (e.g. `"200"`). Fail iff

```
leaks(t̃_r)  and  not leaks(t_r)
```

where `leaks(x)` is "the normalised answer value appears among the numeric literals of `x`".
Add a phrase rule (`"the answer is"`, `"so the answer"`, `"the final answer"`) fired only when
gold has none, to catch non-numeric reveals.

The conditional-on-gold form is load-bearing: **51.8% of gold final turns state the answer**,
legitimately, after the student has produced it (§0). An unconditional rule fails half the
final turns.

Write a small local `impl5/answer_leak.py` modelled on
`math_eval/grade_math_logic.py:extract`/`check` (the `int` branch) rather than importing that
script — it is CLI-shaped and owned by the eval team.

### Stage 2 — (b) one step / one idea, heuristic

Fail if any of:

- `words(t̃) > max(2.5 × words(t_gold), 90)` — a rewrite that balloons is walking through
  multiple steps;
- more than 2 question marks (gold's median is one guiding question);
- an enumerated multi-step list (`^\s*(\d+[.)]|[-*])\s` on ≥ 3 lines) where gold has none;
- more than 6 sentences.

Thresholds are **calibrated in Stage 4, not guessed.**

### Stage 3 — (c) intent match

ROUGE-L F1 against gold `t_r`, reusing `impl4_ssd/impl4/gate.py:rouge_l_f1` (LCS over
`impl4/textutil.py`-normalised tokens).

State the asymmetry explicitly, because it is the opposite of impl4's B2:

- **Too high** a threshold keeps only near-copies — that is vanilla SFT with extra steps, and
  it removes the very KL reduction we are buying.
- **Too low** and the rewrite no longer sets up the gold student reply that follows it, so the
  dialogue stops making sense.

Provisional **0.25** (impl4's B2 used 0.3 for a *fidelity* task; this is a *paraphrase* task, so
lower). Set it in Stage 4.

### Stage 4 — blind-judge calibration (a calibration pass, **not** a per-turn gate)

A full blind-judge pass over 119k turns is ~180M input tokens and would dominate the cost of the
whole experiment. Instead: a **~600-turn stratified sample** (across turn index, gate outcome,
and ROUGE band) through `day1eval/scoring.py:build_judge_messages` +
`day1eval/llm_client.py:chat_completion` (`PROMPTLENS_API_KEY`, `openai-group/gpt-5.6-sol`).

Two jobs:

1. **Fit the Stage-2/3 thresholds** so the deterministic gate agrees maximally with the judge's
   `Revealing_of_the_Answer` = "No" and with `Coherence` / `Actionability` not falling below
   gold.
2. **Report gate precision/recall** and a blind, position-shuffled **rewrite-vs-gold pedagogy
   delta with bootstrap CIs** (`day1eval/stats.py:bootstrap_ci`).

Run it on a **~2,000-turn pilot before the full distillation pass**, and again on a fresh sample
after. **This is the designated kill/go gate**: it measures the Definition of Done's "matched
pedagogy quality" on the *data*, before a single GPU-hour of training. If the pilot shows the
rewrites are materially worse than gold, stop and fix the template or the sampling config —
do not train five arms and find out afterwards.

### What the manifest must record

Per-stage fallback rates; **fallback rate by turn index** (expect it to climb with `r` as the
rewritten prefix drifts from gold — that is the §13 coherence risk made visible, and it caps
realised δ); realised δ in *dialogues* and, separately, in *label tokens*.

## 5. Token matching — and why Impl 4's answer does not transfer

Loss is token-mean (measured, §0) and pedagogy is ~89% of label tokens. If rewrites are
systematically shorter or longer than gold, then the pedagogy:general token ratio moves across
δ arms and **a δ sweep silently becomes a stream-weight sweep** — the same failure mode
impl4 §5 identified, with a much bigger lever arm.

Impl 4 had it easy: its pedagogy stream was fixed, so token-matching the replay slot preserved
*both* the pairing and the ratio. Here the two conflict — matching by selecting different
dialogues per arm would break the paired δ contrast. Resolution:

- **Hold the pedagogy set fixed across all δ arms** — the same 22,488 dialogues, the same ids,
  in the same order. The δ contrast stays exactly paired on identical prompts, which is worth
  more than an exactly matched ratio.
- **Absorb the drift in the general slot.** Keep 7,496 Tülu examples (so example count and the
  937-step count are unchanged) but choose *which* ones with
  `impl4_ssd/impl4/mixing.py:token_matched_select`, targeting the token total that makes
  `general_tokens / pedagogy_tokens` equal to D0's realised value in every arm. Over-draw
  ~15,000 Tülu candidates for selection headroom. Tolerance ±5%; record the realised ratio per
  arm.
- **D0 is built first** and writes `data/pedagogy_reference.json` — the same role
  `data/tulu_reference.json` plays in Impl 4. Everything else refuses to build without it.
- **Escape hatch, stated not defaulted:** if pedagogy drift is large enough that 7,496 Tülu
  examples cannot reach the required total, report the unmatched ratio and say so when the arms
  are compared. Do not fake the match by rescaling.

## 6. Training recipe — deliberately *not* Impl 4's

Import `ORCD-SFT/train_sft.py` and reuse `make_tokenize_fn` and `load_model_and_tokenizer`
unchanged, via the same path-import trick as `impl4_ssd/impl4/chat.py:impl2_trainer_module()`.
That is the strongest available guarantee that PRD §2.2 (per-dialogue SI), §2.4 (assistant-only
masking) and §2.6 (LoRA r=16/α=32, cosine, warmup 0.03, LR 2e-4, 1 epoch, `max_len=1024`) are
untouched — they are literally the same code objects.

**No sampler override. No `dataloader_drop_last`. No 24/8 block ordering.** Impl 2's shuffled
mix and the `Trainer` default `RandomSampler`, per "the same training recipe as Impl 1 and 2".

Say plainly what this costs: the replay stream is a 25%-**in-expectation** constraint per step,
not a per-step one. impl4 PLAN §6's critique of that still stands and we are accepting it —
because in Impl 5 the anchor is the *targets*, not the stream layout, and deviating from Impl 2's
recipe would confound the δ effect with a batching change.

Arithmetic: `22,488 + 7,496 = 29,984 = 937 × 32` (`per_device_batch=8 × grad_accum=4`), so 937
optimizer steps and no partial final batch.

**Known bookkeeping difference:** `curve_run/`'s Impl 2 run reports 923 steps, so its
checkpoints cannot be reused as D0 — they will not pair with the §7 grid. Re-run D0. (Same
reason `impl4_ssd/PLAN.md` §9 gives for A1.)

## 7. Checkpointing

Identical to impl4 §7, for the same reason: forgetting is concentrated in the **first ~20
steps** (POC finding #3 — math 20% → 11% by step 20 while KL jumps to 0.33), and a uniform
`save_steps` grid misses the entire effect.

Save at **5, 10, 20, 40, 80, 160, 320, 480, 640, 800, 937** (11 points). Step 0 is π₀. Use a
`TrainerCallback`, reusing `impl4_ssd/impl4/trainer.py:checkpoint_grid_callback`. **Adapter
only** (~25 MB each; 11 × 5 runs ≈ 1.4 GB). HF's own `save_strategy="steps"` stays coarse
(`save_steps=300`, `save_total_limit=2`) and exists **for resume only** — the ORCD partition
caps at 6h. Do not conflate the two.

`warmup_ratio=0.03 × 937 ≈ 28`, so the 5/10/20 points sit *inside* warmup. That is intentional —
it is where the damage happens — but flag it in the manifest so nobody reads them as points on
the cosine schedule.

Using impl4's **exact** grid is what lets Impl 4 and Impl 5 arms share one KL–forgetting plane.
Do not "improve" it.

## 8. Run matrix

### Block D — distilled fraction (primary axis; spec §5.2.4, mirrors SDFT §5.1)

δ is assigned at the **dialogue** level, not the turn level. Two reasons: mixing rewritten and
gold turns inside one dialogue creates prefixes that neither the rewriter nor the trainer ever
sees coherently; and dialogue-level assignment keeps realised δ interpretable. (Gate fallbacks
still mix gold turns into distilled dialogues — that is unavoidable, and is precisely why
"realised δ in label tokens" is the reported quantity.)

Assignment is seeded and **nested**: D1 ⊂ D2 ⊂ D3 ⊂ D4. That makes the sweep monotone rather
than four independent samples, so a non-monotone result means something.

| Arm | δ | Question |
|---|---|---|
| `D0` | 0.00 | vanilla Impl 2 reference locus (must be re-run — see §6) |
| `D1` | 0.25 | |
| `D2` | 0.50 | the mix-ratio interior |
| `D3` | 0.75 | |
| `D4` | 1.00 | full SDFT — the intervention |

Five runs, ~40 min each ⇒ **~3.5 GPU-hours**, plus one shared distillation pass (~30–60 min).

**Cut, if only three are affordable:** `D0`, `D2`, `D4`. Answers "does distillation reduce
forgetting" and gives one interior point; loses the shape of the curve.

### Block R — rewriting configuration (secondary; second wave)

All δ = 1. `R1` **is the same run as `D4`** — train once, reference from both blocks.

| Arm | Config | Question |
|---|---|---|
| `R1` (= `D4`) | `T=1.0`, untruncated, reference-in-context | the default |
| `R2` | `T=1.0`, k=20, p=0.8 | does truncation lift realised δ (fewer degeneracy fallbacks) without raising KL? |
| `R3` | greedy (`T=0`) | the mode of π₀ — the lowest-KL targets available. Does losing diversity flatten the tutor's style? |
| `R4` | **reference-free continuation** | π₀ generates the next tutor turn from the *exact* training prefix, no reference block, gated identically. The only variant where impl4's §4 invariant holds **strictly**, so it prices what reference-in-context costs in KL. It is really "context-distilling Impl 1 into Impl 2". Expect a much higher fallback rate. |

`R4` is the one not to cut from the second wave: without it there is no measurement of the §3.2
compromise.

### Priority checkpoints

All 11 for **Block D** (the curve comparison needs the trajectory); `{20, 160, 937}` for
**Block R** (those only need "where does this arm land"). All 11 are saved regardless —
checkpoints are ~25 MB and re-running to recover one costs far more than disk. Flag in every
manifest that this is a **deliberate coverage cap, not complete coverage.**

### Not swept

`GENERAL_FRAC`, held at 0.25. Raising it would give the replay stream more leverage but also
reduce pedagogy data, confounding "anchoring" with "less pedagogy pressure" (impl4 §9's
reasoning, unchanged).

## 9. Acceptance checks — run these before any full run

1. **Label-span round-trip.** A distilled record through `make_tokenize_fn`: the unmasked label
   span decodes to exactly the concatenated rewritten turns + EOS. Reuse
   `impl4_ssd/impl4/chat.py:assert_label_span_roundtrip` (already multi-assistant aware).
2. **Multi-turn prefix invariant.** For every round `r`, the distillation prompt's token ids
   must equal the training prefix's ids at every position **before** the appended reference
   block — the only permitted divergence (§3.2). This requires extending
   `impl4/chat.py:training_prefix_ids`, which currently *raises* on an assistant turn, to mirror
   `make_tokenize_fn`'s assistant branch (`enc("<|assistant|>\n") + enc(content) + [eos] + nl`).
   Put the extension in a local `impl5/chat5.py` and assert the reconstructed prefix equals
   `apply_chat_template(..., add_generation_prompt=True)` on the same messages.
3. **System-message contract, both directions.** Every pedagogy record has a system message;
   every general record has none.
4. **Loss-normalisation probe.** `python ../impl4_ssd/probe_loss_norm.py --arm <arm>`; record
   the verdict in the manifest. The answer depends on the installed `transformers`.
5. **δ arithmetic.** Realised distilled-dialogue count `== round(δ × 22,488)` exactly, and
   nestedness holds across D1…D4.
6. **Answer-leak rule sanity on gold.** Run the Stage-1 rule with `t̃ := t_gold`; it must fire on
   ~0%. This is the check that catches the conditional-on-gold logic being inverted — without
   it, §0's 51.8% silently becomes a 51.8% fallback rate on final turns.
7. **Decontamination is unchanged, not zero.** 13-gram overlap with
   `math_eval/math_logic_prompts.jsonl` and `general_eval/general_prompts.jsonl` must be
   **identical between D0 and D4** (reuse `impl4/ngram.py`). The target is "unchanged" rather
   than "zero" because SocraTeach is built on GSM8K/MAWPS: any overlap is inherited from Impl 2
   and must not be altered here. The check's job is to prove distillation introduced none.
8. **`--poc` smoke run** end to end (63 blocks / ~2,000 examples, POC grid `{5,10,20,40,63}`),
   confirming the callback writes adapters at the expected steps and that `--resume auto` works.

## 10. Scripts to write

Build order. The distillation pass is shared by every arm and is run once.

| Script | Role |
|---|---|
| `build_pedagogy_pool.py` | Wraps `socrateach_sft/prepare_socrateach_sft.py --seed 13 --general_frac 0 --max_total 30000`. `--seed 13` is required (it reproduces Impl 2's problem-grouped split so val/test stay comparable); `--general_frac 0` because Impl 5 owns the general slot. Asks for **30,000**, not impl4's 22,500, so §5 has headroom. Asserts pedagogy-only + system message present; reports the turn-count histogram. |
| `distill_pedagogy.py` | **The core, new.** 8 sequential gated rounds (§3) over the whole pool, one shared pass, resumable per round. Writes `data/distilled_pool.jsonl` (per dialogue: gold turns, rewritten turns, per-turn gate verdict + reason, sampling config) and `data/distill_meta.json`. |
| `calibrate_gate.py` | §4 Stage 4 — pilot and post-hoc blind-judge calibration → `data/gate_calibration.json`. |
| `build_general_slot.py` | Tülu-only, via `impl4/tulu.py:load_tulu_slot`, token-matched per §5. Far smaller than impl4's version: no SuperNI, no generation. |
| `mix_arm.py` | Pick the δ-fraction of dialogues (seeded, nested), substitute rewritten targets, attach the general slot, apply **Impl 2's shuffle** (`random.Random(seed+1).shuffle`, *not* block ordering), write `runs/<arm>/socrateach_sft_train.jsonl` + the manifest `mix` section, link `socrateach_sft_{val,test}.jsonl`. |
| `train_sft_impl5.py` | Stock Impl 2 trainer + the dense checkpoint grid + `--arm`. |
| `acceptance_checks.py` | §9. |
| `run_arm.sbatch`, `run_all.sh` | Parameterised, mirroring impl4's (1× L40S, `mit_normal_gpu`, 4h). |
| `tests/test_impl5.py` | Stdlib-only, no GPU/network: gate rules, δ arithmetic and nestedness, round scheduling, the answer-leak conditional. |

### Reuse policy

`impl5/_impl4.py` puts `../impl4_ssd` on `sys.path` and re-exports `degeneracy`, `textutil`,
`gate.rouge_l_f1`, `mixing.token_matched_select`, `manifest`, `trainer.checkpoint_grid_callback`,
`tulu`, `chat`, `ngram`.

**Import rather than copy, and the reason is not convenience:** both implementations' arms are
going onto the same KL–forgetting plane. A silent divergence in tokenisation, token-matching, or
the degeneracy rules between Impl 4 and Impl 5 would invalidate that comparison without
producing an error anywhere. `paths` is redefined locally (layout-specific and trivial).
`impl4/paths.py` resolves relative to its own file, so importing it from here keeps `POC_ROOT`
correct.

Risk of the coupling: Impl 5 breaks if Impl 4 changes. Mitigation: §9's checks fail loudly, and
`tests/test_impl5.py` pins the behaviours we depend on.

## 11. Deliverables

Per arm, under `impl5_ssd/runs/<arm>/`:

- `ckpt-{5,10,20,40,80,160,320,480,640,800,937}/` — PEFT adapters
- `socrateach_sft_train.jsonl` — the exact training file
- `manifest.json` — δ nominal / realised-dialogues / **realised-label-tokens**; sampling config;
  rewriting rounds; per-stage gate fallback rates; **fallback rate by turn index**; realised
  pedagogy and general label-token totals + their ratio; gate thresholds and their calibration
  provenance; loss-normalisation probe result; step count; checkpoint grid;
  `priority_checkpoints`; seed; `transformers` version
- `checkpoint_index.json`, `train.log`

Shared, once:

- `data/distilled_pool.jsonl`, `data/distill_meta.json` — the single distillation pass every arm
  draws from
- `data/gate_calibration.json` — thresholds, judge model, precision/recall, and the judged
  rewrite-vs-gold pedagogy delta with bootstrap CIs
- `data/pedagogy_reference.json` — D0's realised pedagogy token total

## 12. Asks of the eval team

- **Both conditions, both KLs** — `kl_new_SI` and `kl_ped_noSI`, per `curve_run/` (spec §5.3).
  This matters more here than in Impl 4, because Impl 5 changes the *SI-conditioned* stream: if
  SI-gating breaks anywhere, it breaks here. Two named risks:
  - rewrites drifting toward π₀'s default assistant register would erode gating from both sides
    (`pedD` down, `pedC` up);
  - because the targets sit closer to π₀'s no-SI behaviour, `kl_ped_noSI` should fall *more*
    than `kl_new_SI`.
  POC finding #4 records that the two KLs were near-collinear under Impl 2 and therefore could
  not separate the gating hypothesis. **Impl 5 may be the regime where they diverge** — which
  would be a bonus result worth reporting on its own.
- **Pedagogy quality with CIs** (`llm_judge/`, 8-dim blind judge). The Definition of Done is
  reduced forgetting *at matched pedagogy quality* — a forgetting win bought with a pedagogy
  loss is not a win. Cells `C_sft_noSI` and `D_sft_SI` at minimum, at δ=0 and δ=1, so D−C is
  comparable.
- **Plot every arm on the KL–forgetting plane** against D0's locus, not endpoint deltas. The DoD
  is "moves down-left".
- **Split the math metric** into format-failure rate vs. wrong-answer rate (inherited from
  impl4's ask; a missing `\boxed{}` and a wrong number currently score identically and have
  opposite implications).
- **Hold decoding temperature fixed across arms and record it.**

## 13. Definition of done

Versus vanilla Impl 2 (`D0`) at matched pedagogy quality (P5 rubric / blind judge, CIs):

- (a) **reduced** math/logic forgetting;
- (b) **lower** new-task KL — the arm moves down-left on the KL–forgetting plane;
- (c) **SI-gating preserved** — no-SI behaviour and `kl_ped_noSI` stay close to base.

Reuse the `curve_run/` pipeline for KL, math grading, and pedagogy judging.

## 14. Open questions and risks

- **The 1B rewriter is the largest threat.** π₀ is `allenai/OLMo-2-0425-1B-Instruct`; its
  *prompted* pedagogy is `pedB ≈ 0.79` overall against SFT+SI's 0.93
  (`curve_run/Report_KL_POC.md`). With the gold turn as a reference the rewrite should
  comfortably beat free-running `pedB`, but the ceiling is still a 1B model, and that is the
  single biggest risk to "matched pedagogy quality". §4 Stage 4 is the kill/go gate for exactly
  this, and it runs before any training.
- **The coherence risk is real and measurable.** Gold student turns were written in response to
  gold tutor turns. Track fallback rate by turn index; if it climbs steeply with `r`, the
  rewritten prefix is drifting and the later turns are increasingly gold — which caps realised
  δ and must be reported, not hidden.
- **Whether δ should ever be applied per-turn** rather than per-dialogue. Currently no; revisit
  only if the per-turn fallback structure turns out to be benign.
- **Whether Block R is worth a second wave** at all, and if so whether `R4`'s strict-invariant
  contrast is better spent as a standalone experiment.
- **The Impl 4 × Impl 5 composition cell** (σ=1, δ=1). Out of scope, but both manifests carry
  enough provenance to build it without re-deriving anything.
