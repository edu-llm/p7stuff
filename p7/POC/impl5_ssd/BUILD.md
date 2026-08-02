# Impl 5 — what this build actually is

`PLAN.md` is the full spec. This file records what was built, what was cut, and every place
the build knowingly departs from the plan. Read it before quoting a number from `runs/`.

## Run status: STOPPED — compute units exhausted, no ped_nll numbers

The implementation is complete and validated. **The training run did not finish.** The
distillation pass completed (118,870 rewrites, 47.4% keep), the mix and all acceptance checks
passed, training started and reached ~step 20 of 923 — and then the Colab runtime was
reclaimed because compute units ran out. A100, L4 and T4 are all now refused for this account.

Lost with the runtime: `data/distilled_pool.jsonl`, the nine round caches, the mix, and the
handful of early checkpoints. **None of it was downloaded first, which was my error** — see
`stash_pool` in `run_impl5.py`, added afterwards, which packages the pool the moment it exists
and prints the download command. Everything else survives in git.

To finish, on any GPU runtime:

```bash
colab upload impl3_handoff.tar.gz /content/impl3_handoff.tar.gz
colab exec -f colab_bootstrap5.py          # IMPL5_STAGES defaults to all
```

Cost is ~90 accelerator-minutes to regenerate the pool, ~45 for training, ~25 for ped_nll.
The `probe` stage now fails fast if the rewriter's keep rate is unusable, so a bad template
costs two minutes rather than ninety.

## The one-line version

**D4** (δ = 1.0, every tutor turn rewritten by π₀, Tülu-3 gold replay) trained for 923 steps
on Impl 4's exact recipe and graded on pedagogy NLL only. Its baseline **D0 is impl4's A1** —
already trained, already graded, and already shown to reproduce Impl 3's `impl2-rerun` on
every axis.

## Deviations from PLAN.md, and why

| # | PLAN says | This build does | Why |
|---|---|---|---|
| 1 | 937 steps (§6) | **923 steps**, impl4's 22-point union grid (§7) | §7 also says "using impl4's *exact* grid is what lets Impl 4 and Impl 5 arms share one KL–forgetting plane. Do not 'improve' it." impl4's grid *is* 923. At 937 no Impl 5 checkpoint would share a step number with any Impl 3 or Impl 4 checkpoint. |
| 2 | Stock Impl 2 batching — `RandomSampler`, no block layout (§6) | **`SequentialSampler` + 24/8 blocks**, identical to A1 | §6's argument holds when D0 is re-run alongside. Here D0 *is* A1. Changing the sampler would compare D4 against a baseline differing in two ways and cost a training run the budget does not have. |
| 3 | Re-run D0 (§6, §8) | **D0 = impl4-A1**, not re-run | Same reason. A1 is vanilla Impl 2 on the same pool, same seed, same 923 steps, and it gates against Impl 3. |
| 4 | Token-match the Tülu slot to D0's ped:gen ratio (§5) | **Byte-identical slot to A1's**; ratio drift measured and reported | §5's matching is right for a full D0…D4 sweep. With one trained arm it would change the pedagogy targets *and* the replay conversations in the single contrast the run exists to make. `mix_arm5.py` reports the drift and warns past ±5%. Pass `--token_match` to restore §5. |
| 5 | Stage 4 blind-judge calibration is the **kill/go gate** (§4) | **Not run** | Needs `day1eval`'s judge, `PROMPTLENS_API_KEY`, and ~600 judged turns. This run is training + ped_nll only. Thresholds are §4's provisional values, marked `calibrated: false` everywhere. |
| 6 | D0…D4 sweep, Block R second wave (§8) | **D4 only** | Compute budget. The distillation pass is shared, so D1/D2/D3 cost only training+eval if credits reappear. |
| 7 | Math/KL/pedagogy-judge evals (§12) | **ped_nll only** | Explicitly scoped out for this run. Rows carry `axis: "ped_nll"` so a partial file cannot merge as though complete. |

### What deviation 5 costs, stated plainly

The Definition of Done (§13) is reduced forgetting **at matched pedagogy quality**. Without
Stage 4, "matched pedagogy quality" is **unverified**. A δ=1 arm that looks good on forgetting
could have got there by distilling away the teaching rather than the phrasing — §2's second
case, which "will look great on the KL–forgetting plane" and is not a win. Do not report a
win from this run alone.

## Measured on the real pool (re-derived, not trusted — §9)

| Fact | PLAN §0 | Measured here |
|---|---|---|
| Dialogues | 22,488 | **22,500** |
| Tutor turns per dialogue | mean 5.30, max 8 | mean **5.28**, max **9** |
| Total rewrites | 119,288 | **118,870** |
| Round sizes | 8 rounds | **9 rounds**: 22500/22500/22500/20180/15840/10413/4930/5/2 |
| Tutor turn words | mean 30.6, p90 52 | mean **30.3**, p90 **51**, max 169 |
| Gold turns stating the answer | 2.3% mid / 51.8% final | **3.9% mid / 67.4% final** |

The last row is the one that matters, and it is *more* lopsided than the plan's (which was
measured on the 1,724-example val split). An unconditional answer-leak rule would fall back
to gold on two thirds of all final turns.

### The rewriter does not write gold-shaped turns, and that breaks PLAN §3.4

The single biggest finding of the build, discovered on the first round of the real pass.

PLAN §3.4 sets `max_tokens = 128`, justified as "covers ~p99 of gold turn length at ~1.35
tok/word", and states: *"No length-calibration loop: the problem impl4 §4 had to solve does
not arise here, because gold and rewrite are the same kind of object."*

They are not the same kind of object. Round 1 under PLAN §3.2's template:

| | value |
|---|---|
| keep rate | **2.1%** |
| rejected `unterminated` | **89.1%** |
| median generated tokens | 128 — i.e. the cap |
| mean rewrite length | **108 words** (gold: 8.7) |

Handed a math problem and a reference, the 1B model writes a *worked explanation*, not a
Socratic prompt. At a 2.1% keep rate the distilled pool is 98% gold and D4 collapses onto D0,
so the pass was stopped rather than run to completion.

Crucially, **the cap was not the binding constraint.** Re-running the plan's template at
`max_tokens = 160` moved the keep rate only from 2.1% to 2.5% — it buys longer essays, not
more terminations. The fix is register, not budget: the template now states the target
register (ask, don't explain), the prohibition the gate checks, and gold's own word count so
the target scales with the turn (gold runs 8.7 words at round 1 and 35.8 by round 4).

Measured keep rate by template (240 dialogues per cell, `max_tokens = 160`):

| template | r1 | r4 | r7 | weighted | words @r1 (gold 8.7) |
|---|---|---|---|---|---|
| `plan` (PLAN §3.2 verbatim) | 2.5% | 11.7% | — | ~7% | 107.9 |
| `mirror` **(default)** | 71.5% | 48.0% | 25.5% | **56.8%** | 14.4 |
| `brief` | 56.0% | 48.0% | 28.5% | 49.8% | 28.8 |
| `cover` | 33.0% | 39.5% | 40.5% | 36.5% | 21.0 |

Weighted by real round sizes (round 1 runs 22,500 dialogues, round 7 runs 4,930). The
ranking is not what reading the templates suggests — `cover`, which presses hardest on
covering the reference, is the *worst* overall: it draws the model back toward explaining and
its answer-leak rejections run 2-4x the others'. `mirror` wins early rounds decisively and
loses late ones; since most turns are early, it wins.

**The dominant rejection reason after the fix is `low_rouge`**, not length — the ROUGE-L
floor of 0.25 (PLAN §4 Stage 3, provisional). Lowering it would buy keep rate directly, and
it was deliberately left alone: that threshold is precisely what Stage 4 exists to calibrate,
and tuning it against the keep rate rather than against the judge would be trading a
measurable δ for an unmeasured pedagogy risk.

`max_tokens` is kept at 160 so that "unterminated" now means the model rambled rather than
that the budget was short.

**What this costs conceptually.** SDFT's premise is that targets are "what π₀ would say".
The targets are now what π₀ says *when told how long to be and to ask rather than explain* —
still the model's own distribution, but a conditioned slice of it. That is a real departure
from PLAN §3.2 and is recorded in `distill_meta.json` and every arm manifest. Block R's `R4`
(reference-free) is what would price it, and it did not run.

### Two more facts the plan does not have:

- **Reference-block overhead**: mean 84 tokens, max 160, appended to the last user message.
- **Gate strictness floor**: running the whole gate with `t̃ := t_gold` rejects **1.38%** of
  real tutor turns (`too_many_questions` 1,283, `too_many_sentences` 355). Every reported
  fallback rate should be read against that floor — 1.38% of it is the thresholds, not the
  rewrites.

## What the distillation pass actually produced

| | |
|---|---|
| turn-level keep rate | **47.4%** of 118,870 tutor turns |
| realised δ, dialogues | 1.00 (every dialogue has some rewritten content) |
| **realised δ, label tokens** | **0.368** |
| tutor words | 3,603,889 gold → 3,286,161 distilled (0.912×) |
| pedagogy label tokens | 4,579,557 gold → 4,206,645 (0.919×) |
| ped:gen token ratio vs D0 | **+8.9% — outside the ±5% tolerance** |
| decontamination reverts | 1 dialogue (`GSM8K_test_439_0`) |
| replay slot | reproduces impl4-A1 exactly (7,384 / 631,395 tokens) |

Rejections by stage: `intent_match` 50,290 · `answer_leak` 6,350 · `degeneracy` 4,721 ·
`one_step` 1,126 · `decontamination` 4.

**Call this arm δ=0.37, not δ=1.** Nominal δ is 1.0 — every dialogue was put through the
rewriter — but gate fallbacks put gold turns back, and 47.4% of *turns* kept becomes only
36.8% of *label tokens* because accepted rewrites are systematically shorter than the gold
turns they replace. In effective strength that sits between PLAN §8's D1 and D2. It is still
a substantial intervention (37% of the pedagogy gradient is self-distilled against 0% for
D0), but "full SDFT" would be the wrong description and the manifest reports the realised
figure everywhere.

### Fallback rate climbs with turn index, as §13 predicted

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| fallback | 30% | 53% | 53% | 58% | 61% | 66% | 74% |

This is the coherence risk made visible: gold student turns were written in response to gold
tutor turns, so as the rewritten prefix drifts from gold the later turns fit worse and are
rejected more. It caps realised δ, and it is the reason the token-weighted figure is what
gets reported.

### The stream-weight confound, and why it was not corrected

Rewrites are ~8% shorter, so pedagogy carries fewer label tokens and the general stream's
relative weight rises **8.9%** — past PLAN §5's ±5% tolerance. Part of any D4-vs-D0
difference is therefore a stream-weight difference, and more replay weight should push D4's
`ped_nll` slightly *worse* and its forgetting slightly *better*.

It was left uncorrected deliberately. The drift is a **consequence** of the intervention —
self-distilled targets are shorter — not an independent variable. PLAN §5's fix (choose
different Tülu conversations to rebalance) is right when comparing D arms to each other,
where δ must be the only axis; here it would compensate for a downstream effect of the
intervention by perturbing the one stream that is currently byte-identical to the baseline's.
Reporting it is the more honest option. `--token_match` restores §5's behaviour.

## Acceptance checks (§9)

Split into two stages, because a broken invariant found *after* a 90-minute rewriting pass is
90 minutes wasted.

`--stage fast` (tokenizer only, ~40 s, runs before distillation):

- **check 2** — both prefix invariants. The strict one (training prefix == generation prompt
  over multi-turn contexts) **holds exactly**: 798/798 prefixes over 150 dialogues. The
  reference-carrying prompt is verified to perturb *only* a suffix of the last user message.
- **check 3** — system-message contract, both directions.
- **check 5** — δ counts exact and D1 ⊂ D2 ⊂ D3 ⊂ D4.
- **check 6** — the conditional answer-leak rule fires on **0 / 118,870** gold turns.
- **check 0** (extra) — the gate-strictness floor above.

`--stage full` (after the mix): check 1 label-span round-trip, check 7 decontamination
unchanged between gold and distilled, block layout, realised δ in the mix.

**check 4** (loss normalisation) is inherited from A1 rather than re-run: same recipe, same
pins, same PEFT wrapping.

## Pipeline

```
build_pedagogy_pool.py   (impl4's, pinned Hub revision)   -> 22,500 gold dialogues
acceptance_checks5.py --stage fast
distill_pedagogy.py      9 gated rounds, resumable        -> data/distilled_pool.jsonl
build_general_slot5.py   Tulu-3 gold, asserts it reproduces A1 exactly
mix_arm5.py              nested δ, 24/8 blocks            -> runs/D4/socrateach_sft_train.jsonl
acceptance_checks5.py --stage full
train_sft_impl5.py       923 steps, 22 adapters
impl3_compat/bridge.py --prefix impl5-
impl3_compat/nll_only.py                                   -> ped_nll rows
```

`run_impl5.py` drives all of it headless; `colab_bootstrap5.py` launches that detached on a
Colab runtime.

## Reading the results

Two arms differ by exactly one thing — the wording of the tutor turns — because the
distilled pool is written in the gold pool's row order and the substitution is positional, so
`block_order`'s seeded shuffle puts **the same dialogues in the same block positions** in D4
as in A1. D4 block *b* and A1 block *b* teach the same problems in the same order.

What a ped_nll comparison can and cannot say:

- `ped_nll` is measured on **held-out gold** dialogues (128 of them, never distilled). It
  therefore asks "how well does this model fit *gold* Socratic tutoring?" — and D4 is trained
  on paraphrases, so some gap is expected by construction and is **not** evidence of worse
  teaching.
- It says nothing about forgetting. That needs the math axis, which this run did not measure.
- It says nothing about pedagogy quality. That needs the blind judge (deviation 5).
