# Impl 5 — what this build actually is

`PLAN.md` is the full spec. This file records what was built, what was cut, and every place
the build knowingly departs from the plan. Read it before quoting a number from `runs/`.

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

Two facts the plan does not have:

- **Reference-block overhead**: mean 84 tokens, max 160, appended to the last user message.
- **Gate strictness floor**: running the whole gate with `t̃ := t_gold` rejects **1.38%** of
  real tutor turns (`too_many_questions` 1,283, `too_many_sentences` 355). Every reported
  fallback rate should be read against that floor — 1.38% of it is the thresholds, not the
  rewrites.

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
