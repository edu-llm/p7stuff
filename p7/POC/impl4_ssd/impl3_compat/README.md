# impl3_compat — scoring Impl 4 on Impl 3's axes

Impl 4 does **not** reimplement the three eval axes. It runs
`eval/sweep_ckpt_eval.py` **from the Impl-3 comparability bundle, unmodified**, against Impl 4's
checkpoints. Reimplementing KL, GSM8K scoring and pedagogy NLL would produce numbers that
resemble Impl 3's; running their driver produces numbers that *are* on their axes.

```bash
python impl3_compat/setup_compat.py                    # assemble + verify the bundle assets
python impl3_compat/bridge.py                          # expose runs/<arm>/ckpt-N as out/impl4-<arm>/checkpoint-N
cd impl3_compat/work
python eval/sweep_ckpt_eval.py --runs 'out/*' --out out/ckpt_sweep_impl4.jsonl --batch 32
cd - && python impl3_compat/compare.py                 # A1 gate + merged jsonl + figure
```

## What the three axes are

| axis | what | knobs |
|---|---|---|
| `kl_new_SI`, `kl_ped_noSI` | forward KL(π₀‖π), per-token, over base-greedy continuations of 64 held-out pedagogy contexts — each truncated **before the first tutor turn** — in two conditions: canonical SI prepended, and no system message at all | 200 max new tokens |
| `math_bare`, `math_hint` | 250 GSM8K items, integer exact match, in two conditions: question alone, and question + `Put your final answer inside \boxed{ }.` Neither carries a pedagogy SI | greedy, 512 max new tokens, left-padded batches |
| `ped_nll` | mean per-token NLL of the gold tutor turns over `val[:128]` | forward passes only |

Plus `commit` (a parsable answer came out) and `deflect` (the response ends on `?`) per math
condition. Impl 3's warning stands: on this item set `commit` is contaminated — the extractor
picks a number out of a Socratic counter-question and scores it wrong — so **`deflect` is the
trustworthy refusal signal** and `acc_given_commit` should not be read when deflect is high.

## Why a bridge is needed

Their driver scans `out/<run>/checkpoint-<step>/` and enforces `epoch >= 0.99` from
`trainer_state.json` — the guard added after two of their runs crashed at ~0.28 epoch and were
graded as complete. Impl 4 writes `runs/<arm>/ckpt-<step>/` and its grid callback writes no
`trainer_state.json` at all.

`bridge.py` builds the layout they expect: one real directory per grid point with the adapter
files symlinked (nothing is copied, `runs/` is never written to), and `trainer_state.json`
synthesized **only** in the final checkpoint, **only** when `checkpoint_index.json` shows the run
reached its final grid step *and* that adapter is on disk. An arm that died mid-training is
skipped and named. The guard is honoured, not routed around.

## The two shims

The bundle's own code imports two modules the bundle does not contain. `shims/` supplies them:

| shim | why it is not a reimplementation |
|---|---|
| `common/modeling.py` — `load_for_inference` | loads base ± adapter (merged), bf16 where supported else fp16. The tokenizer always comes from the **base** model, via `impl4.chat.load_tokenizer`, so eval-time tokenization is identical to Impl 4's training-time tokenization. LoRA dirs contain no tokenizer (their pitfall #8) and LoRA never changes one. |
| `common/chat.py` — `make_tokenize_fn` | delegates to `impl4.chat.make_tokenize_fn`, which *is* `ORCD-SFT/train_sft.py:157` imported by path. Pedagogy NLL is defined by the masking, so this must be the same function the model trained under. |

`common/prompts/impl1_system_prompt.txt` is also missing from the bundle, and
`common/system_instructions.py` reads it at import time. `setup_compat.py` writes a placeholder
whose text is unmissable if it ever reaches a model. `IMPL1_SYSTEM_PROMPT` is
defined-but-never-used across the entire bundle; `CANONICAL_SI` is the only prompt the KL path
touches, and that one is hash-verified.

## Verified assets

`setup_compat.py` refuses to continue if any of these differ, because each one silently
invalidates a different column:

| asset | expected | breaks if wrong |
|---|---|---|
| `common/prompts/canonical_si.txt` | sha256 `e2bde3bb…`, 614 bytes | `kl_new_SI` is measured against a different string |
| `eval/math_eval/math_logic_prompts.jsonl` | 250 items, id-hash `995cd590` | `math_bare` / `math_hint` are a different probe |
| `data/socrateach_sft_val.jsonl` | sha256 `23d4ee3c…`, 1,724 rows | the 64 KL contexts and 128 NLL dialogues are different items |

It also asserts their stated KL invariant: `pedagogy_contexts(val, 64)` returns 64 contexts, each
a **single user turn** ending on `role: user`. Confirmed against the real file — the 64 are
exactly `val[:64]`, no rows skipped.

The protocol string every row must carry:

```
kl=ctx-first-turn;math=bare+hint@250/995cd590;ifeval=off
```

`compare.py` refuses to merge across differing stamps rather than warning.

## The A1 gate

Impl 4's **A1** arm is vanilla Impl 2 on the same data as Impl 3's `impl2-rerun`, so its final
checkpoint should reproduce their numbers:

| metric | their SFT @923 |
|---|---|
| `kl_new_SI` | 0.7607 |
| `kl_ped_noSI` | 0.1500 |
| `ped_nll` | 0.862 |
| `math_hint` | 0.212 |
| `math_bare` | 0.456 |
| `math_hint_commit` | 0.904 |
| `math_hint_deflect` | 0.476 |

One comparison validates the canonical SI, the item set, the KL truncation, the masking shim, the
modeling shim, the pinned dataset revision, and the training config simultaneously. **Run it
before spending GPU time on any other arm.** `compare.py` prints deltas as a share of the axis
range measured from their own 194 rows and flags anything above 5%.

Where a miss points:

| symptom | first suspect |
|---|---|
| `ped_nll` off, KL and math fine | the `common/chat.py` masking shim, or the dataset revision |
| `kl_*` off | `canonical_si.txt`, or the KL contexts |
| `math_*` off | the item set, or generation settings |
| everything slightly off | ordering (our 24/8 blocks vs their shuffle), or dtype |

## Residual divergences

Recorded because none are fixable without abandoning either Impl 4's design or their setup:

1. **Within-epoch ordering.** Impl 4 pre-orders the mix into 24-pedagogy/8-general blocks with a
   `SequentialSampler` (PLAN §6 — the replay stream as a per-step constraint, which is part of
   what Impl 4 *is*). Impl 3 shuffles. Most visible at steps 1-8. The A1 gate measures the cost.
2. **29,536 rows vs their 29,509**, and 22,152/7,384 vs their 22,500/7,500 — whole 32-example
   blocks force it. Identical 75/25 ratio and identical 923 steps.
3. **torch version and dtype** — theirs is 2.5.1+cu121 bf16 on an H200. Their own evidence that
   this is tolerable: a POC-lineage adapter matched `impl2-rerun` to within 1% of axis range
   across different seeds *and* fp16-vs-bf16.
4. **The two shims are reconstructions**, not their files. The gate is what validates them; ask
   them for `common/chat.py` and `common/modeling.py` to remove the assumption.
5. **Batch geometry** — `8×4` with gradient checkpointing on (Colab) vs their `32×1` off. Same
   effective batch 32, same 923 steps.

## Handing rows back

`compare.py` writes a merged JSONL in their schema. Their `plot_figure3.py` styles every run with
`variant is None` as a black X, so eight Impl 4 arms merged into their figure would render as
nine identical black curves. One line fixes it on their side:

```python
MARKER = {"a": "s", "b": "o", "impl4": "^"}
```

Until then, `compare.py`'s own two-panel figure is the one that separates the two projects.
