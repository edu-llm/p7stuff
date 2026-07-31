# Impl 4 — runbook

How to actually run the experiments in `PLAN.md`. Everything is a plain `.py` script;
there are no notebooks.

**Scope reminder.** This directory builds data, trains, and saves checkpoints. It
contains no evaluation, KL, judge, or grading code — that is another team's (PLAN
header). The deliverable is checkpoints + data + a manifest.

---

## 0. Layout

```
impl4_ssd/
  impl4/                     library (config, filters, generation, mixing, manifest)
  build_pedagogy_pool.py     step 1 — 22,500 pedagogy examples, shared by all arms
  build_prompt_pool.py       step 2a — the shared SuperNI prompt pool + held-out split
  build_general_slot.py      step 2b — one arm's replay slot (generation lives here)
  mix_and_order.py           step 3 — 24/8 block ordering -> socrateach_sft_train.jsonl
  train_sft_impl4.py         step 4 — SequentialSampler + dense checkpoint grid
  probe_loss_norm.py         PLAN §5 — which loss normalisation is actually in effect
  acceptance_checks.py       PLAN §11 — run before any full run
  run_arm.sbatch             one arm, end to end, on Slurm
  run_all.sh                 the whole run matrix
  tests/test_impl4.py        stdlib-only unit tests (~1s, no GPU/network)
  data/                      shared pools + the A1 token reference
  shared/                    superni_train_task_ids.txt, superni_heldout_prompts.jsonl
  runs/<arm>/                the per-arm deliverables (PLAN §10)
```

## 1. Environment

```bash
bash setup_env.sh                      # extends ORCD-SFT/setup_orcd_env.sh with vLLM
conda activate socrateach

# Strongly recommended: clone SuperNI once instead of streaming it per arm.
git clone --depth 1 https://github.com/allenai/natural-instructions ~/natural-instructions
export SUPERNI_DIR=~/natural-instructions
export HF_HOME=/orcd/pool/<yourpath>/hf_cache    # not your home quota
```

Sanity check with no GPU and no network:

```bash
python tests/test_impl4.py
```

## 2. Two measured facts that change how you drive this

Both were measured against the pinned SuperNI commit on 2026-07-31, over all 757
English train tasks. Reproduce either with `build_prompt_pool.py --scan_only`.

### 2a. The ≥30-word filter is far more aggressive than the plan assumes

PLAN §3 filter 3 keeps tasks whose gold answers average ≥30 whitespace words. SuperNI
is dominated by short-answer classification — **the median task's mean gold answer is
1 word**, and p90 is 12.7. Retention:

| `--min_gold_words` | tasks kept | categories |
|---|---|---|
| 15 | 66 | 24 |
| 20 | 45 | 19 |
| 25 | 22 | 12 |
| **30 (the plan's value)** | **15** | **9** |
| 50 | 7 | — |

At 30, nine of the fifteen survivors are summarization / story-generation / poem
tasks, and three of them (`task103_facts2story`, `task853_hippocorpus`,
`task1291_multi_news`) carry most of the length. PLAN §3 asks for a `Categories`
histogram precisely so this is visible — report it.

Measure before you commit; the scan caches every fetched task under
`data/superni_cache/`, so re-scanning at another threshold is free:

```bash
python build_prompt_pool.py --scan_only --instances_per_task 40
```

Do not pick a value from this table alone — §2c combines it with the token-matching
constraint, which rules some of these out.

**Why the filter is not optional.** Loss is token-normalised (§2d), so the replay
stream's share of each gradient tracks its *tokens*, not its examples. Unfiltered:

```
        NO FILTER: 748 tasks | mean gold  5.9 words | ~ 66,846 label tokens =  11% of A1
min_gold_words=25:  22 tasks | mean gold 69.6 words | ~706,948 label tokens = 118% of A1
```

At 11% of A1's replay weight, A2 would not be "Impl 2 with SuperNI gold replay" — it
would be "Impl 2 with the replay stream mostly switched off", which forgets more for a
reason that has nothing to do with gold vs self-generation. The filter also keeps A2
and A3 comparable in target *form* (one-word labels vs ~80-token generations would
differ in more than provenance), and it is what makes the SSD arms reachable at all:
`max_tokens` is a cap, so on a prompt whose natural answer is "yes" no cap setting gets
π₀ to 80 tokens.

**Caveat to carry into the handoff.** PLAN §3 says to filter on length, not domain —
but in SuperNI the two are correlated. At 25, nine of the twenty-two survivors are
Summarization or Story Composition, and those are also the longest, so they dominate by
token weight. The replay slot is therefore a summarization/generation anchor more than
a broad general-domain one. Report the Categories histogram alongside any claim about
general-domain anchoring.

### 2b. Tülu's target length is ~80 tokens, not the ~300–500 PLAN §4 expects

PLAN §4 says to measure the Tülu slot's mean target length first and "expect ~300–500".
Measured, on the real A1 slot (7,496 conversations, seed 13, `max_len=1024`):

```
A1 reference: total 600,173 label tokens | mean 80.1 | median 77 | max 1007
```

**80, not 350.** Two consequences:

* `max_tokens` for the SSD arms calibrates to ~80–100, not ~400. Generation is
  correspondingly cheap; the auto-calibration in `build_general_slot.py` finds this on
  its own, but do not hand it `--max_tokens 400` "because the plan said so".
* A2 *is* token-matchable — the concern PLAN §5 raises is real, but at the right
  threshold the numbers work out.

### 2c. Putting §2a and §2b together — pick `--min_gold_words 25`

The threshold has to satisfy two constraints at once: keep enough tasks for domain
breadth (§2a), *and* leave the SuperNI-gold pool able to reach A1's 600,173 tokens for
a 7,496-example subset (§2b). Measured across both:

| `--min_gold_words` | tasks | categories | pool | reachable token total | can hit 600,173 | headroom |
|---|---|---|---|---|---|---|
| 15 | 66 | 24 | 12,000 | 152,379 … 508,734 | **no** | 0.85× |
| 20 | 45 | 19 | 12,000 | 174,120 … 617,244 | yes (ratio 1.00000) | 1.03× |
| **25** | **22** | **12** | **12,000** | **190,434 … 840,667** | **yes (ratio 1.00000)** | **1.40×** |
| 30 (plan's value) | 15 | 9 | 9,359 | 275,895 … 921,873 | yes (ratio 1.00000) | 1.54× |

**15 fails outright** — the pool cannot reach A1's total no matter what is selected.
**20 works with only 3% headroom**, which is fragile against any change in the mix.
**25 is the recommendation**: comfortable headroom and roughly double the task and
category count of the plan's 30. So the plan's instinct was close; 25 rather than 30
buys back breadth at no cost to matching.

```bash
python build_prompt_pool.py --min_gold_words 25 --superni_dir "$SUPERNI_DIR"
```

Whatever you pick, verify the realized ratio in `runs/A2/manifest.json`
(`general_slot.token_ratio_to_A1`) rather than assuming.

If a future pool cannot bracket A1's total, `--token_reference superni_gold` is the
escape hatch: it matches the SSD arms to A2's realized total instead, making the
A2↔A3 paired control exact and leaving A1 as the external reference — which is PLAN
§5's own framing. It is **not** the default and is not needed at the numbers above.

### 2d. The loss-normalisation probe says token-mean — so §5 is the binding constraint

PLAN §5 flags that the gradient-accumulation loss fix "can silently fall back to
per-micro-batch mean when the model is PEFT-wrapped". Measured, PEFT-wrapped:

```
VERDICT: token_mean     |obs - token_mean| = 7e-06   |obs - micro_batch_mean| = 0.077
model_accepts_loss_kwargs: True   peft_wrapped: True   transformers: 5.14.1
```

So stream weight **is** token-proportional and token-matching is the requirement that
actually binds; the §6 block layout is the free belt-and-braces, not the load-bearing
mitigation. Re-run it on the cluster (`python probe_loss_norm.py --arm A1`) — the
answer depends on the installed `transformers`, and the result is recorded per arm as
`manifest.loss_normalization`.

## 3. Build the shared assets (once)

```bash
python build_pedagogy_pool.py                                  # 22,500 pedagogy examples
python build_prompt_pool.py --min_gold_words 25 --superni_dir "$SUPERNI_DIR"
python build_prompt_pool.py --min_gold_words 25 --superni_dir "$SUPERNI_DIR" --split test
```

## 4. Build the reference arm(s) first

Every other arm's replay slot is token-matched to a reference, so the reference has to
exist first. A1 always writes `data/tulu_reference.json`; under
`--token_reference superni_gold`, A2 also writes `data/superni_gold_reference.json`.

```bash
python build_general_slot.py --arm A1
python build_general_slot.py --arm A2 --token_reference superni_gold   # only in that regime
```

## 5. One arm, end to end

```bash
python build_general_slot.py --arm A3 --backend vllm   # generation (vLLM, or hf)
python mix_and_order.py      --arm A3                  # 24/8 blocks -> train file
python acceptance_checks.py  --arm A3 --with_probe     # PLAN §11 checks 1-6
python train_sft_impl4.py    --arm A3 --resume auto    # train + dense checkpoint grid
```

or, on Slurm:

```bash
ARM=A3 SUPERNI_DIR=$SUPERNI_DIR sbatch run_arm.sbatch
```

## 6. The whole matrix

```bash
./run_all.sh                 # all eight arms (PLAN §9), submitted to Slurm
./run_all.sh --cut four      # PLAN §12 cut 1: A1 A2 A3 T4
./run_all.sh --cut one       # PLAN §12 cut 2: A3 only
./run_all.sh --local         # run here, sequentially, instead of sbatch
```

| Arm | Replay slot | Question |
|---|---|---|
| `A1` | Tülu-3 gold (σ=0) | vanilla Impl 2 reference locus |
| `A2` | SuperNI **gold** | prompt shift, or self-generation? |
| `A3` = `T1` | SuperNI SSD, σ=1, `T=1.0`, no truncation | the intervention / pure anchor |
| `A4` | σ=0.5 — half Tülu gold, half SuperNI SSD | how much is needed? |
| `T2` | SSD, `T=1.0`, k=20, p=0.8 | does truncation alone help? |
| `T3` | SSD, `T=1.3`, k=20, p=0.8 | interior point — trend or peak? |
| `T4` | SSD, `T=1.6`, k=20, p=0.8 | does the paper's tuned config win? |
| `B2` | SSD at `T1`, gated against SuperNI gold | does checking the output help or hurt? |

`A3` and `T1` are **the same run** — trained once, referenced from both blocks. `T2`
is the one not to cut: without it, a `T4` win cannot be attributed to truncation
rather than reshaping.

## 7. Smoke run (PLAN §11 check 7)

A 63-block (~2,000-example) rehearsal of the entire pipeline, including that the
callback writes adapters at the expected steps:

```bash
./run_all.sh --poc --local --arms "A1 A3"
# then confirm resume works:
python train_sft_impl4.py --arm A3 --poc --resume auto
```

## 8. What lands, per arm

`runs/<arm>/` (PLAN §10):

* `ckpt-{5,10,20,40,80,160,320,480,640,800,937}/` — PEFT adapters, ~25 MB each
* `socrateach_sft_train.jsonl` — the exact ordered training file
* `general_slot.jsonl` — the replay slot with full provenance
* `manifest.json` — σ, δ=0, sampling config, realized mean output length, realized
  example **and** token ratios, gate/degeneracy drop rates, the loss-normalisation
  probe result, step count, checkpoint grid, `priority_checkpoints`, the SuperNI
  source used, retained task count, seed, `transformers` version
* `checkpoint_index.json`, `train.log`

Shared, once: `shared/superni_train_task_ids.txt`,
`shared/superni_heldout_prompts.jsonl`.

## 9. Asks of the eval team

Carried in every manifest, restated here because the first one is a hard requirement
for Block T to mean anything (PLAN §2.2, §10):

* **Hold `T_eval` fixed across all arms and record the value.** `T_train` and `T_eval`
  compose (`T_eff = T_train × T_eval`), so a `T_train` sweep read at a drifting
  `T_eval` is uninterpretable. If budget allows, evaluate the `T` arms at 2–3 `T_eval`
  values so `T_eff` is identifiable.
* Split the math metric into **format-failure rate vs. wrong-answer rate**. A missing
  `\boxed{}` and a wrong number currently score identically, and they have opposite
  implications for whether Impl 4 worked.
* Plot every arm on the **KL–forgetting plane** against A1's locus, not endpoint
  deltas.
* `priority_checkpoints` is a deliberate coverage cap, not complete coverage: all 11
  points for Block S, `{20, 160, 937}` for Blocks T and G. All 11 are saved regardless.

## 10. Things that will bite

* **Run A1 first.** Everything else is token-matched to it and will refuse to build
  without `data/tulu_reference.json`. Under `--token_reference superni_gold`, A2 has
  to come second for the same reason.
* **`--token_reference` must be the same for every arm you intend to compare.** It is
  recorded in each manifest; mixing the two silently changes what "token-matched"
  means between arms.
* **`--min_gold_words` must be the same for every arm too** — it defines the shared
  prompt pool, so changing it mid-matrix means A2 and A3 no longer draw from an
  identical pool and stop being a paired control.
* **`--poc` is sticky.** A slot built with `--poc` has 504 examples, not 7,496;
  `mix_and_order.py` will refuse to mix it into a full run and vice versa.
* **The 5/10/20 checkpoints sit inside warmup** (`warmup_ratio=0.03 × 937 ≈ 28`). That
  is intentional — it is where the damage happens — but they are not points on the
  cosine schedule and should not be read as such.
* **Anchor staleness is correct.** π₀ is frozen, so by step 900 the replay data is far
  from θ_t. Do not "fix" this by regenerating mid-run, and do not describe Impl 4 as
  on-policy self-distillation.
* **Token-match and degeneracy rate are per-arm.** `T` and `ρ` change output length
  and junk rate, so a `T` comparison across arms with different realized token weights
  is not a `T` comparison. Every arm's manifest records both ratios; check them.
