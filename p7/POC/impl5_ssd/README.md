# Impl 5 — Self-distilled pedagogy targets (team overview)

## What changes

| | Impl 2 | Impl 4 | Impl 5 |
|---|---|---|---|
| 75% pedagogy (SocraTeach + system instruction) | gold | **unchanged** | **the model rewrites every tutor turn in its own words** |
| 25% replay (no system instruction) | Tülu-3 gold | model's own output on SuperNI prompts | **back to Tülu-3 gold** |
| LoRA / masking / LR / 1 epoch | — | + fixed 24/8 batch layout | **unchanged** |

Impl 4 cleaned up the *replay* slot. Impl 5 does the thing the original spec actually asked for:
rewrite the **tutor turns themselves** into the base model's own voice, so the fine-tune has less
distance to travel and forgets less math.

The two are independent and could be combined later. This directory does not.

## The one number that reframes everything

The pedagogy stream is 75% of the *examples* but **~89% of the label tokens** — the general slot
is only ~10.6% of what the loss actually sees, because a SocraTeach dialogue has ~5.3 tutor turns
and a Tülu conversation averages one 80-token reply.

So Impl 4 was changing ~11% of the gradient on prompts where we weren't trying to change
behaviour. **Impl 5 is changing ~89% of the gradient on exactly the prompts where we are.** Set
expectations accordingly: a big effect, and we genuinely don't know which way pedagogy will move.
That is why the pedagogy judge isn't a nice-to-have here — it's the thing that decides whether
the run counts.

## Steps

1. **Rebuild the pedagogy pool.** SocraTeach, per-dialogue system instructions, seed 13, no
   general data (we add that ourselves). Ask for 30,000 so there's headroom.
2. **Rewrite every tutor turn**, one round per turn position. ~119,000 generations, done once and
   shared by every run. Each rewrite sees the *already-rewritten* conversation so far plus the
   gold turn as a reference — so the context it was written in is the context it'll be trained
   in.
3. **Gate each rewrite.** Keep it only if it doesn't give away the answer, stays one step, and
   still means what the gold turn meant. Otherwise keep gold.
4. **Calibrate the gate against the blind judge** on ~600 turns. This is the go/no-go: if the
   rewrites are worse than gold here, stop before training anything.
5. **Mix.** 75/25 as usual, with the Tülu slot sized so the pedagogy:general *token* ratio stays
   put across runs.
6. **Train and save**, on Impl 2's recipe unchanged, with dense early checkpoints.
7. **Hand off** checkpoints + data + manifest.

## Two details worth knowing

**The gate falls back to gold, it doesn't retry.** Impl 4's B2 gate resampled and never fell back
— falling back there would have reinjected exactly the off-policy targets we were removing. Here
it's the opposite: the spec (and the SDFT paper) say fall back to gold. The cost is that the
*realised* distilled fraction is lower than the nominal one, so we measure and report both.

**The answer-leak check has to be conditional on gold.** A good tutor doesn't hand over the
answer — except at the end, after the student has produced it. We measured it: **2.3% of
mid-dialogue gold turns state the final answer, but 51.8% of final turns do.** So the rule is
"the rewrite fails if it states the answer *and the gold turn didn't*", not "the rewrite fails if
it states the answer." Get this backwards and half of all final turns fall back to gold — and
those are the highest-value turns in the dataset.

## Runs

Five runs, ~40 min each, plus one shared rewriting pass. Only the fraction of self-distilled
dialogues differs.

| Arm | Self-distilled fraction | Question it answers |
|---|---|---|
| `D0` | 0% | baseline — this is Impl 2 (has to be re-run; `curve_run`'s checkpoints are at different steps and won't pair up) |
| `D1` | 25% | |
| `D2` | 50% | how much do you need? |
| `D3` | 75% | |
| `D4` | 100% | **the actual idea** |

Nested, so `D1`'s distilled dialogues are a subset of `D2`'s and so on — a non-monotone result
then means something.

If only three fit: `D0`, `D2`, `D4`.

### Second wave (optional)

All at 100%, varying how the rewrite is produced:

| Arm | What's different | Question |
|---|---|---|
| `R2` | truncated sampling (top-k 20, top-p 0.8) | does cleaner sampling mean fewer gold fallbacks? |
| `R3` | greedy | the base model's single most likely phrasing — lowest KL, but does the tutor's style go flat? |
| `R4` | **no reference shown** | the model just writes the next tutor turn cold. The only version where the rewriting prompt is *exactly* the training prompt. |

`R4` is the one not to cut. Showing the model the gold turn is the one compromise in the whole
design — it means the rewrite was produced under a slightly different prompt than it's trained
under. `R4` is how we find out what that costs.

## What "done" looks like

Against `D0`, **at matched pedagogy quality** (blind judge, with CIs):

- less math/logic forgetting,
- lower new-task KL — the point moves **down-left** on the KL–forgetting plane,
- SI-gating still holds: with no system instruction, the model still behaves like a normal
  assistant.

All three, or it isn't done. A forgetting win bought with a pedagogy loss just means the gate was
too loose and we distilled away the teaching, not the phrasing.

---

Full build spec: [`PLAN.md`](PLAN.md). Reference implementation for the shared machinery:
[`../impl4_ssd/`](../impl4_ssd/).
