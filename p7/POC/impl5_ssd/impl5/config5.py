"""Run matrix and pinned constants for Impl 5 (PLAN §6, §7, §8).

Everything that governs *comparability* is imported from Impl 4's config rather than
restated, because Impl 4's arms, Impl 5's arms and Impl 3's runs all have to land on one
KL–forgetting plane. Only what Impl 5 genuinely changes is defined here: the δ axis, the
rewriting template/sampling, and the gate thresholds.

Two deliberate deviations from ``impl5_ssd/PLAN.md``, both taken to preserve comparability
with the runs that already exist. They are recorded in every manifest.

**1. 923 steps, not 937.** PLAN §6 computes 22,488 + 7,496 = 29,984 = 937 × 32. Impl 4
settled on 923 so that step numbers coincide with Impl 3's checkpoints exactly, and its A1
and A3 arms are already trained and graded on that grid. Using 937 here would mean no Impl 5
checkpoint shared a step number with any Impl 3 or Impl 4 checkpoint. PLAN §7's closing
instruction — "Using impl4's **exact** grid is what lets Impl 4 and Impl 5 arms share one
KL–forgetting plane. Do not 'improve' it." — is the governing one, and 923 is what impl4's
grid actually is.

**2. `SequentialSampler` + 24/8 blocks, not Impl 2's shuffle.** PLAN §6 argues for stock
Impl 2 batching so the δ effect is not confounded with a batching change. That is the right
call when D0 is re-run alongside. It is the wrong call here: D0 *is* impl4's A1, an already
trained 24/8 run that reproduces Impl 3's `impl2-rerun` on every axis. Re-running D0 under a
different sampler to satisfy §6 would cost a training run we do not have the budget for and
would compare D4 against a baseline that differs from A1 in *two* ways. Holding the batching
fixed at A1's makes the pedagogy targets the single moving part.

The consequence to state plainly: this file's arms are comparable to impl4's arms and to
Impl 3's runs, and are **not** a clean instance of PLAN §6's "stock Impl 2 recipe".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._impl4 import config4

# --- Inherited verbatim from Impl 4. Do not restate; import. ------------------
BASE_MODEL = config4.BASE_MODEL
SEED = config4.SEED
MAX_LEN = config4.MAX_LEN
GENERAL_FRAC = config4.GENERAL_FRAC
BLOCK_SIZE = config4.BLOCK_SIZE
PED_PER_BLOCK = config4.PED_PER_BLOCK
GEN_PER_BLOCK = config4.GEN_PER_BLOCK

N_BLOCKS = config4.N_BLOCKS            # 923 — see the module docstring
N_PED = config4.N_PED                  # 22,152
N_GEN = config4.N_GEN                  # 7,384
N_TRAIN = config4.N_TRAIN              # 29,536

CKPT_GRID = config4.CKPT_GRID          # the 22-point union grid
POC_N_BLOCKS = config4.POC_N_BLOCKS
POC_CKPT_GRID = config4.POC_CKPT_GRID
WARMUP_STEPS_APPROX = config4.WARMUP_STEPS_APPROX
TOKEN_MATCH_TOLERANCE = config4.TOKEN_MATCH_TOLERANCE

# Impl 4's A1 replay slot, reproduced bit-for-bit (see build_general_slot5.py).
OVERGENERATE = config4.OVERGENERATE

# --- The pedagogy pool. -------------------------------------------------------
# 22,500 dialogues on the pinned Hub revision; 118,870 tutor turns; max 9 turns.
# PLAN §0 says "max 8" and 119,288 rewrites — both are close but neither is exactly what
# the published pool contains, so the numbers are re-derived at build time (PLAN §9) and
# these are assertions, not inputs.
PED_POOL_EXPECTED = 22500
MAX_TUTOR_TURNS = 9                    # measured; PLAN §0's table says 8

# --- Rewriting (PLAN §3). -----------------------------------------------------
# T=1.0 untruncated: "the maximal-anchor setting, where 'the target is what π₀ would say'
# is literally true". Secondary axis here — Block R is out of scope for this build.
SAMPLING_DEFAULT = "T1"

# PLAN §3.4 sets 128 ("covers ~p99 of gold turn length at ~1.35 tok/word") and adds "no
# length-calibration loop: the problem impl4 §4 had to solve does not arise here, because
# gold and rewrite are the same kind of object."
#
# Measured: they are not the same kind of object. Under PLAN §3.2's template this rewriter
# writes a worked *explanation* — 108 words at round 1 against gold's 8.7 — so 89% of
# generations ran into the cap without emitting EOS and the round-1 keep rate was 2.1%.
# Raising the cap to 160 moved it to 2.5%: the budget was never the binding constraint.
# The fix is in the template (see REWRITE_TEMPLATES); 160 is kept because it covers gold's
# longest turns (169 words) so that "unterminated" now means the model rambled rather than
# that the budget was short.
MAX_NEW_TOKENS = 160

#: PLAN §3.2's rule, unchanged: the block is appended to the *content of the last user
#: message* rather than inserted as a new turn, so role alternation is identical to training
#: and the divergence from the training prefix is a checkable suffix
#: (chat5.assert_reference_suffix_only).
REFERENCE_JOIN = "\n\n"

# Keep rates below are measured, 200-240 dialogues per cell at max_new_tokens=160, weighted
# by real round sizes (round 1 runs 22,500 dialogues, round 7 runs 4,930). They are the
# reason the default is what it is — the templates were not ranked by reading them.
REWRITE_TEMPLATES: dict[str, str] = {
    # PLAN §3.2 verbatim ("SDFT Fig. 3 'Using', adapted"). Keep rate 7.1%. Kept so the
    # finding above is reproducible and so Block R has its reference point. Do NOT build a
    # pool with it.
    "plan": (
        "Write your next tutor message. A reference version of that message is given "
        "below —\nuse it as a guide for what to cover, but write it in your own words.\n"
        "\n### Reference tutor message:\n{gold}\n\n### Your tutor message:"
    ),
    # 56.8% — the default. States the role, the length target (gold's own word count, so it
    # scales: gold runs 8.7 words at round 1 and ~36 by round 4), the register, and the
    # prohibition the gate checks. "Write only your next message — nothing else" is doing
    # much of the work: without it the model narrates its reasoning first.
    "mirror": (
        "You are the tutor. Write only your next message to the student — nothing else.\n"
        "It must be about the same length as the reference below ({words} words), ask one "
        "guiding question, and not reveal the final answer.\n"
        "\n### Reference tutor message:\n{gold}\n\n### Your tutor message:"
    ),
    # 49.8%. Register as a sentence budget rather than a word count. Steadier across rounds
    # than `mirror` (49/48/29% vs 72/48/26%) but lower early, where most turns are.
    "brief": (
        "Write your next tutor message. Keep it short — one or two sentences that ask a "
        "single guiding question. Do not work through the step for the student and do not "
        "give away the final answer. A reference version of the message is given below; "
        "cover the same ground, but write it in your own words.\n"
        "\n### Reference tutor message:\n{gold}\n\n### Your tutor message (one or two "
        "sentences):"
    ),
    # 36.5%. Pressing on coverage holds up best at round 7 (40.5%, the highest of the four)
    # but costs the early rounds badly — it draws the model back toward explaining, and its
    # answer_leak rejections are 2-4x the others'.
    "cover": (
        "You are the tutor. Write only your next message to the student — nothing else.\n"
        "Cover everything the reference message below covers, at about the same length "
        "({words} words). Phrase it in your own words. Ask rather than explain, and do not "
        "reveal the final answer.\n"
        "\n### Reference tutor message:\n{gold}\n\n### Your tutor message (about {words} "
        "words):"
    ),
}

#: Chosen on the measured weighted keep rate above, not on taste. Recorded in
#: distill_meta.json and every arm manifest, because it is a real departure from PLAN §3.2
#: and it changes what "the target is what π₀ would say" means: π₀ *told how long to be and
#: to ask rather than explain*. Still the model's own distribution, but a conditioned slice
#: of it. Block R's R4 (reference-free) is what would price the difference; it did not run.
TEMPLATE_DEFAULT = "mirror"


def reference_block(gold: str, template: str = TEMPLATE_DEFAULT) -> str:
    return REFERENCE_JOIN + REWRITE_TEMPLATES[template].format(
        gold=gold, words=len((gold or "").split()))


# --- Gate thresholds (PLAN §4). -----------------------------------------------
@dataclass(frozen=True)
class GateThresholds:
    """Stage 2/3 thresholds.

    PLAN §4 says these are "calibrated in Stage 4, not guessed". **Stage 4 did not run in
    this build** — it needs the blind judge behind ``day1eval/llm_client.py`` and a
    ``PROMPTLENS_API_KEY``, and this run is training + pedagogy-NLL only. So these are the
    plan's own provisional values, used as-is, and the realised gate rates are reported per
    stage and per turn index so the omission is visible rather than buried.

    What that costs: the Definition of Done's "at matched pedagogy quality" is **not
    established** by this run. A δ=1 arm that wins on forgetting could have won by
    distilling away the teaching. Do not report a win without Stage 4.
    """

    word_ratio: float = 2.5            # words(t̃) > max(2.5 × words(gold), 90) -> fail
    word_floor: int = 90
    max_questions: int = 2
    list_min_lines: int = 3            # >=3 enumerated lines where gold has none
    max_sentences: int = 6
    rouge_min: float = 0.25            # paraphrase task; impl4's B2 used 0.3 for fidelity
    calibrated: bool = False
    calibration_note: str = (
        "PLAN §4 Stage 4 (blind-judge calibration) was NOT run: it requires the external "
        "judge and this build is training + ped_nll only. Thresholds are PLAN §4's "
        "provisional values. 'Matched pedagogy quality' is therefore unverified."
    )

    def as_dict(self) -> dict:
        return {
            "word_ratio": self.word_ratio,
            "word_floor": self.word_floor,
            "max_questions": self.max_questions,
            "list_min_lines": self.list_min_lines,
            "max_sentences": self.max_sentences,
            "rouge_min": self.rouge_min,
            "calibrated": self.calibrated,
            "calibration_note": self.calibration_note,
        }


DEFAULT_THRESHOLDS = GateThresholds()


# --- Run matrix (PLAN §8, Block D). -------------------------------------------
@dataclass(frozen=True)
class Arm5:
    """One Impl 5 run. Only the distilled *fraction of dialogues* differs across Block D."""

    name: str
    delta: float
    question: str = ""
    #: D0 is not trained here — it already exists as impl4's A1 (see module docstring).
    external_run: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_distilled(self) -> int:
        """Dialogue count at this δ, over the pool actually consumed by the mix."""
        return int(round(self.delta * N_PED))

    @property
    def priority_checkpoints(self) -> tuple[int, ...]:
        return CKPT_GRID       # Block D needs the whole trajectory (PLAN §8)


ARMS: dict[str, Arm5] = {
    "D0": Arm5("D0", 0.00, "vanilla Impl 2 reference locus",
               external_run="impl4-A1", aliases=("A1",)),
    "D1": Arm5("D1", 0.25, ""),
    "D2": Arm5("D2", 0.50, "the mix-ratio interior"),
    "D3": Arm5("D3", 0.75, ""),
    "D4": Arm5("D4", 1.00, "full SDFT — the intervention", aliases=("R1",)),
}

ARM_ALIASES = {a: arm.name for arm in ARMS.values() for a in arm.aliases}
ARM_CHOICES = tuple(sorted(set(ARMS) | set(ARM_ALIASES)))

ALL_ARMS = ("D0", "D1", "D2", "D3", "D4")
THREE_RUN_CUT = ("D0", "D2", "D4")     # PLAN §8 "if only three are affordable"
#: What this budget actually buys: D4 against D0=impl4-A1, which is already trained.
BUDGET_CUT = ("D4",)


def resolve_arm(name: str) -> Arm5:
    key = ARM_ALIASES.get(name, name)
    if key not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)} "
                       f"(aliases: {', '.join(ARM_ALIASES)})")
    return ARMS[key]


def n_blocks(poc: bool = False) -> int:
    return POC_N_BLOCKS if poc else N_BLOCKS


def slot_sizes(poc: bool = False) -> tuple[int, int]:
    b = n_blocks(poc)
    return b * PED_PER_BLOCK, b * GEN_PER_BLOCK


def checkpoint_grid(poc: bool = False) -> tuple[int, ...]:
    return POC_CKPT_GRID if poc else CKPT_GRID


def distilled_ids(all_ids, delta: float, seed: int = SEED) -> list[str]:
    """The δ-fraction of dialogue ids to distill, **nested** across D1 ⊂ D2 ⊂ D3 ⊂ D4.

    One seeded permutation, sliced by δ (PLAN §8): a non-monotone result across the sweep
    then means something, rather than being four independent samples.
    """
    import random

    ids = sorted(all_ids)              # sorted first, so the permutation is input-order-free
    random.Random(seed).shuffle(ids)
    return sorted(ids[:int(round(delta * len(ids)))])
