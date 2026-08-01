"""Run matrix, sampling grid and the fixed numbers from PLAN.md.

Nothing here is discovered at runtime — these are the constants the plan pins, in
one place, so a manifest can be diffed against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Base model. π₀, frozen. Generation and training both use this. -----------
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

# --- Impl 2 training config we must NOT change (PLAN §8.4). -------------------
SEED = 13
MAX_LEN = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
NUM_EPOCHS = 1.0
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4

# --- Mix arithmetic (PLAN §6). ------------------------------------------------
GENERAL_FRAC = 0.25          # φ. Not swept (PLAN §9).
BLOCK_SIZE = 32              # per_device_batch 8 × grad_accum 4
PED_PER_BLOCK = 24
GEN_PER_BLOCK = 8

# 923, not PLAN §6's 937, so step numbers line up with Impl 3's checkpoints exactly.
# Impl 3 trains 29,509 usable rows at effective batch 32 -> 923 steps; whole 32-example
# blocks put us at 29,536 rows for the same 923 steps (+0.09% of rows, identical 75/25
# ratio). Comparing per-checkpoint curves is worth more than the 14 extra steps, and a
# step-923 point that means the same thing on both sides is the whole objective.
# See impl3_compat/README.md; PLAN §6's arithmetic is otherwise unchanged.
N_BLOCKS = 923
N_PED = N_BLOCKS * PED_PER_BLOCK      # 22,152
N_GEN = N_BLOCKS * GEN_PER_BLOCK      # 7,384
N_TRAIN = N_PED + N_GEN               # 29,536
PED_POOL_TARGET = 22500               # the Hub dataset's pedagogy row count

# Over-generate so the degeneracy filter cannot shrink the slot below N_GEN.
OVERGENERATE = 1.15
# Prompt pool is shared by every SuperNI arm; keep margin for B2 resampling too.
SUPERNI_POOL_SIZE = 12000

# Token-matching tolerance against A1's Tülu slot (PLAN §5).
TOKEN_MATCH_TOLERANCE = 0.05

# Measured 2026-07-31 on the real A1 slot (7,496 Tülu-3 conversations, seed 13,
# max_len 1024): total 600,173 label tokens, mean 80.1, median 77, max 1007.
# PLAN §4 guesses "~300-500" for this and says to measure it first — it is ~80, which
# is what `max_tokens` calibration should be aiming at. Informational only; the
# pipeline always re-measures and writes data/tulu_reference.json.
TULU_MEAN_LABEL_TOKENS_OBSERVED = 80.1

# --- Checkpoint grid (PLAN §7 ∪ Impl 3's log grid). ---------------------------
# The union of two grids, because they answer different questions and the points are
# ~25 MB each:
#   PLAN §7 (dense early):  5,10,20,40,80,160,320,480,640,800,923
#   Impl 3 (log-spaced):    1,2,3,4,8,16,32,64,128,256,512,923
# Every Impl 3 checkpoint therefore has an exactly matching point here, and PLAN §7's
# grid survives so these runs still line up with curve_run's Impl 2 curve.
IMPL3_LOG_GRID = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 923)
PLAN7_GRID = (5, 10, 20, 40, 80, 160, 320, 480, 640, 800, 923)
CKPT_GRID = tuple(sorted(set(IMPL3_LOG_GRID) | set(PLAN7_GRID)))   # 22 points
# Steps inside warmup (warmup_ratio 0.03 × 923 ≈ 28), flagged in the manifest.
WARMUP_STEPS_APPROX = 28
# PLAN §9: Blocks T and G only need "where does this arm land". Chosen from
# IMPL3_LOG_GRID so the priority points are directly comparable rather than interpolated.
PRIORITY_CKPTS_BLOCK_TG = (16, 128, 923)

# --poc smoke run (PLAN §11.7): ~2,000 examples = 63 blocks.
POC_N_BLOCKS = 63
POC_CKPT_GRID = (5, 10, 20, 40, 63)


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding config for the self-distilled replay stream (PLAN §2.1).

    ``top_k=0`` / ``top_p=1.0`` mean "no truncation" and are normalised to each
    backend's own sentinel in :mod:`impl4.generate`.
    """

    name: str
    temperature: float
    top_k: int
    top_p: float
    isolates: str

    @property
    def truncated(self) -> bool:
        return self.top_k > 0 or self.top_p < 1.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "T_train": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "rho_train": "k=%d,p=%g" % (self.top_k, self.top_p) if self.truncated else "none",
        }


SAMPLING: dict[str, SamplingConfig] = {
    "T1": SamplingConfig("T1", 1.0, 0, 1.0, "pure anchor — no reshaping, no compression"),
    "T2": SamplingConfig("T2", 1.0, 20, 0.8, "support compression alone"),
    "T3": SamplingConfig("T3", 1.3, 20, 0.8, "+ moderate reshaping"),
    "T4": SamplingConfig("T4", 1.6, 20, 0.8, "the paper's Qwen3-Instruct recommended point"),
}


@dataclass(frozen=True)
class Arm:
    """One training run. Only the replay slot differs across arms."""

    name: str
    block: str                       # "S" (replay source) | "T" (sampling) | "G" (gating)
    sigma: float                     # fraction of the replay slot that is self-generated
    gold_source: Optional[str]       # "tulu" | "superni" | None — source of the (1-σ) part
    sampling: Optional[str]          # key into SAMPLING for the σ part
    gated: bool = False              # B2's ROUGE-L / substring gate (PLAN §9)
    question: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    # δ is 0 for every arm: pedagogy targets are never self-distilled (PLAN §1).
    delta: float = 0.0

    @property
    def sampling_config(self) -> Optional[SamplingConfig]:
        return SAMPLING[self.sampling] if self.sampling else None

    @property
    def priority_checkpoints(self) -> tuple[int, ...]:
        """PLAN §9: all 11 for Block S, {20,160,937} for Blocks T and G."""
        return CKPT_GRID if self.block == "S" else PRIORITY_CKPTS_BLOCK_TG

    @property
    def n_ssd(self) -> int:
        return int(round(N_GEN * self.sigma))

    @property
    def n_gold(self) -> int:
        return N_GEN - self.n_ssd


ARMS: dict[str, Arm] = {
    # Block S — replay source. Sampling held at T1 so σ is the only thing moving.
    "A1": Arm("A1", "S", 0.0, "tulu", None,
              question="vanilla Impl 2 reference locus"),
    "A2": Arm("A2", "S", 0.0, "superni", None,
              question="prompt shift, or self-generation?"),
    "A3": Arm("A3", "S", 1.0, None, "T1", aliases=("T1",),
              question="the intervention (doubles as Block T's first point)"),
    "A4": Arm("A4", "S", 0.5, "tulu", "T1",
              question="how much self-generation is needed?"),
    # Block T — sampling config. All σ=1; T1 is the same run as A3.
    "T2": Arm("T2", "T", 1.0, None, "T2",
              question="does truncation alone help?"),
    "T3": Arm("T3", "T", 1.0, None, "T3",
              question="interior point — trend or peak?"),
    "T4": Arm("T4", "T", 1.0, None, "T4",
              question="does the paper's tuned config beat holding at 1.0?"),
    # Block G — gating.
    "B2": Arm("B2", "G", 1.0, None, "T1", gated=True,
              question="does checking the output against gold help or hurt?"),
}

ARM_ALIASES = {alias: arm.name for arm in ARMS.values() for alias in arm.aliases}
# Everything an --arm flag should accept, for argparse `choices`.
ARM_CHOICES = tuple(sorted(set(ARMS) | set(ARM_ALIASES)))

# The eight runs of PLAN §9, in build order. T1 is not listed: it *is* A3.
ALL_ARMS = ("A1", "A2", "A3", "A4", "T2", "T3", "T4", "B2")
# PLAN §12, cut 1: still answers both primary questions.
FOUR_RUN_CUT = ("A1", "A2", "A3", "T4")
# PLAN §12, cut 2.
ONE_RUN_CUT = ("A3",)


def resolve_arm(name: str) -> Arm:
    """Look an arm up by name or alias. ``resolve_arm("T1") is ARMS["A3"]``."""
    key = ARM_ALIASES.get(name, name)
    if key not in ARMS:
        raise KeyError(
            f"unknown arm {name!r}; known arms: {', '.join(ARMS)} "
            f"(aliases: {', '.join(ARM_ALIASES) or 'none'})"
        )
    return ARMS[key]


def checkpoint_grid(poc: bool = False) -> tuple[int, ...]:
    return POC_CKPT_GRID if poc else CKPT_GRID


def priority_checkpoints(arm: Arm, poc: bool = False) -> tuple[int, ...]:
    """Which grid points the eval team is asked to prioritise.

    All of them for Block S (the curve comparison needs the trajectory), and
    "early / middle / end" for Blocks T and G (those only need "where does this arm
    land"). Under ``--poc`` the grid is different, so the T/G subset is derived from
    it rather than hardcoded — otherwise the final step would be dropped.
    """
    grid = checkpoint_grid(poc)
    if arm.block == "S":
        return grid
    if not poc:
        return tuple(s for s in PRIORITY_CKPTS_BLOCK_TG if s in grid)
    return tuple(sorted({grid[len(grid) // 3], grid[-1]}))


def n_blocks(poc: bool = False) -> int:
    return POC_N_BLOCKS if poc else N_BLOCKS


def slot_sizes(poc: bool = False) -> tuple[int, int]:
    """(pedagogy, general) example counts for the ordered train file."""
    b = n_blocks(poc)
    return b * PED_PER_BLOCK, b * GEN_PER_BLOCK
