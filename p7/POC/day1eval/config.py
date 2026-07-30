"""Central configuration for the day1 MRBench tutor-generation eval.

Everything a run depends on lives here so the CLI in ``generate.py`` stays thin.
Change ``DEFAULT_MODEL`` or add entries to ``MODELS`` to swap models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")        # cached MRBench json lands here
OUTPUT_DIR = os.path.join(ROOT, "outputs")   # generated responses land here


# --------------------------------------------------------------------------- #
# Dataset sources (raw GitHub URLs from kaushal0494/UnifyingAITutorEvaluation)
# --------------------------------------------------------------------------- #
# MRBench itself (V1/V2) carries human tutor responses + 8-dimension annotations.
# The V3 dev/test splits are the BEA-2025 shared-task files (4 dimensions).
# For tutor GENERATION we only consume ``conversation_history``, so any split works;
# V1 is the default because it is the original NAACL benchmark.
RAW_BASE = "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/main"
DATASETS: dict[str, str] = {
    "V1": f"{RAW_BASE}/MRBench/MRBench_V1.json",
    "V2": f"{RAW_BASE}/MRBench/MRBench_V2.json",
    "V3_dev": f"{RAW_BASE}/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json",
    "V3_test": f"{RAW_BASE}/BEA_Shared_Task_2025_Datasets/mrbench_v3_testset.json",
}
DEFAULT_DATASET = "V1"


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """One selectable model.

    Attributes
    ----------
    model_id:
        Hugging Face repo id passed to vLLM / the tokenizer.
    chat_template_kwargs:
        Extra kwargs forwarded to ``tokenizer.apply_chat_template``. Model
        specific — e.g. Qwen3 accepts ``enable_thinking`` to toggle its
        <think> reasoning traces; OLMo templates take none.
    has_chat_template:
        Whether the tokenizer ships a chat template. Base (non-instruct)
        checkpoints like ``OLMo-2-0425-1B`` do NOT, so we fall back to a plain
        text prompt (see ``prompts.render_prompt``). ``None`` = auto-detect.
    notes:
        Human-facing note shown in ``--list-models``.
    """

    model_id: str
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    has_chat_template: bool | None = None
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    # --- Default: post-trained (instruct) OLMo-2 1B ------------------------ #
    "olmo": ModelSpec(
        model_id="allenai/OLMo-2-0425-1B-Instruct",
        chat_template_kwargs={},
        has_chat_template=True,
        notes="OLMo-2 1B post-trained (SFT+DPO+RLVR). Default; has a chat template.",
    ),
    # The fully-open base checkpoint — no instruction tuning, no chat template.
    "olmo-base": ModelSpec(
        model_id="allenai/OLMo-2-0425-1B",
        chat_template_kwargs={},
        has_chat_template=False,  # base model: no chat template -> plain-text prompt
        notes="Allen AI OLMo-2 1B *base* (fully open). No chat template.",
    ),
    # --- The requested alternative ---------------------------------------- #
    "qwen": ModelSpec(
        model_id="Qwen/Qwen3-1.7B",
        # Qwen3 is a hybrid reasoning model; disable <think> traces so the tutor
        # reply is direct. Flip to True (or use --thinking) to keep reasoning.
        chat_template_kwargs={"enable_thinking": False},
        has_chat_template=True,
        notes="Qwen3 1.7B hybrid reasoning model. Thinking disabled by default.",
    ),
}
DEFAULT_MODEL = "olmo"


# --------------------------------------------------------------------------- #
# Generation defaults (override on the CLI)
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    seed: int = 0
    # Stop when the model tries to hallucinate the next dialogue turn.
    stop: list[str] = field(default_factory=lambda: ["\nStudent:", "\nTutor:", "Student:"])


# --------------------------------------------------------------------------- #
# vLLM engine defaults
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1        # bump to #GPUs on multi-GPU EC2 instances
    trust_remote_code: bool = True
    enforce_eager: bool = False          # set True if CUDA-graph capture OOMs


# --------------------------------------------------------------------------- #
# Prompting defaults
# --------------------------------------------------------------------------- #
# Whether to give the tutor the reference solution as context (real math tutors
# in MathDial had it). The system prompt still forbids revealing the final answer.
INCLUDE_SOLUTION = True


# --------------------------------------------------------------------------- #
# LLM-as-a-judge (evaluator) — TrueFoundry PromptLens gateway, OpenAI-compatible
# --------------------------------------------------------------------------- #
# The judge scores generated tutor responses on the 8 MRBench dimensions
# (see scoring.py). Calls go through the PromptLens gateway; the API key is read
# from the environment (or a local .env) — never hard-code it here.
JUDGE_GATEWAY_URL = "https://tfy.promptlens.trilogy.com/v1/chat/completions"
JUDGE_MODEL = "openai-group/gpt-5.6-sol"     # alt: "claude-group/claude-opus-4-8"
JUDGE_API_KEY_ENV = "PROMPTLENS_API_KEY"
JUDGE_MAX_TOKENS = 4000       # headroom: reasoning models spend tokens before content
JUDGE_TEMPERATURE = None      # None = omit (gpt-5.x reasoning models reject non-default)
JUDGE_WORKERS = 8             # concurrent judge requests
JUDGE_MAX_RETRIES = 5         # retry 429 / 5xx with exponential backoff
