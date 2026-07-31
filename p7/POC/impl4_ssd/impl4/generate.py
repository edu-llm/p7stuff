"""Sampling from π₀ over the SuperNI prompt pool (PLAN §4).

vLLM is the default (a 1B model over ~8.6k prompts is a few minutes); batched HF
``generate`` is the fallback (~20-30 min on the L40S). Both are driven from
``prompt_token_ids`` produced by :func:`impl4.chat.generation_prompt_ids`, so
neither backend ever re-templates a raw string — that is the §4 invariant.

π₀ is frozen and the anchor is *deliberately* stale (PLAN §4): do not regenerate
mid-run, and do not describe this as on-policy self-distillation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .chat import generation_prompt_ids, load_tokenizer
from .config import BASE_MODEL, SEED, SamplingConfig


@dataclass
class GenerationResult:
    texts: list[str]
    backend: str
    n_prompts: int
    mean_output_chars: float


def _resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import vllm  # noqa: F401
        return "vllm"
    except Exception:
        return "hf"


@dataclass
class Engine:
    """A loaded sampler, so B2's resampling rounds do not reload the model each try."""

    backend: str
    llm: object | None = None
    model: object | None = None
    tokenizer: object | None = None


def build_engine(backend: str = "auto", model_id: str = BASE_MODEL, seed: int = SEED,
                 gpu_memory_utilization: float = 0.85, log=print) -> Engine:
    chosen = _resolve_backend(backend)
    tok = load_tokenizer(model_id)
    if chosen == "vllm":
        from vllm import LLM
        log(f"Loading vLLM engine for {model_id} ...")
        return Engine("vllm", llm=LLM(model=model_id, dtype="bfloat16", seed=seed,
                                      gpu_memory_utilization=gpu_memory_utilization),
                      tokenizer=tok)
    if chosen == "hf":
        import torch
        from transformers import AutoModelForCausalLM
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float32
        log(f"Loading HF model {model_id} (dtype={dtype}) ...")
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        m = m.to("cuda" if torch.cuda.is_available() else "cpu")
        return Engine("hf", model=m, tokenizer=tok)
    raise ValueError(f"unknown backend {backend!r} (use vllm|hf|auto)")


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------
def _generate_vllm(prompt_ids: Sequence[Sequence[int]], sampling: SamplingConfig,
                   max_tokens: int, model_id: str, seed: int,
                   gpu_memory_utilization: float, llm=None, log=print) -> list[str]:
    from vllm import LLM, SamplingParams

    owns_llm = llm is None
    if owns_llm:
        llm = LLM(model=model_id, dtype="bfloat16", seed=seed,
                  gpu_memory_utilization=gpu_memory_utilization,
                  enforce_eager=False)
    params = SamplingParams(
        n=1,
        temperature=sampling.temperature,
        # vLLM sentinels: top_k=-1 and top_p=1.0 both mean "no truncation".
        top_k=sampling.top_k if sampling.top_k > 0 else -1,
        top_p=sampling.top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    log(f"  vLLM: {len(prompt_ids)} prompts | {params}")

    # Every branch below passes *token ids*. Handing vLLM a raw string and letting it
    # re-template would break the §4 invariant, so there is deliberately no string
    # fallback here — if all three fail, that is a hard error.
    ids = [list(p) for p in prompt_ids]
    outs = None
    errors: list[str] = []

    def _tokens_prompt_cls():
        try:
            from vllm import TokensPrompt          # re-exported in newer vLLM
            return TokensPrompt
        except ImportError:
            from vllm.inputs import TokensPrompt   # its original home
            return TokensPrompt

    for label, call in (
        ("TokensPrompt", lambda: llm.generate([_tokens_prompt_cls()(prompt_token_ids=i)
                                               for i in ids], params)),
        ("dict prompts", lambda: llm.generate([{"prompt_token_ids": i} for i in ids],
                                              params)),
        ("prompt_token_ids kwarg", lambda: llm.generate(prompt_token_ids=ids,
                                                        sampling_params=params)),
    ):
        try:
            outs = call()
            log(f"  vLLM input style: {label}")
            break
        except (ImportError, TypeError, AttributeError, ValueError) as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")

    if outs is None:
        raise RuntimeError(
            "could not pass prompt_token_ids to this vLLM version; tried:\n  "
            + "\n  ".join(errors)
            + "\nDo NOT work around this by passing prompt strings — vLLM would "
              "re-apply the chat template and the targets would no longer be on-policy "
              "w.r.t. pi_0 (PLAN §4). Use --backend hf instead."
        )
    texts = [o.outputs[0].text for o in outs]
    if owns_llm:
        del llm
    return texts


# ---------------------------------------------------------------------------
# HF fallback
# ---------------------------------------------------------------------------
def _generate_hf(prompt_ids: Sequence[Sequence[int]], sampling: SamplingConfig,
                 max_tokens: int, model_id: str, seed: int, batch_size: int,
                 model=None, tokenizer=None, log=print) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM

    tok = tokenizer or load_tokenizer(model_id)
    owns_model = model is None
    if owns_model:
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(seed)

    # Sort by length so each batch pads little, then restore the original order.
    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
    texts: list[Optional[str]] = [None] * len(prompt_ids)
    n_batches = math.ceil(len(order) / batch_size)

    for b in range(n_batches):
        idxs = order[b * batch_size:(b + 1) * batch_size]
        chunk = [list(prompt_ids[i]) for i in idxs]
        width = max(len(c) for c in chunk)
        # Left padding: decoder-only generation must have the prompt flush right.
        input_ids = torch.tensor([[pad_id] * (width - len(c)) + c for c in chunk])
        attn = torch.tensor([[0] * (width - len(c)) + [1] * len(c) for c in chunk])
        input_ids, attn = input_ids.to(model.device), attn.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                do_sample=True,
                temperature=sampling.temperature,
                top_k=sampling.top_k if sampling.top_k > 0 else 0,   # HF: 0 disables
                top_p=sampling.top_p,
                max_new_tokens=max_tokens,
                eos_token_id=tok.eos_token_id, pad_token_id=pad_id,
            )
        for j, i in enumerate(idxs):
            texts[i] = tok.decode(out[j][width:], skip_special_tokens=True)
        if (b + 1) % 20 == 0 or b + 1 == n_batches:
            log(f"  HF generate: batch {b + 1}/{n_batches}")

    if owns_model:
        del model
    return [t or "" for t in texts]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_targets(
    messages_list: Sequence[list[dict]],
    sampling: SamplingConfig,
    max_tokens: int,
    model_id: str = BASE_MODEL,
    backend: str = "auto",
    seed: int = SEED,
    batch_size: int = 32,
    gpu_memory_utilization: float = 0.85,
    tokenizer=None,
    llm=None,
    model=None,
    engine: "Engine | None" = None,
    log=print,
) -> GenerationResult:
    """One sample per prompt (N=1; PLAN §4: "the paper shows one sample suffices")."""
    if engine is not None:
        backend, llm, model = engine.backend, engine.llm, engine.model
        tokenizer = tokenizer or engine.tokenizer
    tok = tokenizer or load_tokenizer(model_id)
    prompt_ids = [generation_prompt_ids(tok, m) for m in messages_list]

    chosen = _resolve_backend(backend)
    log(f"Generating {len(prompt_ids)} targets | backend={chosen} | "
        f"{sampling.as_dict()} | max_tokens={max_tokens}")

    if chosen == "vllm":
        texts = _generate_vllm(prompt_ids, sampling, max_tokens, model_id, seed,
                               gpu_memory_utilization, llm=llm, log=log)
    elif chosen == "hf":
        texts = _generate_hf(prompt_ids, sampling, max_tokens, model_id, seed,
                             batch_size, model=model, tokenizer=tok, log=log)
    else:
        raise ValueError(f"unknown backend {chosen!r} (use vllm|hf|auto)")

    mean_chars = sum(len(t) for t in texts) / max(1, len(texts))
    return GenerationResult(texts=texts, backend=chosen, n_prompts=len(prompt_ids),
                            mean_output_chars=mean_chars)
