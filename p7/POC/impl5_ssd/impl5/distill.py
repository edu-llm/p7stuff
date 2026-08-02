"""Batched sampling from π₀ for the rewriting pass, with a per-sequence finish reason.

Why this is not just a call to ``impl4.generate.generate_targets``: that function returns
decoded text and nothing else, and Impl 5 needs to know whether each sequence *terminated*.
A rewrite that ran into ``max_new_tokens`` without emitting EOS is a sentence cut in half,
and training on it teaches the model to stop mid-word. Impl 4 never had to care — its
SuperNI targets were allowed to be long — so the information was never plumbed through.

Everything that governs the §4 invariant is still Impl 4's code: prompts are built by
``impl4.chat.generation_prompt_ids`` (token ids, never a re-templated string) and batched by
``impl4.generate._token_budget_batches``, which bounds a batch by rows *and* padded tokens so
that a long prefix travels in a small batch instead of OOMing on the ``[B, heads, L, L]``
attention scores.

π₀ is frozen for the whole pass. The rewriting is sequential across *turn positions*, not
across training steps: the model that writes turn 5 is the same base model that wrote turn 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from ._impl4 import chat as chat4, config4, generate as generate4

SAMPLING = config4.SAMPLING


@dataclass
class Sample:
    text: str
    finished: bool                     # emitted EOS rather than hitting the token cap
    n_tokens: int


def load_hf_model(model_id: str, log=print):
    """π₀ in bf16 where available, eval mode, KV cache on."""
    import torch
    from transformers import AutoModelForCausalLM

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available()
                                         else torch.float32)
    log(f"Loading {model_id} (dtype={dtype}) ...")
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    m = m.to("cuda" if torch.cuda.is_available() else "cpu")
    m.eval()
    m.config.use_cache = True
    return m


def generate_samples(
    messages_list: Sequence[list[dict]],
    model,
    tokenizer,
    sampling_name: str = "T1",
    max_new_tokens: int = 128,
    batch_size: int = 128,
    max_batch_tokens: int = 262144,
    seed: int = config4.SEED,
    log=print,
    log_every: int = 25,
) -> list[Sample]:
    """One sample per prompt, N=1, left-padded, sorted by length.

    Returns results in the caller's order. ``max_batch_tokens`` is the real safety valve:
    row count alone does not bound attention memory, because a batch of 128 six-hundred-token
    prefixes is a very different object from a batch of 128 short ones.
    """
    import torch

    sc = SAMPLING[sampling_name]
    prompt_ids = [chat4.generation_prompt_ids(tokenizer, m) for m in messages_list]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    torch.manual_seed(seed)

    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
    batches = generate4._token_budget_batches(order, prompt_ids, batch_size, max_batch_tokens)
    out: list[Sample | None] = [None] * len(prompt_ids)
    t0 = time.time()

    for b, idxs in enumerate(batches):
        chunk = [list(prompt_ids[i]) for i in idxs]
        width = max(len(c) for c in chunk)
        input_ids = torch.tensor([[pad_id] * (width - len(c)) + c for c in chunk],
                                 device=model.device)
        attn = torch.tensor([[0] * (width - len(c)) + [1] * len(c) for c in chunk],
                            device=model.device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids, attention_mask=attn,
                do_sample=sc.temperature > 0,
                temperature=sc.temperature if sc.temperature > 0 else None,
                top_k=sc.top_k if sc.top_k > 0 else 0,
                top_p=sc.top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id, pad_token_id=pad_id,
            )
        for j, i in enumerate(idxs):
            new = gen[j][width:].tolist()
            # `finished` is "EOS is present in the continuation". HF pads every sequence in
            # the batch out to the longest one with pad_id, and pad_id == eos_id here, so
            # asking "does it end with EOS" would call every short sequence finished and
            # every truly-truncated one finished too. Membership is the reliable test.
            finished = eos_id in new
            if finished:
                new = new[:new.index(eos_id)]
            out[i] = Sample(text=tokenizer.decode(new, skip_special_tokens=True).strip(),
                            finished=finished, n_tokens=len(new))
        if log_every and ((b + 1) % log_every == 0 or b + 1 == len(batches)):
            done = sum(len(x) for x in batches[:b + 1])
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (len(prompt_ids) - done) / max(rate, 1e-9)
            log(f"    batch {b + 1}/{len(batches)} | {done}/{len(prompt_ids)} prompts | "
                f"{rate:.1f}/s | elapsed {el / 60:.1f}m eta {eta / 60:.1f}m")

    return [s if s is not None else Sample("", False, 0) for s in out]
