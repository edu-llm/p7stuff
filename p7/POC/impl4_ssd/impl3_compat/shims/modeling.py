"""``common.modeling`` — SHIM. Not part of the Impl-3 bundle.

The bundle ships ``common/kl.py`` and ``eval/sweep_ckpt_eval.py``, both of which import
``load_for_inference`` from ``common.modeling`` — a module the bundle does not include.
This reconstructs it. The contract, read off the call sites:

    load_for_inference(base_model_id)                       -> (base_model, tokenizer, device)
    load_for_inference(base_id, adapter_dir=p, merge=True)   -> (merged_model, tokenizer, device)

Two details are deliberate rather than incidental:

* **The tokenizer always comes from the base model**, never from the adapter dir. LoRA
  checkpoints contain no tokenizer files (Impl 3's pitfall #8), and it never changes the
  tokenizer anyway. We route through :func:`impl4.chat.load_tokenizer` so the tokenizer —
  including the chat-template fallback and the pad-token choice — is set up exactly as it is
  during Impl 4 training. If eval-time and train-time tokenization diverged, pedagogy NLL
  would be measuring the wrong thing.
* **The adapter is merged** (``merge_and_unload``) rather than left as a PEFT wrapper, because
  the KL path calls the model twice per item and a merged model avoids the adapter hook on
  every forward.

Because this is a shim, the A1 gate is what validates it: A1 is vanilla Impl 2 on the same
data as Impl 3's ``impl2-rerun``, so if the numbers land on their table this shim behaves like
the real module. See impl3_compat/README.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Written by setup_compat.py: the absolute path to impl4_ssd/, so `impl4.chat` is importable
# from inside the compat workdir.
_ROOT_FILE = Path(__file__).resolve().parent / "_impl4_root.txt"
if _ROOT_FILE.exists():
    import sys
    _impl4_root = _ROOT_FILE.read_text(encoding="utf-8").strip()
    if _impl4_root and _impl4_root not in sys.path:
        sys.path.insert(0, _impl4_root)


def _dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_for_inference(model_id: str, adapter_dir=None, merge: bool = True,
                       dtype=None, device=None):
    """Load ``model_id``, optionally apply a LoRA adapter, and return (model, tok, device).

    :param model_id: base HF model id — also the KL reference π₀.
    :param adapter_dir: optional PEFT adapter directory to apply on top.
    :param merge: fold the adapter into the base weights (default; faster for repeated forwards).
    :returns: ``(model.eval(), tokenizer, device)``
    """
    from transformers import AutoModelForCausalLM

    from impl4.chat import load_tokenizer

    dt = dtype or _dtype()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = load_tokenizer(model_id)
    if tok.pad_token_id is None:                     # generation needs a pad id
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt)
    if adapter_dir:
        from peft import PeftModel
        # No torch_dtype here: peft 0.20.0's from_pretrained has no such parameter, so it would
        # fall through **kwargs into PeftConfig.from_pretrained. The base is already in `dt` and
        # peft's default autocast_adapter_dtype handles the adapter weights.
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if merge:
            model = model.merge_and_unload()

    model = model.to(dev)
    model.eval()
    model.config.use_cache = True                    # inference only; no grad checkpointing here
    return model, tok, dev
