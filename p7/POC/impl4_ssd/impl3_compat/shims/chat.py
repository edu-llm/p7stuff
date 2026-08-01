"""``common.chat`` — SHIM. Not part of the Impl-3 bundle.

``eval/sweep_ckpt_eval.py`` calls ``make_tokenize_fn(tok, max_len)`` and feeds each held-out
row through it to build the pedagogy-NLL targets. The bundle does not include the module, so
this supplies it — by delegating to :func:`impl4.chat.make_tokenize_fn`, which *is*
``ORCD-SFT/train_sft.py:157`` imported by path rather than copied.

That delegation is the point. Pedagogy NLL is the mean per-token NLL over the unmasked label
span, so the number is defined by the masking: which tokens are labels, whether the assistant
header is masked, whether EOS is included, and where ``max_len`` truncates. Re-implementing it
here would risk measuring a slightly different quantity than Impl 4 trains on — and the same
POC lineage is what Impl 3's own ``common/chat.py`` descends from, so this is also the closest
available match to theirs.

Assumption worth naming: Impl 3's file is assumed to be the same POC-lineage function. It was
not in the bundle, so this is unverified by inspection — the A1 gate is what tests it. If A1's
``ped_nll`` lands on Impl 3's 0.862 the masking agrees; if it is off while KL and math match,
this shim is the first thing to suspect. Ask them for ``common/chat.py`` to remove the doubt.
"""

from __future__ import annotations

from pathlib import Path

# Written by setup_compat.py: absolute path to impl4_ssd/, so `impl4.chat` is importable
# from inside the compat workdir.
_ROOT_FILE = Path(__file__).resolve().parent / "_impl4_root.txt"
if _ROOT_FILE.exists():
    import sys
    _impl4_root = _ROOT_FILE.read_text(encoding="utf-8").strip()
    if _impl4_root and _impl4_root not in sys.path:
        sys.path.insert(0, _impl4_root)


def make_tokenize_fn(tokenizer, max_len: int):
    """Impl 2's assistant-only masking, unmodified (see module docstring)."""
    from impl4.chat import make_tokenize_fn as _impl4_make_tokenize_fn

    return _impl4_make_tokenize_fn(tokenizer, max_len)
