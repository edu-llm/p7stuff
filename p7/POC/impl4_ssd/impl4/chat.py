"""The §4 invariant: generation formatting must equal training formatting, exactly.

"Sample under *exactly* the chat template, prompt formatting, and assistant header
used at training time. If generation formatting differs from training formatting,
the targets are not on-policy w.r.t. π₀ and the entire premise is void."

Concretely, for ``[{"role": "user", "content": X}]`` the OLMo-2 template renders

    <|endoftext|><|user|>\\n{X}\\n<|assistant|>\\n

and ``make_tokenize_fn`` (``ORCD-SFT/train_sft.py:157``) builds

    [bos] + enc("<|user|>\\n" + X + "\\n") + enc("<|assistant|>\\n")

which is the same string. :func:`assert_prompt_invariant` checks that at both the
string and the token-id level for real records, rather than trusting the reading.

``make_tokenize_fn`` is imported from the Impl 2 trainer by path rather than copied,
so the masking used for token accounting here is byte-for-byte the masking used in
training.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from typing import Iterable

from .config import BASE_MODEL, MAX_LEN
from .paths import TRAIN_SFT_PY

IGNORE = -100
ASSISTANT_HEADER = "<|assistant|>\n"


@lru_cache(maxsize=1)
def impl2_trainer_module():
    """Import ``ORCD-SFT/train_sft.py`` as a module (the dash makes it unimportable)."""
    spec = importlib.util.spec_from_file_location("impl4_train_sft_ref", TRAIN_SFT_PY)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {TRAIN_SFT_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_tokenize_fn(tokenizer, max_len: int = MAX_LEN):
    """The Impl 2 assistant-only masking function, unmodified."""
    return impl2_trainer_module().make_tokenize_fn(tokenizer, max_len)


@lru_cache(maxsize=4)
def load_tokenizer(model_id: str = BASE_MODEL):
    """Tokenizer set up exactly as ``load_model_and_tokenizer`` sets it up.

    Including the chat-template fallback (``train_sft.py:133``): the base OLMo-2
    checkpoint ships without one, and training would silently borrow the Instruct
    template while generation crashed or used something else — precisely the
    train/serve mismatch the §4 invariant exists to rule out.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        template_src = impl2_trainer_module().TEMPLATE_SRC
        tok.chat_template = AutoTokenizer.from_pretrained(template_src).chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        raise ValueError(
            f"{model_id} has no bos_token_id, but make_tokenize_fn "
            f"(ORCD-SFT/train_sft.py:166) unconditionally prepends one — training would "
            f"fail with an opaque dtype error inside the collator. OLMo-2 uses "
            f"<|endoftext|> as BOS; check the model id."
        )
    return tok


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def training_prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Token ids of the training-time prefix up to (and including) the assistant header.

    Mirrors ``make_tokenize_fn`` exactly for the non-assistant messages, then appends
    the header the model is trained to continue from.
    """
    def enc(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    ids = [tokenizer.bos_token_id]
    for m in messages:
        if m["role"] == "assistant":
            raise ValueError("training_prefix_ids expects a prompt with no assistant turn")
        tag = "<|system|>\n" if m["role"] == "system" else "<|user|>\n"
        ids += enc(tag + m["content"] + "\n")
    return ids + enc(ASSISTANT_HEADER)


def _as_id_list(out) -> list[int]:
    """Normalise ``apply_chat_template`` output to a flat ``list[int]``.

    The return type is not stable across the versions this may run under:
    transformers 4.48 returns a plain ``list[int]``, while 5.x returns a
    ``BatchEncoding``. Getting this wrong is silent — ``list(BatchEncoding)`` yields
    the *key names* — so normalise explicitly rather than duck-type it.
    """
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):          # torch / numpy
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):   # batched-of-one
        out = out[0]
    ids = list(out)
    if not all(isinstance(i, int) for i in ids):
        raise TypeError(
            f"apply_chat_template returned {type(out).__name__} that does not normalise "
            f"to a list of token ids (first element: {ids[0]!r}). Update _as_id_list "
            f"for this transformers version before trusting the §4 invariant."
        )
    return ids


def generation_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """What we hand to the sampler. ``prompt_token_ids``, never a re-templated string."""
    return _as_id_list(tokenizer.apply_chat_template(messages, add_generation_prompt=True))


def generation_prompt_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def assert_prompt_invariant(tokenizer, messages_list: Iterable[list[dict]]) -> dict:
    """PLAN §11 check 2. Raises on the first mismatch; returns a summary otherwise."""
    n = 0
    for messages in messages_list:
        n += 1
        train_ids = training_prefix_ids(tokenizer, messages)
        gen_ids = generation_prompt_ids(tokenizer, messages)
        if train_ids != gen_ids:
            train_s = tokenizer.decode(train_ids)
            gen_s = tokenizer.decode(gen_ids)
            raise AssertionError(
                "generation prompt != training prefix — the §4 invariant is broken, "
                "generated targets would not be on-policy w.r.t. pi_0.\n"
                f"  training : {train_s!r}\n  generation: {gen_s!r}\n"
                f"  ids differ at index "
                f"{next((i for i, (a, b) in enumerate(zip(train_ids, gen_ids)) if a != b), 'len')}"
            )
        text = generation_prompt_text(tokenizer, messages)
        if not text.endswith(ASSISTANT_HEADER):
            raise AssertionError(
                f"generation prompt does not end with the training assistant header "
                f"{ASSISTANT_HEADER!r}: {text[-40:]!r}"
            )
    return {"checked": n, "assistant_header": ASSISTANT_HEADER, "ok": True}


# ---------------------------------------------------------------------------
# Label-token accounting (PLAN §5 — stream weight is token-proportional)
# ---------------------------------------------------------------------------
def label_token_count(tok_fn, record: dict) -> int:
    """Unmasked label tokens for one record, *after* the ``max_len`` truncation.

    This is the quantity ``num_items_in_batch`` sums over, so it — not example count
    — is what token-matching has to equalise.
    """
    enc = tok_fn({"messages": record["messages"]})
    return sum(1 for t in enc["labels"] if t != IGNORE)


def label_token_counts(tok_fn, records: Iterable[dict]) -> list[int]:
    return [label_token_count(tok_fn, r) for r in records]


def assert_label_span_roundtrip(tokenizer, tok_fn, record: dict,
                                max_len: int = MAX_LEN) -> dict:
    """PLAN §11 check 1 — the unmasked span decodes to exactly assistant content + EOS.

    A record long enough to hit ``max_len`` loses its tail, so there the assertion is
    the weaker (but still sufficient) "decodes to a prefix of assistant content".
    """
    enc = tok_fn({"messages": record["messages"]})
    ids, labels = enc["input_ids"], enc["labels"]
    kept = [i for i, t in zip(ids, labels) if t != IGNORE]
    got = tokenizer.decode(kept)
    expected = "".join(
        m["content"] + tokenizer.eos_token
        for m in record["messages"] if m["role"] == "assistant"
    )
    truncated = len(ids) >= max_len
    ok = expected.startswith(got) if truncated else got == expected
    if not ok:
        raise AssertionError(
            "unmasked label span does not decode to assistant content + EOS "
            f"(truncated={truncated}).\n"
            f"  got     : {got[:300]!r}\n  expected: {expected[:300]!r}"
        )
    return {"n_label_tokens": len(kept), "truncated": truncated, "ok": True}
