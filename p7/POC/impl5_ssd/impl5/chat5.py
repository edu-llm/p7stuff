"""Multi-turn extension of Impl 4's §4 invariant (PLAN §9 check 2).

``impl4.chat.training_prefix_ids`` *raises* on an assistant turn — Impl 4 only ever sampled
one-turn SuperNI prompts, so it never needed the branch. Impl 5's rewriting prompts carry the
already-rewritten prefix, so the assistant branch has to mirror ``make_tokenize_fn``
(``ORCD-SFT/train_sft.py``) exactly::

    head = enc("<|assistant|>\\n");  body = enc(content) + [eos];  ids += head + body + nl

Two checks live here, and they are not the same check.

:func:`assert_training_prefix_invariant`
    The strict one, and the one that carries the SDFT premise: for the **reference-free**
    prefix, what the sampler is handed must equal what training will contain. This is
    impl4's §4 invariant, unchanged, extended over assistant turns.

:func:`assert_reference_suffix_only`
    The honest one. The distillation prompt cannot satisfy the strict invariant — it has to
    carry the gold turn as a reference, so it is strictly longer (PLAN §3.2; the SDFT paper
    has the same gap between its Fig. 3 and Fig. 10 templates). What *is* checkable is that
    the divergence is confined to a suffix of the last user message: every token before that
    message is identical. That is what this asserts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ._impl4 import chat as chat4

ASSISTANT_HEADER = chat4.ASSISTANT_HEADER
load_tokenizer = chat4.load_tokenizer
make_tokenize_fn = chat4.make_tokenize_fn
generation_prompt_ids = chat4.generation_prompt_ids
generation_prompt_text = chat4.generation_prompt_text
label_token_count = chat4.label_token_count
label_token_counts = chat4.label_token_counts
assert_label_span_roundtrip = chat4.assert_label_span_roundtrip


def _enc(tokenizer, s: str) -> list[int]:
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def message_ids(tokenizer, messages: Sequence[dict]) -> list[int]:
    """Token ids for ``messages`` under ``make_tokenize_fn``'s rules, without BOS.

    Assistant turns are rendered the way *training* renders them (header, content, EOS,
    newline) rather than the way a chat template renders a completed turn, because the
    point of the exercise is to compare against training.
    """
    nl = _enc(tokenizer, "\n")
    ids: list[int] = []
    for m in messages:
        if m["role"] == "assistant":
            ids += (_enc(tokenizer, ASSISTANT_HEADER) + _enc(tokenizer, m["content"])
                    + [tokenizer.eos_token_id] + nl)
        else:
            tag = "<|system|>\n" if m["role"] == "system" else "<|user|>\n"
            ids += _enc(tokenizer, tag + m["content"] + "\n")
    return ids


def training_prefix_ids(tokenizer, messages: Sequence[dict]) -> list[int]:
    """Training-time ids up to and including the assistant header the model continues from.

    The multi-turn generalisation of ``impl4.chat.training_prefix_ids``: same construction,
    with the assistant branch added.
    """
    if messages and messages[-1]["role"] == "assistant":
        raise ValueError("training_prefix_ids expects a prompt ending on system/user")
    return ([tokenizer.bos_token_id] + message_ids(tokenizer, messages)
            + _enc(tokenizer, ASSISTANT_HEADER))


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def assert_training_prefix_invariant(tokenizer, messages_list: Iterable[Sequence[dict]]) -> dict:
    """impl4 §4, over multi-turn prefixes. Raises on the first mismatch."""
    n = 0
    for messages in messages_list:
        n += 1
        train_ids = training_prefix_ids(tokenizer, messages)
        gen_ids = generation_prompt_ids(tokenizer, list(messages))
        if train_ids != gen_ids:
            i = common_prefix_len(train_ids, gen_ids)
            raise AssertionError(
                "generation prompt != training prefix — targets would not be on-policy "
                f"w.r.t. pi_0 (impl4 PLAN §4, impl5 PLAN §9 check 2).\n"
                f"  diverge at token {i}\n"
                f"  training  : {tokenizer.decode(train_ids[max(0, i - 12):i + 12])!r}\n"
                f"  generation: {tokenizer.decode(gen_ids[max(0, i - 12):i + 12])!r}"
            )
        text = generation_prompt_text(tokenizer, list(messages))
        if not text.endswith(ASSISTANT_HEADER):
            raise AssertionError(f"prompt does not end with {ASSISTANT_HEADER!r}: "
                                 f"{text[-40:]!r}")
    return {"checked": n, "ok": True}


def assert_reference_suffix_only(tokenizer, clean: Sequence[dict],
                                 with_reference: Sequence[dict]) -> dict:
    """PLAN §9 check 2 — the reference block may only perturb the last user message.

    Compares the two prompts token-for-token and requires the common prefix to cover
    everything up to the start of the final user message. Anything shorter means the
    reference block changed how an *earlier* turn tokenises, which would put the sampler
    on a different context from training in a place the design does not permit.
    """
    if len(clean) != len(with_reference):
        raise AssertionError("the reference block must not add or remove a turn "
                             f"({len(clean)} vs {len(with_reference)} messages)")
    if clean[-1]["role"] != "user" or with_reference[-1]["role"] != "user":
        raise AssertionError("both prompts must end on a user turn")
    if not with_reference[-1]["content"].startswith(clean[-1]["content"]):
        raise AssertionError("the reference block must be appended to the last user "
                             "message, not spliced into it")

    clean_ids = generation_prompt_ids(tokenizer, list(clean))
    ref_ids = generation_prompt_ids(tokenizer, list(with_reference))
    # Everything strictly before the final user message, encoded the training way.
    head = [tokenizer.bos_token_id] + message_ids(tokenizer, clean[:-1])
    shared = common_prefix_len(clean_ids, ref_ids)
    if shared < len(head):
        raise AssertionError(
            f"the reference block perturbs tokens before the final user message: the two "
            f"prompts share only {shared} tokens but the preceding turns occupy "
            f"{len(head)}. Diverges at: "
            f"{tokenizer.decode(clean_ids[max(0, shared - 12):shared + 12])!r}"
        )
    return {"shared_prefix_tokens": shared, "head_tokens": len(head),
            "clean_tokens": len(clean_ids), "reference_tokens": len(ref_ids),
            "reference_overhead": len(ref_ids) - len(clean_ids), "ok": True}
