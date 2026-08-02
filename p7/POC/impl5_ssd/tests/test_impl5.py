"""Stdlib-only tests: no GPU, no network, no tokenizer. Run with ``python -m pytest``.

These pin the behaviours Impl 5 depends on Impl 4 for (PLAN §10's stated risk of the
coupling) plus the rules that are easy to get quietly backwards — above all the
conditional-on-gold answer-leak rule, which fails *silently* when inverted: every log looks
healthy, the fallback rate just climbs on final turns and δ collapses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impl5 import answer_leak, dialogue, gate5                    # noqa: E402
from impl5.config5 import (                                        # noqa: E402
    ARMS,
    DEFAULT_THRESHOLDS,
    N_GEN,
    N_PED,
    N_TRAIN,
    distilled_ids,
    reference_block,
    resolve_arm,
    slot_sizes,
)


def make_record(n_turns=3, answer="42", did="d0"):
    msgs = [{"role": "system", "content": "SI"},
            {"role": "user", "content": "problem"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"tutor {i + 1}?"})
        if i < n_turns - 1:
            msgs.append({"role": "user", "content": f"student {i + 1}"})
    return {"messages": msgs, "dialogue_id": did, "problem_id": "p", "answer": answer,
            "source": "s", "kind": "pedagogy"}


# --- dialogue decomposition ---------------------------------------------------
@pytest.mark.parametrize("n", [3, 4, 5, 8])
def test_parse_roundtrip(n):
    r = make_record(n)
    d = dialogue.parse(r)
    assert d.n_turns == n and len(d.student) == n - 1
    assert d.with_rewritten(d.tutor)["messages"] == r["messages"]


def test_parse_rejects_broken_alternation():
    r = make_record(3)
    r["messages"].append({"role": "assistant", "content": "extra"})
    with pytest.raises(ValueError):
        dialogue.parse(r)


def test_parse_requires_system_message():
    r = make_record(3)
    r["messages"] = r["messages"][1:]
    with pytest.raises(ValueError):
        dialogue.parse(r)


def test_round_schedule():
    ds = dialogue.parse_all([make_record(3, did="a"), make_record(5, did="b")])
    assert dialogue.round_schedule(ds) == {1: 2, 2: 2, 3: 2, 4: 1, 5: 1}


def test_distill_prompt_appends_reference_to_last_user_turn():
    d = dialogue.parse(make_record(3))
    clean = d.training_messages(["X", "Y"], 3)
    ref = d.distill_messages(["X", "Y"], 3)
    assert [m["role"] for m in clean] == [m["role"] for m in ref]
    assert ref[-1]["role"] == "user"
    assert ref[-1]["content"] == clean[-1]["content"] + reference_block(d.tutor[2])
    assert all(a == b for a, b in zip(clean[:-1], ref[:-1]))


def test_distill_prompt_uses_rewritten_prefix_not_gold():
    """The §3.1 property: round r conditions on what rounds 1…r-1 accepted."""
    d = dialogue.parse(make_record(3))
    msgs = d.training_messages(["REWRITTEN-1", "REWRITTEN-2"], 3)
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == \
        ["REWRITTEN-1", "REWRITTEN-2"]


# --- answer leak, the conditional rule ----------------------------------------
def test_number_normalisation():
    assert answer_leak.normalize_number("1,200") == "1200"
    assert answer_leak.normalize_number("30.0") == "30"
    assert answer_leak.normalize_number("7.50") == "7.5"
    assert answer_leak.normalize_number("not a number") is None


def test_leak_fires_when_rewrite_reveals_and_gold_did_not():
    assert answer_leak.leaks_conditional("So it is 42.", "What comes next?", "42") \
        == "answer_leak_value"


def test_leak_does_not_fire_when_gold_also_reveals():
    """51.8%+ of gold FINAL turns state the answer. This is the case that must not fire."""
    assert answer_leak.leaks_conditional("Right, 42.", "Exactly — the answer is 42.", "42") \
        is None


def test_leak_never_fires_on_gold_against_itself():
    """PLAN §9 check 6, in miniature."""
    for gold in ("The answer is 42.", "42 it is.", "What is 6 times 7?"):
        assert answer_leak.leaks_conditional(gold, gold, "42") is None


def test_leak_matches_across_number_formats():
    assert answer_leak.states_answer("that gives 1,200 dollars", "1200")
    assert answer_leak.states_answer("that gives 30 cents", "30.0")
    assert answer_leak.states_answer("it costs $25.00 each", "25")


def test_leak_respects_token_boundaries():
    """Digits inside a word are not a reveal — over-firing here costs realised δ."""
    assert not answer_leak.states_answer("look at step42 again", "42")
    assert not answer_leak.states_answer("the 42nd item", "42")
    assert answer_leak.numeric_literals("25.00") == {"25"}


def test_phrase_leak_conditional():
    assert answer_leak.leaks_conditional("So the answer is here.", "Try again.", None) \
        == "answer_leak_phrase"
    assert answer_leak.leaks_conditional("So the answer is here.",
                                         "So the answer is what we want.", None) is None


# --- the gate -----------------------------------------------------------------
def test_gate_passes_a_faithful_paraphrase():
    gold = "How much does each can cost at the bulk warehouse?"
    new = "What is the cost of a single can at the bulk warehouse?"
    v = gate5.evaluate(new, gold, "25")
    assert v.passed, v.reason


def test_gate_rejects_unterminated_generation():
    v = gate5.evaluate("a perfectly fine sentence", "a perfectly fine sentence", "1",
                       finished=False)
    assert not v.passed and v.stage == "degeneracy" and v.reason == "unterminated"


def test_gate_rejects_degenerate():
    assert gate5.evaluate("", "gold turn here", "1").stage == "degeneracy"
    assert gate5.evaluate("ok", "gold turn here", "1").stage == "degeneracy"


def _long_text(n: int) -> str:
    """``n`` distinct words. Repeating one word instead would trip the stage-0 repeated-
    10-gram rule and attribute the failure to degeneracy rather than to length."""
    return " ".join(f"w{i}" for i in range(n))


def test_gate_rejects_ballooning():
    gold = "What is next?"
    v = gate5.evaluate(_long_text(200), gold, "99")
    assert not v.passed and v.reason == "too_long"


def test_gate_word_floor_protects_short_gold():
    """2.5x a 4-word gold is 10 words — the 90-word floor is what stops that being absurd."""
    gold = "What is next?"
    v = gate5.evaluate("Let us think about what the very next step might reasonably be here.",
                       gold, "1")
    assert v.reason != "too_long"


def test_gate_rejects_too_many_questions():
    gold = "What next?"
    v = gate5.evaluate("What? Why? How? When?", gold, "1")
    assert v.reason == "too_many_questions"


def test_gate_rejects_enumerated_list_when_gold_has_none():
    # answer="99": the list markers 1/2/3 are themselves numeric literals, so a smaller
    # answer would fire the leak rule first and the list rule would never be reached.
    gold = "What is the first step?"
    v = gate5.evaluate("1. do this\n2. do that\n3. and then this", gold, "99")
    assert v.reason == "enumerated_list"


def test_gate_allows_list_when_gold_also_has_one():
    gold = "1. first\n2. second\n3. third"
    v = gate5.evaluate("1. one\n2. two\n3. three", gold, "99")
    assert v.reason != "enumerated_list"


def test_gate_rejects_low_rouge_last():
    """Stage 3 is last, so an unrelated-but-clean rewrite is attributed to intent_match."""
    v = gate5.evaluate("Completely unrelated commentary about giraffes today.",
                       "How much does each can cost at the warehouse?", "25")
    assert not v.passed and v.stage == "intent_match"


def test_gate_stage_order_is_first_failure_wins():
    """A rewrite that both leaks and balloons is attributed to the leak, not the length."""
    gold = "What next?"
    assert gate5.evaluate("It is 42. " + _long_text(200), gold, "42").stage == "answer_leak"


def test_degeneracy_outranks_length():
    """Stage 0 comes first, so a repetition loop is 'degenerate', never merely 'too_long'."""
    v = gate5.evaluate(" ".join(["padding"] * 200), "What next?", "99")
    assert v.stage == "degeneracy" and v.reason == "degenerate_repeated_10gram"


def test_summarize_counts_stages():
    vs = [gate5.evaluate("", "gold turn here", "1"),
          gate5.evaluate("How much does each can cost here?",
                         "How much does each can cost here?", "25")]
    s = gate5.summarize(vs)
    assert s["n"] == 2 and s["n_kept"] == 1 and s["by_stage"]["degeneracy"] == 1


def test_thresholds_are_flagged_uncalibrated():
    """Stage 4 did not run in this build; nothing may claim otherwise."""
    assert DEFAULT_THRESHOLDS.calibrated is False


# --- delta arithmetic ---------------------------------------------------------
def test_delta_counts_exact():
    ids = [f"d{i}" for i in range(1000)]
    for delta in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert len(distilled_ids(ids, delta)) == int(round(delta * 1000))


def test_delta_nested():
    ids = [f"d{i}" for i in range(1000)]
    sets = [set(distilled_ids(ids, d)) for d in (0.25, 0.5, 0.75, 1.0)]
    for a, b in zip(sets, sets[1:]):
        assert a <= b


def test_delta_is_input_order_independent():
    """Sorting before shuffling means a reordered pool picks the same dialogues."""
    ids = [f"d{i}" for i in range(500)]
    assert distilled_ids(ids, 0.5) == distilled_ids(list(reversed(ids)), 0.5)


def test_delta_is_seed_deterministic():
    ids = [f"d{i}" for i in range(500)]
    assert distilled_ids(ids, 0.5, 13) == distilled_ids(ids, 0.5, 13)
    assert distilled_ids(ids, 0.5, 13) != distilled_ids(ids, 0.5, 99)


# --- config coherence ---------------------------------------------------------
def test_mix_arithmetic_matches_impl4():
    assert N_PED + N_GEN == N_TRAIN == 923 * 32
    assert slot_sizes() == (N_PED, N_GEN)


def test_d0_is_not_trained_here():
    assert ARMS["D0"].external_run == "impl4-A1"
    assert ARMS["D4"].external_run is None


def test_arm_aliases():
    assert resolve_arm("R1") is ARMS["D4"]
    assert resolve_arm("A1") is ARMS["D0"]
