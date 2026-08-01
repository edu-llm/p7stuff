#!/usr/bin/env python
"""Unit tests for the parts of Impl 4 that do not need a GPU, a model, or the network.

    python tests/test_impl4.py          # stdlib only, ~1s
    pytest tests/test_impl4.py -v       # if pytest is available

Everything tokenizer- or generation-dependent is covered by ``acceptance_checks.py``
against real data instead.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impl4 import config, degeneracy, gate, mixing, ngram, superni, textutil  # noqa: E402


# ---------------------------------------------------------------------------
class TestArmRegistry(unittest.TestCase):
    def test_eight_runs_and_the_t1_alias(self):
        self.assertEqual(len(config.ALL_ARMS), 8)
        self.assertIs(config.resolve_arm("T1"), config.ARMS["A3"])
        self.assertEqual(config.resolve_arm("T1").name, "A3")
        with self.assertRaises(KeyError):
            config.resolve_arm("A9")

    def test_sigma_splits_the_slot(self):
        cases = {"A1": (7384, 0), "A2": (7384, 0), "A3": (0, 7384),
                 "A4": (3692, 3692), "T4": (0, 7384), "B2": (0, 7384)}
        for name, (gold, ssd) in cases.items():
            arm = config.resolve_arm(name)
            self.assertEqual((arm.n_gold, arm.n_ssd), (gold, ssd), name)

    def test_delta_is_zero_everywhere(self):
        # PLAN §1: pedagogy targets are never self-distilled.
        for arm in config.ARMS.values():
            self.assertEqual(arm.delta, 0.0, arm.name)

    def test_priority_checkpoints(self):
        # Every grid point for Block S; a comparable early/middle/end triple for T and G,
        # all three drawn from Impl 3's log grid so they need no interpolation.
        self.assertEqual(config.resolve_arm("A3").priority_checkpoints, config.CKPT_GRID)
        self.assertEqual(config.resolve_arm("T4").priority_checkpoints, (16, 128, 923))
        self.assertEqual(config.resolve_arm("B2").priority_checkpoints, (16, 128, 923))
        self.assertTrue(set(config.PRIORITY_CKPTS_BLOCK_TG) <= set(config.IMPL3_LOG_GRID))

    def test_priority_checkpoints_stay_inside_the_poc_grid(self):
        for name in config.ALL_ARMS:
            arm = config.resolve_arm(name)
            for poc in (False, True):
                grid = config.checkpoint_grid(poc)
                pri = config.priority_checkpoints(arm, poc)
                self.assertTrue(pri, f"{name} poc={poc} has no priority checkpoints")
                self.assertTrue(set(pri) <= set(grid), f"{name} poc={poc}: {pri} !<= {grid}")
                # The final step must always be prioritised — it is "where this arm lands".
                self.assertIn(grid[-1], pri, f"{name} poc={poc} drops the final step")

    def test_mix_arithmetic_matches_the_plan(self):
        # 923 blocks, not PLAN §6's 937, so step numbers line up with Impl 3's checkpoints.
        self.assertEqual(config.N_PED, 22152)
        self.assertEqual(config.N_GEN, 7384)
        self.assertEqual(config.N_TRAIN, 29536)
        self.assertEqual(config.N_TRAIN // config.BLOCK_SIZE, 923)
        self.assertAlmostEqual(config.N_GEN / config.N_TRAIN, 0.25, places=4)

    def test_grid_covers_both_source_grids(self):
        """The union grid is what makes per-checkpoint comparison possible at all.

        Dropping a point from either source grid silently turns a comparison into an
        interpolation, so assert containment rather than the literal 22-point list.
        """
        self.assertTrue(set(config.IMPL3_LOG_GRID) <= set(config.CKPT_GRID))
        self.assertTrue(set(config.PLAN7_GRID) <= set(config.CKPT_GRID))
        self.assertEqual(config.CKPT_GRID[-1], config.N_BLOCKS)
        self.assertEqual(config.IMPL3_LOG_GRID[-1], config.N_BLOCKS)
        self.assertEqual(list(config.CKPT_GRID), sorted(set(config.CKPT_GRID)))

    def test_sampling_grid(self):
        self.assertFalse(config.SAMPLING["T1"].truncated)
        for k in ("T2", "T3", "T4"):
            self.assertTrue(config.SAMPLING[k].truncated)
            self.assertEqual((config.SAMPLING[k].top_k, config.SAMPLING[k].top_p), (20, 0.8))
        self.assertEqual([config.SAMPLING[k].temperature for k in ("T1", "T2", "T3", "T4")],
                         [1.0, 1.0, 1.3, 1.6])


# ---------------------------------------------------------------------------
class TestDegeneracy(unittest.TestCase):
    def test_exact_rules(self):
        self.assertEqual(degeneracy.degeneracy_reason("   "), "empty")
        self.assertEqual(degeneracy.degeneracy_reason("one two"), "too_few_tokens")
        # single line, < 8 chars, but >= 3 tokens
        self.assertEqual(degeneracy.degeneracy_reason("a b c"), "single_short_line")
        # a 10-gram repeated 5 times -> drop; 4 times -> keep
        unit = " ".join(str(i) for i in range(10))
        self.assertEqual(degeneracy.degeneracy_reason(" ".join([unit] * 6)),
                         "repeated_10gram")
        self.assertIsNone(degeneracy.degeneracy_reason("a" * 20 + "\nsecond line here"))

    def test_normal_text_survives(self):
        text = ("Photosynthesis converts light energy into chemical energy stored in "
                "glucose, releasing oxygen as a by-product.")
        self.assertIsNone(degeneracy.degeneracy_reason(text))

    def test_filter_reports_counts(self):
        kept, reasons = degeneracy.filter_outputs(["", "ok this is fine and long enough",
                                                   "hi"])
        self.assertEqual(kept, [1])
        self.assertEqual(reasons["empty"], 1)
        self.assertEqual(reasons["too_few_tokens"], 1)


# ---------------------------------------------------------------------------
class TestGate(unittest.TestCase):
    def test_substring_passes_regardless_of_rouge(self):
        gold = "Paris"
        pred = "After weighing the options, I would say the answer is Paris, the capital."
        ok, how, _ = gate.gate_result(pred, gold)
        self.assertTrue(ok)
        self.assertEqual(how, "substring")

    def test_rouge_l_threshold(self):
        gold = "the quick brown fox jumps over the lazy dog"
        self.assertGreaterEqual(gate.rouge_l_f1("the quick brown fox jumps over the lazy dog",
                                                gold), 0.999)
        self.assertTrue(gate.gate_passed("a quick brown fox jumped over a lazy dog", gold))
        self.assertFalse(gate.gate_passed("completely unrelated statement about turbines",
                                          gold))

    def test_lcs(self):
        self.assertEqual(gate.lcs_length(list("abcde"), list("ace")), 3)
        self.assertEqual(gate.lcs_length([], ["a"]), 0)

    def test_max_tries_is_four(self):
        self.assertEqual(gate.MAX_TRIES, 4)
        self.assertEqual(gate.GATE_THRESHOLD, 0.3)


# ---------------------------------------------------------------------------
class TestNGram(unittest.TestCase):
    def test_long_reference_uses_13grams(self):
        idx = ngram.NGramIndex(n=13)
        ref = " ".join(f"w{i}" for i in range(30))
        idx.add(ref)
        self.assertIsNotNone(idx.hit("prefix text " + " ".join(f"w{i}" for i in range(5, 20))))
        self.assertIsNone(idx.hit(" ".join(f"w{i}" for i in range(5, 15))))  # only 10 tokens

    def test_short_reference_uses_exact_phrase(self):
        idx = ngram.NGramIndex(n=13)
        idx.add("What is the capital of Australia?")
        self.assertIsNotNone(idx.hit("Trivia round: what is the capital of Australia -- "
                                     "answer in one word."))
        self.assertIsNone(idx.hit("What is the capital of Austria?"))

    def test_normalization_ignores_punctuation_and_case(self):
        idx = ngram.NGramIndex(n=13)
        idx.add("Solve for x: 3x + 6 = 21.")
        self.assertIsNotNone(idx.hit("please SOLVE FOR X 3x 6 21 now"))

    def test_load_eval_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            p.write_text(json.dumps({"id": "a", "prompt": "hello world"}) + "\n"
                         + json.dumps({"id": "b", "prompt": "second one"}) + "\n")
            self.assertEqual(ngram.load_eval_prompts(p), ["hello world", "second one"])


# ---------------------------------------------------------------------------
class TestTokenMatching(unittest.TestCase):
    def test_hits_the_target_when_the_pool_allows(self):
        counts = [10 + (i % 40) for i in range(400)]
        target = 100 * 30
        keep, stats = mixing.token_matched_select(counts, 100, target, seed=13)
        self.assertEqual(len(keep), 100)
        self.assertEqual(len(set(keep)), 100)
        self.assertEqual(sum(counts[i] for i in keep), stats["realized_total"])
        self.assertLessEqual(abs(stats["realized_total"] - target) / target, 0.05)
        self.assertTrue(stats["within_tolerance"])

    def test_is_deterministic(self):
        counts = [7 + (i * 13) % 91 for i in range(300)]
        a, _ = mixing.token_matched_select(counts, 80, 80 * 50, seed=13)
        b, _ = mixing.token_matched_select(counts, 80, 80 * 50, seed=13)
        c, _ = mixing.token_matched_select(counts, 80, 80 * 50, seed=14)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_no_target_means_plain_shuffled_prefix(self):
        counts = list(range(50))
        keep, stats = mixing.token_matched_select(counts, 10, None, seed=13)
        self.assertEqual(len(keep), 10)
        self.assertEqual(stats["swaps"], 0)

    def test_refuses_an_impossible_request(self):
        with self.assertRaises(ValueError):
            mixing.token_matched_select([1, 2, 3], 5, 6)


# ---------------------------------------------------------------------------
def _ped(i):
    return {"kind": "pedagogy", "dialogue_id": f"p{i}", "messages": [
        {"role": "system", "content": "si"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}]}


def _gen(i, kind="general_ssd"):
    return {"kind": kind, "dialogue_id": f"g{i}", "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}]}


class TestBlockOrdering(unittest.TestCase):
    def test_layout_is_24_then_8(self):
        ped = [_ped(i) for i in range(24 * 5)]
        gen = [_gen(i) for i in range(8 * 5)]
        ordered = mixing.block_order(ped, gen, 5, seed=13)
        self.assertEqual(len(ordered), 160)
        info = mixing.verify_block_layout(ordered)
        self.assertEqual(info["n_blocks"], 5)
        for b in range(5):
            chunk = ordered[b * 32:(b + 1) * 32]
            self.assertTrue(all(mixing.is_pedagogy(r) for r in chunk[:24]))
            self.assertTrue(all(not mixing.is_pedagogy(r) for r in chunk[24:]))

    def test_content_is_shuffled_but_structure_is_not(self):
        ped = [_ped(i) for i in range(24 * 4)]
        gen = [_gen(i) for i in range(8 * 4)]
        a = mixing.block_order(ped, gen, 4, seed=13)
        b = mixing.block_order(ped, gen, 4, seed=13)
        c = mixing.block_order(ped, gen, 4, seed=99)
        self.assertEqual([r["dialogue_id"] for r in a], [r["dialogue_id"] for r in b])
        self.assertNotEqual([r["dialogue_id"] for r in a], [r["dialogue_id"] for r in c])
        mixing.verify_block_layout(c)  # structure holds under any seed

    def test_uses_every_example_exactly_once(self):
        ped = [_ped(i) for i in range(24 * 3)]
        gen = [_gen(i) for i in range(8 * 3)]
        ordered = mixing.block_order(ped, gen, 3, seed=13)
        ids = [r["dialogue_id"] for r in ordered]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(ids), {r["dialogue_id"] for r in ped + gen})

    def test_verify_catches_a_swapped_record(self):
        ped = [_ped(i) for i in range(24 * 2)]
        gen = [_gen(i) for i in range(8 * 2)]
        ordered = mixing.block_order(ped, gen, 2, seed=13)
        ordered[0], ordered[31] = ordered[31], ordered[0]
        with self.assertRaises(AssertionError):
            mixing.verify_block_layout(ordered)

    def test_verify_catches_a_system_message_on_a_general_record(self):
        ped = [_ped(i) for i in range(24)]
        gen = [_gen(i) for i in range(8)]
        gen[0]["messages"].insert(0, {"role": "system", "content": "oops"})
        ordered = mixing.block_order(ped, gen, 1, seed=13)
        with self.assertRaises(AssertionError):
            mixing.verify_block_layout(ordered)

    def test_refuses_an_underfilled_pool(self):
        with self.assertRaises(ValueError):
            mixing.block_order([_ped(i) for i in range(10)], [_gen(i) for i in range(8)], 1)


# ---------------------------------------------------------------------------
class TestSuperNIFilters(unittest.TestCase):
    def test_contamination_patterns(self):
        hit = superni.contamination_hit({"Source": ["gsm8k"]}, "task100_x")
        self.assertEqual(hit, "gsm8k")
        self.assertTrue(superni.contamination_hit({"Source": ["BIG-Bench"]}, "task1_y"))
        self.assertTrue(superni.contamination_hit(
            {"Source": ["quoref"]}, "task700_mmmlu_answer_generation_high_school_math"))
        self.assertIsNone(superni.contamination_hit({"Source": ["quoref"]}, "task001_quoref"))

    def test_english_only(self):
        en = {"Input_language": ["English"], "Output_language": ["English"],
              "Instruction_language": ["English"]}
        self.assertTrue(superni.is_english_task(en))
        self.assertFalse(superni.is_english_task({**en, "Output_language": ["Spanish"]}))
        self.assertFalse(superni.is_english_task({}))

    def test_gold_length(self):
        insts = [{"output": ["a " * 40]}, {"output": ["b " * 40]}]
        self.assertGreater(superni.mean_gold_words(insts), 30)
        self.assertLess(superni.mean_gold_words([{"output": ["short"]}]), 30)

    def test_user_message_puts_the_definition_in_the_user_turn(self):
        msg = superni.user_message({"definition": "Do the thing.", "input": "Here is X."})
        self.assertEqual(msg, "Do the thing.\n\nHere is X.")

    def test_streaming_reader_matches_a_full_parse(self):
        doc = {
            "Contributors": ["a"], "Source": ["quoref"], "URL": ["u"],
            "Categories": ["Question Generation"], "Definition": ["Do the thing."],
            "Input_language": ["English"], "Output_language": ["English"],
            "Instruction_language": ["English"], "Domains": ["Wikipedia"],
            "Positive Examples": [{"input": "i", "output": "o"}],
            "Instances": [{"id": f"i{k}", "input": f"in{k}", "output": [f"out{k}"]}
                          for k in range(50)],
        }
        raw = io.BytesIO(json.dumps(doc).encode("utf-8"))
        meta, insts, complete = superni._read_task_stream(raw, 10)
        self.assertEqual(meta["Source"], ["quoref"])
        self.assertEqual(meta["Definition"], ["Do the thing."])
        self.assertEqual(len(insts), 10)
        self.assertEqual(insts[3]["id"], "i3")
        self.assertFalse(complete)

        raw = io.BytesIO(json.dumps(doc).encode("utf-8"))
        _, insts_all, complete = superni._read_task_stream(raw, 0)
        self.assertEqual(len(insts_all), 50)
        self.assertTrue(complete)

    def test_streaming_reader_survives_multibyte_and_tiny_chunks(self):
        doc = {"Source": ["s"], "Definition": ["café — naïve ✓"],
               "Instances": [{"id": "a", "input": "日本語", "output": ["ok"]}]}
        raw = io.BytesIO(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        stream = superni._StrStream(raw, chunk=3)   # split multibyte chars across reads
        # exercise the same code path with a tiny chunk size
        meta, insts, _ = superni._read_task_stream(
            io.BytesIO(json.dumps(doc, ensure_ascii=False).encode("utf-8")), 0)
        self.assertEqual(meta["Definition"], ["café — naïve ✓"])
        self.assertEqual(insts[0]["input"], "日本語")
        while stream.fill():
            pass
        self.assertIn("日本語", stream.buf)

    def test_round_robin_spreads_across_tasks(self):
        tasks = [
            superni.TaskInfo(name=f"task{i}", definition="d", source=["s"],
                             categories=["c"], domains=["dm"], n_instances_seen=100,
                             mean_gold_words=50.0,
                             instances=[{"id": f"{i}-{k}", "input": "x",
                                         "output": ["y " * 40]} for k in range(100)])
            for i in range(5)
        ]
        pool = superni.round_robin_sample(tasks, 25, seed=13)
        self.assertEqual(len(pool), 25)
        per_task = {}
        for p in pool:
            per_task[p["superni_task_id"]] = per_task.get(p["superni_task_id"], 0) + 1
        self.assertEqual(len(per_task), 5)
        self.assertLessEqual(max(per_task.values()) - min(per_task.values()), 1)

    def test_scan_tasks_applies_every_filter(self):
        class FakeSource:
            instances_per_task = 10

            def describe(self):
                return {}

            def split_task_names(self, split):
                return ["task_keep", "task_gsm", "task_es", "task_short"]

            def read_task(self, name):
                en = {"Input_language": ["English"], "Output_language": ["English"],
                      "Instruction_language": ["English"]}
                long_out = [{"id": "1", "input": "x", "output": ["w " * 40]}]
                if name == "task_keep":
                    return {**en, "Source": ["quoref"], "Categories": ["QG"],
                            "Definition": ["d"]}, long_out, True
                if name == "task_gsm":
                    return {**en, "Source": ["gsm8k"], "Definition": ["d"]}, long_out, True
                if name == "task_es":
                    return {**en, "Output_language": ["Spanish"], "Source": ["x"],
                            "Definition": ["d"]}, long_out, True
                return ({**en, "Source": ["y"], "Definition": ["d"]},
                        [{"id": "1", "input": "x", "output": ["tiny"]}], True)

            def __getattr__(self, k):
                raise AttributeError(k)

        tasks, stats = superni.scan_tasks(FakeSource(), log=lambda *a: None)
        self.assertEqual([t.name for t in tasks], ["task_keep"])
        self.assertEqual(stats["dropped_contaminated_source"], 1)
        self.assertEqual(stats["dropped_non_english"], 1)
        self.assertEqual(stats["dropped_short_gold"], 1)
        self.assertEqual(stats["category_histogram"], {"QG": 1})


# ---------------------------------------------------------------------------
class TestManifest(unittest.TestCase):
    def test_sections_merge_without_clobbering_each_other(self):
        from impl4 import manifest
        arm = config.resolve_arm("A3")
        with tempfile.TemporaryDirectory() as d:
            manifest.init(d, arm)
            manifest.merge(d, "general_slot", {"n_written": 7496})
            manifest.merge(d, "mix", {"n_train": 29984})
            data = manifest.load(d)
            self.assertEqual(data["arm"], "A3")
            self.assertEqual(data["aliases"], ["T1"])
            self.assertEqual(data["delta"], 0.0)
            self.assertEqual(data["sampling"]["T_train"], 1.0)
            self.assertEqual(data["sampling"]["rho_train"], "none")
            self.assertEqual(data["general_slot"]["n_written"], 7496)
            self.assertEqual(data["mix"]["n_train"], 29984)
            # re-init must not drop already-written sections
            manifest.init(d, arm)
            self.assertEqual(manifest.load(d)["mix"]["n_train"], 29984)

    def test_truncated_arms_record_their_rho(self):
        from impl4 import manifest
        data = manifest.base_manifest(config.resolve_arm("T4"))
        self.assertEqual(data["sampling"]["T_train"], 1.6)
        self.assertEqual(data["sampling"]["rho_train"], "k=20,p=0.8")
        self.assertEqual(data["block"], "T")


class TestTextUtil(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(textutil.normalize("Hello,  WORLD!!"), "hello world")
        self.assertEqual(textutil.normalize(""), "")

    def test_word_count_is_plain_whitespace(self):
        self.assertEqual(textutil.word_count("one two  three\nfour"), 4)


# ---------------------------------------------------------------------------
class TestDecontaminationAgainstRealEvalFiles(unittest.TestCase):
    """The real eval prompt files ship in the repo, so index them for real."""

    def test_index_builds_and_flags_a_verbatim_eval_prompt(self):
        from impl4.paths import GENERAL_EVAL_PROMPTS, MATH_EVAL_PROMPTS
        if not (MATH_EVAL_PROMPTS.exists() and GENERAL_EVAL_PROMPTS.exists()):
            self.skipTest("eval prompt files not present")
        idx = ngram.build_eval_index([MATH_EVAL_PROMPTS, GENERAL_EVAL_PROMPTS])
        self.assertGreater(idx.n_refs, 100)
        first = ngram.load_eval_prompts(MATH_EVAL_PROMPTS)[0]
        self.assertIsNotNone(idx.hit(f"Here is a question. {first} Answer it."))
        self.assertIsNone(idx.hit("An unrelated sentence about turbine maintenance "
                                  "schedules in coastal wind farms during winter."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
