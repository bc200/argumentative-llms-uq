import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_jev_qbaf import (
    chinese_report, evaluate_sample, metrics, prior_comparisons, summarize,
)
from experiment_io import CachedLlmManager, read_json


class FakeLlm:
    def __init__(self):
        self.calls = 0
        self.last_usage = None

    def chat_completion(self, message, **_):
        self.calls += 1
        if "single short argument supporting" in message:
            return "A supporting fact %d." % self.calls
        if "single short argument attacking" in message:
            return "An attacking fact %d." % self.calls
        return "Likelihood: 70%"


class FakeJev:
    def ask(self, relative_path, *_):
        return 0.6, {
            "_cache_path": str(relative_path),
            "cost_usd": 0.000001,
            "elapsed_seconds": 0.01,
        }


class BenchmarkPipelineTests(unittest.TestCase):
    def test_depth_graphs_are_cached_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                cache_dir=Path(directory),
                generator_model="fake-local",
                jev_model="fake-jev",
                llm_base_url=None,
                llm_thinking=None,
                llm_currency="USD",
                llm_input_price=None,
                llm_output_price=None,
                breadth=1,
                generation_args={
                    "temperature": 0.7,
                    "max_new_tokens": 128,
                    "top_p": 0.95,
                },
            )
            llm = FakeLlm()
            jev = FakeJev()
            first = evaluate_sample(
                args, "TruthfulClaim", 0, "A root claim.", 1, 1, llm, jev
            )
            calls_after_first = llm.calls
            repeat = evaluate_sample(
                args, "TruthfulClaim", 0, "A root claim.", 1, 1, llm, jev
            )
            self.assertEqual(llm.calls, calls_after_first)
            self.assertEqual(first["predictions"], repeat["predictions"])
            self.assertEqual(first["graph_cache"], repeat["graph_cache"])
            self.assertEqual(first["argument_count"], 3)
            second_depth = evaluate_sample(
                args, "TruthfulClaim", 0, "A root claim.", 1, 2, llm, jev
            )
            self.assertEqual(second_depth["argument_count"], 7)
            self.assertEqual(second_depth["edge_count"], 6)
            first_graph = read_json(first["graph_cache"])["graph"]
            second_graph = read_json(second_depth["graph_cache"])["graph"]
            for name in first_graph["arguments"]:
                self.assertEqual(
                    first_graph["arguments"][name]["argument"],
                    second_graph["arguments"][name]["argument"],
                )
            for method in ("original_qbaf", "jev_qbaf"):
                for name, score in first["argument_base_scores"][method].items():
                    self.assertEqual(
                        score, second_depth["argument_base_scores"][method][name]
                    )
            first_weights = {
                (edge["type"], edge["source"], edge["target"]): edge["weight"]
                for edge in first["edge_weights"]
            }
            second_weights = {
                (edge["type"], edge["source"], edge["target"]): edge["weight"]
                for edge in second_depth["edge_weights"]
            }
            for relation, weight in first_weights.items():
                self.assertEqual(weight, second_weights[relation])
            self.assertNotEqual(first["graph_cache"], second_depth["graph_cache"])
            self.assertEqual(set(first["predictions"]), {
                "direct_llm", "direct_jev", "original_qbaf",
                "jev_qbaf", "jev_ew_qbaf", "jev_prior_qbaf",
                "jev_prior_ew_qbaf",
            })
            self.assertTrue(first["root_conflicting_evidence"])
            self.assertFalse(first["root_counterevidence"])
            self.assertEqual(first["usage"]["jev_prior_qbaf"]["calls"], 5)
            self.assertEqual(first["usage"]["jev_prior_ew_qbaf"]["calls"], 7)
            comparisons = prior_comparisons([first, second_depth])
            self.assertEqual(len(comparisons), 8)
            self.assertTrue(all(item["n"] == 1 for item in comparisons))
            report = chinese_report(args, summarize([first, second_depth]),
                                    {"usd": 0.0, "cny": 0.0},
                                    [first, second_depth], prior_stats=comparisons)
            self.assertIn("实验报告", report)
            self.assertIn("TruthfulClaim", report)
            self.assertIn("Jev 根先验修正分析", report)

    def test_probability_metrics(self):
        result = metrics([1, 0], [0.8, 0.2])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.04)
        self.assertAlmostEqual(result["ece"], 0.2)

    def test_prior_revision_counts_corrections_and_harm(self):
        rows = [
            {
                "dataset": "TruthfulClaim", "depth": 1, "valid": 1,
                "root_conflicting_evidence": True,
                "root_counterevidence": False,
                "predictions": {
                    "direct_jev": 0.4,
                    "jev_prior_qbaf": 0.7,
                    "jev_prior_ew_qbaf": 0.7,
                },
            },
            {
                "dataset": "TruthfulClaim", "depth": 1, "valid": 0,
                "root_conflicting_evidence": False,
                "root_counterevidence": True,
                "predictions": {
                    "direct_jev": 0.1,
                    "jev_prior_qbaf": 0.8,
                    "jev_prior_ew_qbaf": 0.8,
                },
            },
        ]
        comparisons = prior_comparisons(rows)
        full = next(item for item in comparisons
                    if item["subset"] == "全部"
                    and item["method"] == "jev_prior_qbaf")
        self.assertEqual(full["n"], 2)
        self.assertEqual(full["decision_flips"], 2)
        self.assertEqual(full["corrected"], 1)
        self.assertEqual(full["spoiled"], 1)
        self.assertEqual(full["accuracy_change"], 0.0)
        self.assertAlmostEqual(full["brier_change"], 0.18)
        self.assertAlmostEqual(full["mean_absolute_revision"], 0.5)

    def test_cny_llm_cost_is_separate_from_jev_usd(self):
        class ChargedLlm:
            last_usage = {"input_tokens": 100, "output_tokens": 20}

            def chat_completion(self, *_args, **_kwargs):
                return "Likelihood: 70%"

        with tempfile.TemporaryDirectory() as directory:
            recorder = CachedLlmManager(
                ChargedLlm(), "openai/deepseek-v4.1-flash", Path(directory),
                input_price=1.9, output_price=7.6, currency="CNY",
            )
            recorder.chat_completion("prompt")
            row = recorder.records[0]
            self.assertEqual(row["cost_usd"], 0.0)
            self.assertAlmostEqual(row["cost_cny"], (100 * 1.9 + 20 * 7.6) / 1000000)

    def test_cache_only_llm_rejects_a_miss_without_calling_model(self):
        with tempfile.TemporaryDirectory() as directory:
            llm = FakeLlm()
            recorder = CachedLlmManager(
                llm, "fake-local", Path(directory), cache_only=True
            )
            with self.assertRaises(FileNotFoundError):
                recorder.chat_completion("missing prompt")
            self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
