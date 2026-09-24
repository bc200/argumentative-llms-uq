import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_jev_qbaf import chinese_report, evaluate_sample, metrics, summarize
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
                "jev_qbaf", "jev_ew_qbaf",
            })
            report = chinese_report(args, summarize([first, second_depth]),
                                    {"usd": 0.0, "cny": 0.0},
                                    [first, second_depth])
            self.assertIn("实验报告", report)
            self.assertIn("TruthfulClaim", report)

    def test_probability_metrics(self):
        result = metrics([1, 0], [0.8, 0.2])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.04)
        self.assertAlmostEqual(result["ece"], 0.2)

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


if __name__ == "__main__":
    unittest.main()
