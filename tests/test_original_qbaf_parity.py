"""Check that the cached-graph baseline matches the upstream DF-QuAD path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import Uncertainpy.src.uncertainpy.gradual as grad

from argument_miner import ArgumentMiner
from benchmark_jev_qbaf import (
    load_or_generate_graph,
    original_argument_scores,
    qbaf_prediction,
)
from prompt import ArgumentMiningPrompts, UncertaintyEvaluatorPrompts
from uncertainty_estimator import UncertaintyEstimator


class PromptDeterministicLlm:
    """Return stable answers so the two execution orders can be compared."""

    last_usage = None

    def __init__(self, root_attack_na=False):
        self.root_attack_na = root_attack_na

    def chat_completion(self, message, **_):
        if message.startswith("Please provide a single short argument"):
            claim = message.split("Claim: ", 1)[1].split("\n", 1)[0]
            support = "argument supporting" in message
            if self.root_attack_na and not support and claim == "Root claim.":
                return "N/A"
            kind = "Support" if support else "Attack"
            return f"{kind} for ({claim})"
        if "For the argument:" in message:
            if 'Argument: "Support' in message:
                return "Likelihood: 83%"
            return "Likelihood: 27%"
        return "Likelihood: 60%"


class OriginalQbafParityTests(unittest.TestCase):
    def test_matches_upstream_base_score_path_at_both_depths(self):
        claim = "Root claim."
        generation_args = {
            "temperature": 0.7,
            "max_new_tokens": 128,
            "top_p": 0.95,
        }
        for depth in (1, 2):
            for root_attack_na in (False, True):
                with self.subTest(depth=depth, root_attack_na=root_attack_na):
                    upstream_llm = PromptDeterministicLlm(root_attack_na)
                    estimator = UncertaintyEstimator(
                        upstream_llm,
                        UncertaintyEvaluatorPrompts.analyst,
                        verbal=False,
                        generation_args=generation_args,
                    )
                    miner = ArgumentMiner(
                        ArgumentMiningPrompts.new_sup_att,
                        UncertaintyEvaluatorPrompts.analyst,
                        upstream_llm,
                        depth=depth,
                        breadth=1,
                        generation_args=generation_args,
                    )
                    upstream, _ = miner.generate_arguments(claim, estimator)
                    expected_graph = upstream.to_dict()
                    expected_scores = {
                        name: argument.initial_weight
                        for name, argument in upstream.arguments.items()
                        if name != "db0"
                    }
                    grad.algorithms.computeStrengthValues(
                        upstream,
                        grad.semantics.modular.ProductAggregation(),
                        grad.semantics.modular.LinearInfluence(conservativeness=1),
                    )

                    with tempfile.TemporaryDirectory() as directory:
                        args = SimpleNamespace(
                            cache_dir=Path(directory),
                            generator_model="fake-local",
                            llm_base_url=None,
                            llm_thinking=None,
                            llm_currency="USD",
                            llm_input_price=None,
                            llm_output_price=None,
                            breadth=1,
                            generation_args=generation_args,
                        )
                        benchmark_llm = PromptDeterministicLlm(root_attack_na)
                        graph, _, _ = load_or_generate_graph(
                            args, "TruthfulClaim", 0, claim, depth, benchmark_llm
                        )
                        prior_graph = None
                        if depth == 2:
                            prior_graph, _, _ = load_or_generate_graph(
                                args, "TruthfulClaim", 0, claim, 1, benchmark_llm
                            )
                        scores, _ = original_argument_scores(
                            args, "TruthfulClaim", 0, depth, graph,
                            benchmark_llm, prior_graph,
                        )

                        self.assertEqual(graph["supports"], expected_graph["supports"])
                        self.assertEqual(graph["attacks"], expected_graph["attacks"])
                        self.assertEqual(
                            {name: arg["argument"]
                             for name, arg in graph["arguments"].items()},
                            {name: arg["argument"]
                             for name, arg in expected_graph["arguments"].items()},
                        )
                        self.assertEqual(scores, expected_scores)
                        self.assertEqual(
                            qbaf_prediction(graph, scores),
                            upstream.arguments["db0"].strength,
                        )


if __name__ == "__main__":
    unittest.main()
