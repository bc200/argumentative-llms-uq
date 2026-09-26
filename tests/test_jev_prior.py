import unittest

from Uncertainpy.src.uncertainpy.gradual import Argument, BAG

from benchmark_jev_qbaf import has_conflicting_root_evidence, qbaf_prediction


class JevPriorTests(unittest.TestCase):
    def test_prior_and_weighted_relations_follow_requested_formulas(self):
        graph = BAG()
        root = Argument("db0", "root", 0.5)
        support = Argument("s1", "supporting evidence", 0.5)
        attack = Argument("a1", "attacking evidence", 0.5)
        graph.add_support(support, root)
        graph.add_attack(attack, root)
        cached_graph = graph.to_dict()
        scores = {"s1": 0.2, "a1": 0.9}
        weights = {
            ("support", "s1", "db0"): 0.5,
            ("attack", "a1", "db0"): 0.5,
        }

        self.assertTrue(has_conflicting_root_evidence(cached_graph, scores))
        self.assertAlmostEqual(
            qbaf_prediction(cached_graph, scores, root_score=0.8), 0.24
        )
        self.assertAlmostEqual(
            qbaf_prediction(cached_graph, scores, weights, root_score=0.8),
            0.52,
        )
        self.assertEqual(
            qbaf_prediction(
                cached_graph, scores,
                {relation: 1.0 for relation in weights}, root_score=0.8,
            ),
            qbaf_prediction(cached_graph, scores, root_score=0.8),
        )

    def test_no_active_opposing_evidence_is_not_a_conflict(self):
        graph = BAG()
        root = Argument("db0", "root", 0.5)
        support = Argument("s1", "support", 0.5)
        attack = Argument("a1", "N/A", 0.0)
        graph.add_support(support, root)
        graph.add_attack(attack, root)
        self.assertFalse(has_conflicting_root_evidence(
            graph.to_dict(), {"s1": 0.7, "a1": 0.0}
        ))


if __name__ == "__main__":
    unittest.main()
