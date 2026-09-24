import unittest

from Uncertainpy.src.uncertainpy.gradual import Argument, BAG
from Uncertainpy.src.uncertainpy.gradual.algorithms.Acyclic import computeStrengthValues
from Uncertainpy.src.uncertainpy.gradual.semantics.modular.LinearInfluence import LinearInfluence
from Uncertainpy.src.uncertainpy.gradual.semantics.modular.ProductAggregation import ProductAggregation

from ew_dfquad import compute_ew_dfquad, existing_edges


def make_graph():
    bag = BAG()
    root = Argument("db0", "root", 0.5)
    support = Argument("s1", "support", 0.7)
    attack = Argument("a1", "attack", 0.4)
    deeper_support = Argument("s2", "supports support", 0.8)
    deeper_attack = Argument("a2", "attacks support", 0.3)
    bag.add_support(support, root)
    bag.add_attack(attack, root)
    bag.add_support(deeper_support, support)
    bag.add_attack(deeper_attack, support)
    return bag


class EdgeWeightedDFQuADTests(unittest.TestCase):
    def test_unit_weights_exactly_match_standard_dfquad(self):
        standard = make_graph()
        weighted = BAG.from_dict(standard.to_dict())
        computeStrengthValues(
            standard, ProductAggregation(), LinearInfluence(conservativeness=1)
        )
        compute_ew_dfquad(weighted, {edge: 1.0 for edge in existing_edges(weighted)})
        self.assertEqual(
            {name: arg.strength for name, arg in standard.arguments.items()},
            {name: arg.strength for name, arg in weighted.arguments.items()},
        )

    def test_zero_edges_remove_influence_without_retyping(self):
        bag = make_graph()
        edge_types_before = list(existing_edges(bag))
        weights = {edge: 1.0 for edge in edge_types_before}
        weights[("support", "s1", "db0")] = 0.0
        weights[("attack", "a1", "db0")] = 0.0
        compute_ew_dfquad(bag, weights)
        self.assertEqual(bag.arguments["db0"].strength, 0.5)
        self.assertEqual(list(existing_edges(bag)), edge_types_before)

    def test_missing_weight_is_rejected(self):
        bag = make_graph()
        with self.assertRaises(ValueError):
            compute_ew_dfquad(bag, {})


if __name__ == "__main__":
    unittest.main()
