"""Check paired model comparisons on the same claims and fixed Jev prior."""

import unittest

from compare_generator_models import aligned_pairs, compare_subset


def sample(index, label, direct_jev, original_qbaf):
    predictions = {
        "direct_llm": 0.5,
        "direct_jev": direct_jev,
        "original_qbaf": original_qbaf,
        "jev_qbaf": 0.5,
        "jev_ew_qbaf": 0.5,
        "jev_prior_qbaf": 0.5,
        "jev_prior_ew_qbaf": 0.5,
    }
    return {
        "dataset": "TruthfulClaim", "depth": 1, "index": index,
        "claim": "claim " + str(index), "valid": label,
        "predictions": predictions,
    }


class GeneratorComparisonTests(unittest.TestCase):
    def test_pairs_count_corrections_and_spoils(self):
        old = {
            ("TruthfulClaim", 1, 0): sample(0, 1, 0.8, 0.4),
            ("TruthfulClaim", 1, 1): sample(1, 0, 0.2, 0.3),
        }
        new = {
            ("TruthfulClaim", 1, 0): sample(0, 1, 0.8, 0.7),
            ("TruthfulClaim", 1, 1): sample(1, 0, 0.2, 0.7),
        }
        result = compare_subset(aligned_pairs(old, new), "TruthfulClaim", 1,
                                "original_qbaf")
        self.assertEqual(result["corrected"], 1)
        self.assertEqual(result["spoiled"], 1)
        self.assertEqual(result["prediction_flips"], 2)
        self.assertEqual(result["accuracy_change"], 0)
        self.assertAlmostEqual(result["brier_change"], 0.065)

    def test_rejects_changed_direct_jev_prior(self):
        key = ("TruthfulClaim", 1, 0)
        old = {key: sample(0, 1, 0.8, 0.4)}
        new = {key: sample(0, 1, 0.7, 0.4)}
        with self.assertRaisesRegex(ValueError, "Direct Jev baseline changed"):
            aligned_pairs(old, new)


if __name__ == "__main__":
    unittest.main()
