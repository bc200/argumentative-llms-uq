import unittest

from analyze_jev_confidence_bins import (
    POOLED, analyze, bucket_summary, confidence_bin, render_report,
)


def row(label, direct, prior_ew, depth=1):
    return {
        "dataset": "TruthfulClaim",
        "depth": depth,
        "valid": label,
        "predictions": {
            "direct_jev": direct,
            "jev_prior_ew_qbaf": prior_ew,
        },
    }


class JevConfidenceBinTests(unittest.TestCase):
    def test_confidence_uses_both_sides_and_right_open_boundaries(self):
        cases = (
            (0.5, 0), (0.59, 0), (0.6, 1), (0.4, 1),
            (0.7, 2), (0.3, 2), (0.8, 3), (0.2, 3),
            (0.9, 4), (0.1, 4), (1.0, 4), (0.0, 4),
        )
        for probability, expected in cases:
            with self.subTest(probability=probability):
                self.assertEqual(confidence_bin(probability), expected)

    def test_rescue_harm_and_accuracy_difference_agree(self):
        rows = [
            row(1, 0.4, 0.8),
            row(0, 0.1, 0.9),
            row(1, 0.9, 0.8),
        ]
        result = bucket_summary(rows)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["rescue"], 1)
        self.assertEqual(result["harm"], 1)
        self.assertEqual(result["rescue_minus_harm"], 0)
        self.assertEqual(result["prediction_flips"], 2)
        self.assertAlmostEqual(result["flip_rate"], 2 / 3)
        self.assertAlmostEqual(result["direct_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["prior_ew_accuracy"], 2 / 3)
        self.assertEqual(result["accuracy_change"], 0.0)
        self.assertAlmostEqual(result["direct_brier"], 0.38 / 3)
        self.assertAlmostEqual(result["prior_ew_brier"], 0.89 / 3)

    def test_each_depth_is_analyzed_separately(self):
        rows = [row(1, 0.55, 0.7, depth=1),
                row(1, 0.55, 0.3, depth=2)]
        result = analyze(rows)
        first = next(item for item in result["buckets"]
                     if item["dataset"] == POOLED
                     and item["depth"] == 1 and item["bin_index"] == 0)
        second = next(item for item in result["buckets"]
                      if item["dataset"] == POOLED
                      and item["depth"] == 2 and item["bin_index"] == 0)
        self.assertEqual(first["n"], 1)
        self.assertEqual(second["n"], 1)
        self.assertEqual(first["rescue_minus_harm"], 0)
        self.assertEqual(second["rescue_minus_harm"], -1)
        report = render_report(result, "test/data")
        self.assertIn("Rescue−Harm", report)
        self.assertIn("D=2", report)


if __name__ == "__main__":
    unittest.main()
