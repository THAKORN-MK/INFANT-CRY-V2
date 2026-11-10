"""Metric contract tests for held-out multi-class evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from infantcry_research.evaluation.metrics import (
    classification_metrics,
    group_bootstrap_intervals,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_reports_macro_and_per_class_metrics_in_configured_order(self) -> None:
        result = classification_metrics(
            ["a", "a", "b", "c"],
            ["a", "b", "b", "c"],
            np.asarray(
                [
                    [0.8, 0.1, 0.1],
                    [0.4, 0.5, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                ]
            ),
            ("a", "b", "c"),
        )

        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 2 / 3 + 1) / 3)
        self.assertEqual(list(result["per_class"]), ["a", "b", "c"])
        self.assertEqual(result["per_class"]["a"]["support"], 2)
        self.assertIsNotNone(result["macro_ovr_auc"])

    def test_absent_class_is_reported_with_zero_support_and_undefined_auc(self) -> None:
        result = classification_metrics(
            ["a", "b"],
            ["a", "a"],
            np.asarray([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]]),
            ("a", "b", "c"),
        )

        self.assertEqual(result["per_class"]["c"]["support"], 0)
        self.assertIsNone(result["per_class"]["c"]["roc_auc_ovr"])
        self.assertIsNone(result["macro_ovr_auc"])

    def test_rejects_scores_that_are_not_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            classification_metrics(
                ["a", "b"], ["a", "b"], np.asarray([[0.8, 0.8], [0.2, 0.2]]), ("a", "b")
            )

    def test_group_bootstrap_is_deterministic_and_returns_95_percent_intervals(self) -> None:
        first = group_bootstrap_intervals(
            ["a", "b", "a", "b", "a", "b"],
            ["a", "a", "a", "b", "b", "b"],
            ["g1", "g1", "g2", "g2", "g3", "g3"],
            class_order=("a", "b"),
            n_resamples=100,
            seed=9,
            confidence_level=0.95,
        )
        second = group_bootstrap_intervals(
            ["a", "b", "a", "b", "a", "b"],
            ["a", "a", "a", "b", "b", "b"],
            ["g1", "g1", "g2", "g2", "g3", "g3"],
            class_order=("a", "b"),
            n_resamples=100,
            seed=9,
            confidence_level=0.95,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["group_count"], 3)
        self.assertEqual(first["confidence_level"], 0.95)
        self.assertLessEqual(first["macro_f1"]["lower"], first["macro_f1"]["upper"])


if __name__ == "__main__":
    unittest.main()
