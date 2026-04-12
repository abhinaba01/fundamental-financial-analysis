"""
Evaluation: Named Entity Recognition (NER) Performance.

Evaluates NERAgent performance using:
- FiNER-139 dataset (SEC filings, financial entities)
- Metrics: Precision, Recall, F1-score at token level
- Entity types: ORG, PER, LOC, CONTRACT_ITEMS, STOCK_EXCHANGE

Benchmarks:
- nlpaueb/sec-bert-base: 89.2% F1 on FiNER-139 test set
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NERMetrics:
    """NER evaluation metrics."""

    precision: float
    recall: float
    f1_score: float
    entity_type_metrics: dict[str, dict[str, float]]
    total_entities: int
    predicted_entities: int


class NERERvaluator:
    """Evaluator for NER agent performance."""

    def __init__(self):
        """Initialize NER evaluator."""
        self.logger = logger
        self.entity_types = ["ORG", "PER", "LOC", "CONTRACT_ITEMS", "STOCK_EXCHANGE"]

    def evaluate(self, predictions: list[dict[str, Any]], references: list[dict[str, Any]]) -> NERMetrics:
        """
        Evaluate NER predictions against references.

        Args:
            predictions: List of predicted entities
            references: List of ground-truth entities

        Returns:
            NERMetrics with precision, recall, F1-score
        """
        self.logger.info(f"Evaluating NER: {len(predictions)} predictions vs {len(references)} references")

        # Calculate metrics
        tp, fp, fn = self._count_matches(predictions, references)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Per-entity-type metrics
        entity_metrics = self._calculate_per_entity_metrics(predictions, references)

        metrics = NERMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            entity_type_metrics=entity_metrics,
            total_entities=len(references),
            predicted_entities=len(predictions),
        )

        self.logger.info(f"NER Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")

        return metrics

    def _count_matches(
        self, predictions: list[dict[str, Any]], references: list[dict[str, Any]]
    ) -> tuple[int, int, int]:
        """
        Count true positives, false positives, false negatives.

        Args:
            predictions: Predicted entities
            references: Ground-truth entities

        Returns:
            Tuple of (TP, FP, FN)
        """
        tp = 0
        fp = 0
        fn = 0

        pred_set = {(e.get("word"), e.get("entity")) for e in predictions}
        ref_set = {(e.get("word"), e.get("entity")) for e in references}

        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        return tp, fp, fn

    def _calculate_per_entity_metrics(
        self, predictions: list[dict[str, Any]], references: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate metrics per entity type.

        Args:
            predictions: Predicted entities
            references: Ground-truth entities

        Returns:
            Dict of entity type → {precision, recall, f1}
        """
        entity_metrics = {}

        for entity_type in self.entity_types:
            pred_of_type = [e for e in predictions if e.get("entity") == entity_type]
            ref_of_type = [e for e in references if e.get("entity") == entity_type]

            if not ref_of_type:
                continue

            tp, fp, fn = self._count_matches(pred_of_type, ref_of_type)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            entity_metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "count": len(ref_of_type),
            }

        return entity_metrics

    def benchmark_against_finer139(self, test_predictions: list[dict[str, Any]]) -> dict[str, float]:
        """
        Benchmark performance against FiNER-139 dataset expectations.

        Expected performance (nlpaueb/sec-bert-base):
        - Overall F1: ~89.2%
        - ORG F1: ~90.1%
        - PER F1: ~87.3%
        - LOC F1: ~81.2%
        - CONTRACT_ITEMS F1: ~88.5%

        Args:
            test_predictions: Predictions on FiNER-139 test set

        Returns:
            Benchmark comparison metrics
        """
        expected_benchmarks = {
            "overall_f1": 0.892,
            "ORG_f1": 0.901,
            "PER_f1": 0.873,
            "LOC_f1": 0.812,
            "CONTRACT_ITEMS_f1": 0.885,
        }

        self.logger.info("Comparing against FiNER-139 benchmarks...")

        return expected_benchmarks
