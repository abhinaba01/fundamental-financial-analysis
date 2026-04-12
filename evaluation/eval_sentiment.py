"""
Evaluation: Sentiment Analysis Performance.

Evaluates SentimentAgent performance using:
- Financial PhraseBank dataset (4,840 financial news sentences)
- Metrics: Accuracy, per-class precision/recall/F1
- Classes: Positive, Negative, Neutral

Models:
- Primary: ProsusAI/finbert (3-class)
- Secondary: yiyanghkust/finbert-tone (tone analysis)

Benchmarks:
- ProsusAI/finbert: ~97% accuracy on Financial PhraseBank
- FinBERT-tone: High precision on tone transitions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import Counter

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentMetrics:
    """Sentiment evaluation metrics."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    total_samples: int


class SentimentEvaluator:
    """Evaluator for sentiment analysis agent performance."""

    def __init__(self):
        """Initialize sentiment evaluator."""
        self.logger = logger
        self.sentiment_classes = ["negative", "neutral", "positive"]

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> SentimentMetrics:
        """
        Evaluate sentiment predictions against references.

        Args:
            predictions: List of predicted sentiments
            references: List of ground-truth sentiments

        Returns:
            SentimentMetrics with accuracy, F1, per-class metrics
        """
        self.logger.info(
            f"Evaluating sentiment: {len(predictions)} predictions vs {len(references)} references"
        )

        # Extract labels
        pred_labels = [p.get("sentiment", "neutral") for p in predictions]
        ref_labels = [r.get("sentiment", "neutral") for r in references]

        # Calculate accuracy
        correct = sum(1 for p, r in zip(pred_labels, ref_labels) if p == r)
        accuracy = correct / len(ref_labels) if ref_labels else 0.0

        # Calculate confusion matrix
        confusion_matrix = self._build_confusion_matrix(pred_labels, ref_labels)

        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(pred_labels, ref_labels)

        # Macro and weighted F1
        macro_f1 = self._calculate_macro_f1(per_class_metrics)
        weighted_f1 = self._calculate_weighted_f1(per_class_metrics, ref_labels)

        metrics = SentimentMetrics(
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
            total_samples=len(ref_labels),
        )

        self.logger.info(
            f"Sentiment Metrics - Accuracy: {accuracy:.3f}, "
            f"Macro-F1: {macro_f1:.3f}, Weighted-F1: {weighted_f1:.3f}"
        )

        return metrics

    def _build_confusion_matrix(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, dict[str, int]]:
        """
        Build confusion matrix.

        Args:
            predictions: Predicted sentiment labels
            references: Ground-truth sentiment labels

        Returns:
            Confusion matrix as dict of dicts
        """
        matrix = {true_label: {pred_label: 0 for pred_label in self.sentiment_classes}
                  for true_label in self.sentiment_classes}

        for pred, true in zip(predictions, references):
            matrix[true][pred] += 1

        return matrix

    def _calculate_per_class_metrics(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate per-class precision, recall, F1.

        Args:
            predictions: Predicted labels
            references: Ground-truth labels

        Returns:
            Dict of label → {precision, recall, f1, count}
        """
        metrics = {}

        for label in self.sentiment_classes:
            tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
            fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
            fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            count = sum(1 for r in references if r == label)

            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "count": count,
            }

        return metrics

    def _calculate_macro_f1(self, per_class_metrics: dict[str, dict[str, float]]) -> float:
        """Calculate macro-averaged F1 score."""
        f1_scores = [m.get("f1_score", 0.0) for m in per_class_metrics.values()]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def _calculate_weighted_f1(
        self, per_class_metrics: dict[str, dict[str, float]], references: list[str]
    ) -> float:
        """Calculate weighted-averaged F1 score."""
        total = len(references)
        if total == 0:
            return 0.0

        weighted_f1 = 0.0
        for label, metrics in per_class_metrics.items():
            weight = metrics.get("count", 0) / total
            f1 = metrics.get("f1_score", 0.0)
            weighted_f1 += weight * f1

        return weighted_f1

    def benchmark_against_financial_phrasebank(self) -> dict[str, float]:
        """
        Benchmark against Financial PhraseBank expectations.

        Expected performance (ProsusAI/finbert):
        - Accuracy: ~97%
        - Positive F1: ~0.96
        - Negative F1: ~0.95
        - Neutral F1: ~0.91

        Returns:
            Benchmark metrics
        """
        benchmarks = {
            "accuracy": 0.97,
            "positive_f1": 0.96,
            "negative_f1": 0.95,
            "neutral_f1": 0.91,
        }

        self.logger.info("Using Financial PhraseBank benchmarks...")
        return benchmarks
