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

from evaluation._cli import (
    build_parser,
    load_samples,
    print_metrics,
    to_dict,
    write_output,
)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Test-set format (see evaluation/_cli.py for the accepted container shapes).
# Each sample:
#
#     {
#       "text": "Operating profit rose to EUR 13.1 mn.",  # required for --run-agent
#       "sentiment": "positive",                          # ground truth
#       "prediction": "positive"                          # optional
#     }
#
# "label" is accepted as an alias for "sentiment"; "predicted_sentiment" and
# "predicted" as aliases for "prediction".

# Label spellings seen in the public datasets and model outputs, mapped onto the
# three classes the evaluator scores. The confusion matrix is keyed by these
# exact strings, so anything unmapped has to be rejected rather than guessed.
LABEL_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "bullish": "positive",
    "2": "positive",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
    "negative": "negative",
    "neg": "negative",
    "bearish": "negative",
    "0": "negative",
}


def _normalize_label(label: Any, *, context: str) -> str:
    """
    Map a raw sentiment label onto one of negative/neutral/positive.

    Args:
        label: Raw label from a test set or model output
        context: Description used in the error message

    Returns:
        Canonical class name

    Raises:
        SystemExit: Label cannot be mapped onto a known class
    """
    normalized = LABEL_ALIASES.get(str(label).strip().lower())

    if normalized is None:
        raise SystemExit(
            f"Unrecognised sentiment label {label!r} in {context}. "
            f"Expected one of: {sorted(set(LABEL_ALIASES.values()))}"
        )

    return normalized


def _predict_with_agent(samples: list[dict[str, Any]]) -> None:
    """
    Populate each sample's "prediction" by running SentimentAgent over its text.

    Classification is per-sample text (the Financial PhraseBank is sentence
    level), not the document-level aggregation the agent performs in the graph.

    Args:
        samples: Test-set samples, mutated in place
    """
    from src.agents.sentiment_agent import SentimentAgent

    agent = SentimentAgent()

    for index, sample in enumerate(samples):
        text = sample.get("text", "")

        if not text:
            logger.warning(f"Sample {index} has no 'text'; predicting neutral")
            sample["prediction"] = "neutral"
            continue

        result = agent.sentiment_pipeline(text[:512])
        sample["prediction"] = result[0]["label"] if result else "neutral"

    logger.info(f"Generated predictions for {len(samples)} samples")


def main() -> None:
    """CLI entry point for sentiment evaluation."""
    parser = build_parser(
        description=(
            "Evaluate the sentiment agent against a Financial PhraseBank style test set"
        ),
        agent_help="Load SentimentAgent and classify each sample's 'text'",
    )
    args = parser.parse_args()

    samples = load_samples(args)

    if args.run_agent:
        _predict_with_agent(samples)

    predictions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    without_predictions = 0

    for index, sample in enumerate(samples):
        reference = sample.get("sentiment", sample.get("label"))

        if reference is None:
            raise SystemExit(
                f"Sample {index} has no 'sentiment' (or 'label') key. "
                "Every sample needs a ground-truth class."
            )

        prediction = sample.get(
            "prediction", sample.get("predicted_sentiment", sample.get("predicted"))
        )

        if prediction is None:
            without_predictions += 1
            prediction = "neutral"

        references.append(
            {"sentiment": _normalize_label(reference, context=f"sample {index} ground truth")}
        )
        predictions.append(
            {"sentiment": _normalize_label(prediction, context=f"sample {index} prediction")}
        )

    if without_predictions:
        logger.warning(
            f"{without_predictions}/{len(samples)} samples carried no 'prediction' "
            "and defaulted to neutral. Pass --run-agent to generate them."
        )

    evaluator = SentimentEvaluator()
    metrics = evaluator.evaluate(predictions, references)

    payload = to_dict(metrics)
    payload["samples_evaluated"] = len(samples)

    if args.benchmark:
        payload["phrasebank_benchmark"] = evaluator.benchmark_against_financial_phrasebank()

    print_metrics("SENTIMENT EVALUATION (Financial PhraseBank)", payload)
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
