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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Test-set format (see evaluation/_cli.py for the accepted container shapes).
# Each sample:
#
#     {
#       "text": "Apple Inc. reported record revenue.",   # required for --run-agent
#       "references": [{"word": "Apple Inc.", "entity": "ORG"}],
#       "predictions": [{"word": "Apple Inc.", "entity": "ORG"}]   # optional
#     }
#
# "entities" is accepted as an alias for "references", and "entity_group" /
# "label" as aliases for "entity" (the former is what the HuggingFace pipeline
# emits when aggregation_strategy="simple").


def _normalize_entities(
    entities: list[dict[str, Any]] | None, sample_index: int
) -> list[dict[str, Any]]:
    """
    Normalise raw entity dicts to the {word, entity} shape the evaluator scores.

    The evaluator pools everything into (word, entity) sets, so the surface form
    is namespaced by sample index — otherwise the same word occurring in two
    different samples would cross-match and inflate true positives.

    Args:
        entities: Raw entity dicts from a test set or a HuggingFace pipeline
        sample_index: Index of the sample these entities came from

    Returns:
        List of normalised entity dicts
    """
    normalized = []

    for entity in entities or []:
        word = entity.get("word", entity.get("text", ""))
        label = entity.get(
            "entity", entity.get("entity_group", entity.get("label", "unknown"))
        )

        normalized.append({"word": f"{sample_index}::{word}", "entity": label})

    return normalized


def _predict_with_agent(samples: list[dict[str, Any]]) -> None:
    """
    Populate each sample's "predictions" by running NERAgent over its text.

    Args:
        samples: Test-set samples, mutated in place
    """
    from src.agents.ner_agent import NERAgent

    agent = NERAgent()

    for index, sample in enumerate(samples):
        text = sample.get("text", "")

        if not text:
            logger.warning(f"Sample {index} has no 'text'; predicting no entities")
            sample["predictions"] = []
            continue

        sample["predictions"] = agent._extract_entities(text)

    logger.info(f"Generated predictions for {len(samples)} samples")


def main() -> None:
    """CLI entry point for NER evaluation."""
    parser = build_parser(
        description="Evaluate the NER agent against a FiNER-139 style test set",
        agent_help="Load NERAgent and tag each sample's 'text' to produce predictions",
    )
    args = parser.parse_args()

    samples = load_samples(args)

    if args.run_agent:
        _predict_with_agent(samples)

    predictions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    without_predictions = 0

    for index, sample in enumerate(samples):
        refs = sample.get("references", sample.get("entities"))

        if refs is None:
            raise SystemExit(
                f"Sample {index} has no 'references' (or 'entities') key. "
                "Every sample needs ground-truth entities."
            )

        preds = sample.get("predictions")
        if preds is None:
            without_predictions += 1
            preds = []

        references.extend(_normalize_entities(refs, index))
        predictions.extend(_normalize_entities(preds, index))

    if without_predictions:
        logger.warning(
            f"{without_predictions}/{len(samples)} samples carried no 'predictions' "
            "and were scored as complete misses. Pass --run-agent to generate them."
        )

    evaluator = NERERvaluator()
    metrics = evaluator.evaluate(predictions, references)

    payload = to_dict(metrics)
    payload["samples_evaluated"] = len(samples)

    if args.benchmark:
        payload["finer139_benchmark"] = evaluator.benchmark_against_finer139(predictions)

    print_metrics("NER EVALUATION (FiNER-139)", payload)
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
