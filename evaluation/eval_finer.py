"""
Evaluation: Financial Entity (FiNER-139) Tagging Performance.

Evaluates FinancialNERAgent's fine-tuned model - tags numeric tokens in SEC
filing sentences with XBRL accounting concepts (e.g. is this figure
"Revenues" or "OperatingLeaseLiability"). This is a different task from
evaluation/eval_ner.py's general ORG/PER/LOC/MISC entity evaluation.

Scoring uses seqeval (lenient/default mode, not strict IOB2), the standard
entity-level metric for this task, since a partially-trained model easily
emits invalid B-/I- transitions that strict mode would raise on instead of
just scoring as wrong.

Benchmark: the real, reported number comes from training on the full
FiNER-139 dataset (see Train_FinBERT_NER_Colab.ipynb) and evaluating on its
full 108,378-sentence test split - that number is a one-time measurement,
not something this CLI reproduces. This module's job is a *local*
smoke-test: score the live model (or a supplied test set) against a small,
deterministic sample of the real test split
(data/eval/finer139_test_sample.json), not a claim about the full-dataset
number. See README.md's "Measured Results (this repo)" vs.
"Performance Benchmarks" for that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
except ImportError:
    classification_report = f1_score = precision_score = recall_score = None

from evaluation._cli import (
    build_parser,
    load_samples,
    print_metrics,
    write_output,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FinerMetrics:
    """Entity-level FiNER-139 tagging metrics."""

    precision: float
    recall: float
    f1_score: float
    entity_type_metrics: dict[str, dict[str, float]]
    total_sentences: int


class FinerEvaluator:
    """Evaluator for FinancialNERAgent's XBRL tagging performance."""

    def __init__(self):
        """Initialize the FiNER-139 evaluator."""
        self.logger = logger

        if classification_report is None:
            raise ImportError(
                "seqeval not installed. Install with: pip install -r requirements-train.txt"
            )

    def evaluate(
        self, predicted_tags: list[list[str]], reference_tags: list[list[str]]
    ) -> FinerMetrics:
        """
        Score predicted IOB2 tag sequences against reference sequences.

        Args:
            predicted_tags: Per-sentence predicted tag sequences
            reference_tags: Per-sentence ground-truth tag sequences

        Returns:
            FinerMetrics with entity-level precision/recall/F1
        """
        self.logger.info(f"Evaluating FiNER-139 tagging: {len(predicted_tags)} sentences")

        precision = precision_score(reference_tags, predicted_tags)
        recall = recall_score(reference_tags, predicted_tags)
        f1 = f1_score(reference_tags, predicted_tags)

        report = classification_report(reference_tags, predicted_tags, output_dict=True, zero_division=0)
        # classification_report includes aggregate rows (micro/macro/weighted avg) -
        # keep only per-entity-type rows for entity_type_metrics.
        entity_type_metrics = {
            tag: {
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1_score": scores["f1-score"],
                "count": scores["support"],
            }
            for tag, scores in report.items()
            if not tag.endswith("avg")
        }

        metrics = FinerMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            entity_type_metrics=entity_type_metrics,
            total_sentences=len(reference_tags),
        )

        self.logger.info(
            f"FiNER-139 Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
        )

        return metrics

    def benchmark_against_finer139(self) -> dict[str, Any]:
        """
        Real benchmark from training on the full FiNER-139 dataset.

        Returns:
            Benchmark numbers, or a placeholder until training has been run
        """
        # TODO: fill in after running Train_FinBERT_NER_Colab.ipynb and
        # evaluating on the full 108,378-sentence test split.
        return {
            "status": "not yet measured - run Train_FinBERT_NER_Colab.ipynb",
            "dataset": "nlpaueb/finer-139",
            "test_split_size": 108378,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Test-set format (see evaluation/_cli.py for the accepted container shapes).
# Each sample:
#
#     {
#       "tokens": ["Net", "sales", "were", "416,161", "."],
#       "tags": ["O", "O", "O", "B-Revenues", "O"],
#       "predicted_tags": ["O", "O", "O", "B-Revenues", "O"]   # optional
#     }
#
# "tags" is the ground truth (required); "predicted_tags" is optional -
# omit it and pass --run-agent to generate it from "tokens" live, or omit
# it without --run-agent and the sample scores as all-"O" (a complete miss
# on every entity in it).


def _tokens_to_text_with_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """
    Join tokens with single spaces, tracking each token's character span.

    Args:
        tokens: Pre-tokenized words

    Returns:
        Tuple of (joined text, list of (start, end) offsets, one per token)
    """
    parts = []
    offsets = []
    cursor = 0

    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        parts.append(token)
        cursor = end + 1  # +1 for the joining space

    return " ".join(parts), offsets


def _spans_to_iob2(
    entities: list[dict[str, Any]], token_offsets: list[tuple[int, int]]
) -> list[str]:
    """
    Convert FinancialNERAgent's character-span entities into a per-token
    IOB2 tag sequence aligned to the original token list.

    Args:
        entities: Entity dicts with "tag", "start", "end" (from tag_texts)
        token_offsets: Per-token (start, end) character spans

    Returns:
        IOB2 tag sequence, one entry per token
    """
    tags = ["O"] * len(token_offsets)

    for entity in entities:
        start, end = entity.get("start"), entity.get("end")
        if start is None or end is None:
            continue

        tag = entity.get("tag", "unknown")
        is_first_token = True

        for i, (token_start, token_end) in enumerate(token_offsets):
            if token_start < end and token_end > start:
                tags[i] = f"{'B' if is_first_token else 'I'}-{tag}"
                is_first_token = False

    return tags


def _predict_with_agent(samples: list[dict[str, Any]]) -> None:
    """
    Populate each sample's "predicted_tags" by running FinancialNERAgent
    over its "tokens", joined into text and re-aligned to token boundaries.

    Args:
        samples: Test-set samples, mutated in place
    """
    from src.agents.financial_ner_agent import FinancialNERAgent

    agent = FinancialNERAgent()

    texts = []
    offsets_per_sample = []

    for sample in samples:
        tokens = sample.get("tokens", [])
        text, offsets = _tokens_to_text_with_offsets(tokens)
        texts.append(text)
        offsets_per_sample.append(offsets)

    per_text_entities = agent.tag_texts(texts)

    for sample, entities, offsets in zip(samples, per_text_entities, offsets_per_sample):
        sample["predicted_tags"] = _spans_to_iob2(entities, offsets)

    logger.info(f"Generated predictions for {len(samples)} samples")


def main() -> None:
    """CLI entry point for FiNER-139 evaluation."""
    parser = build_parser(
        description="Evaluate FinancialNERAgent against a FiNER-139 style test set",
        agent_help="Load FinancialNERAgent and tag each sample's 'tokens'",
    )
    args = parser.parse_args()

    samples = load_samples(args)

    if args.run_agent:
        _predict_with_agent(samples)

    reference_tags: list[list[str]] = []
    predicted_tags: list[list[str]] = []
    without_predictions = 0

    for index, sample in enumerate(samples):
        tags = sample.get("tags")

        if tags is None:
            raise SystemExit(
                f"Sample {index} has no 'tags' key. Every sample needs ground-truth tags."
            )

        predicted = sample.get("predicted_tags")
        if predicted is None:
            without_predictions += 1
            predicted = ["O"] * len(tags)

        reference_tags.append(tags)
        predicted_tags.append(predicted)

    if without_predictions:
        logger.warning(
            f"{without_predictions}/{len(samples)} samples carried no 'predicted_tags' "
            "and were scored as complete misses. Pass --run-agent to generate them."
        )

    evaluator = FinerEvaluator()
    metrics = evaluator.evaluate(predicted_tags, reference_tags)

    payload: dict[str, Any] = {
        "note": "scored against a local sample of the real test split - see module docstring",
        "samples_evaluated": metrics.total_sentences,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "entity_type_metrics": metrics.entity_type_metrics,
    }

    if args.benchmark:
        payload["finer139_benchmark"] = evaluator.benchmark_against_finer139()

    print_metrics("FINANCIAL ENTITY EVALUATION (FiNER-139)", payload)
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
