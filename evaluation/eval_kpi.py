"""
Evaluation: Key Performance Indicator (KPI) Extraction Performance.

Evaluates KPIAgent performance using:
- FinQA dataset (numeral reasoning on financial texts)
- Metrics: Numeric Accuracy (±tolerance), Extraction Recall, Type Coverage
- KPI types: Revenue, Margins, EPS, EBITDA, ROA, ROE
- Tolerance: ±0.01 (1% for percentages, 1% error for absolute values)

Benchmarks:
- Program Aggregation Engine: ~73% numeric accuracy on FinQA
- Top teams: ~85-88% accuracy

Calculations:
- Safe Python REPL for derived calculations
- No hallucination-prone LLM invocation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation._cli import (
    build_parser,
    load_samples,
    mean,
    print_metrics,
    to_dict,
    write_output,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KPIMetrics:
    """KPI evaluation metrics."""

    numeric_accuracy: float
    extraction_recall: float
    type_coverage: float
    per_type_accuracy: dict[str, float]
    mean_absolute_error: float
    rmse: float
    total_kpis: int


class KPIEvaluator:
    """Evaluator for KPI extraction agent performance."""

    def __init__(self):
        """Initialize KPI evaluator."""
        self.logger = logger
        self.kpi_types = [
            "revenue",
            "gross_margin",
            "operating_income",
            "net_income",
            "eps",
            "ebitda",
            "roa",
            "roe",
        ]
        self.tolerance = 0.01  # 1% tolerance

    def evaluate(
        self,
        extracted_kpis: dict[str, float],
        reference_kpis: dict[str, float],
    ) -> KPIMetrics:
        """
        Evaluate extracted KPIs against reference values.

        Args:
            extracted_kpis: Extracted KPI values
            reference_kpis: Ground-truth KPI values

        Returns:
            KPIMetrics with accuracy, recall, MAE, RMSE
        """
        self.logger.info(
            f"Evaluating KPIs: {len(extracted_kpis)} extracted vs {len(reference_kpis)} references"
        )

        # Calculate numeric accuracy
        correct_count = 0
        errors = []
        per_type_accuracy = {}

        for kpi_type in self.kpi_types:
            if kpi_type not in reference_kpis:
                continue

            ref_value = reference_kpis[kpi_type]
            ext_value = extracted_kpis.get(kpi_type)

            if ext_value is None:
                continue

            # Calculate relative error
            if ref_value == 0:
                error = abs(ext_value - ref_value)
            else:
                error = abs(ext_value - ref_value) / abs(ref_value)

            errors.append(error)

            # Check if within tolerance
            if error <= self.tolerance:
                correct_count += 1

            # Per-type accuracy
            type_total = 1
            type_correct = 1 if error <= self.tolerance else 0
            per_type_accuracy[kpi_type] = type_correct / type_total

        # Calculate overall metrics
        numeric_accuracy = correct_count / len(reference_kpis) if reference_kpis else 0.0

        extraction_recall = len(extracted_kpis) / len(reference_kpis) if reference_kpis else 0.0

        type_coverage = len(per_type_accuracy) / len(self.kpi_types)

        mae = sum(errors) / len(errors) if errors else 0.0
        rmse = (sum(e**2 for e in errors) / len(errors))**0.5 if errors else 0.0

        metrics = KPIMetrics(
            numeric_accuracy=numeric_accuracy,
            extraction_recall=extraction_recall,
            type_coverage=type_coverage,
            per_type_accuracy=per_type_accuracy,
            mean_absolute_error=mae,
            rmse=rmse,
            total_kpis=len(reference_kpis),
        )

        self.logger.info(
            f"KPI Metrics - Accuracy: {numeric_accuracy:.3f}, "
            f"Recall: {extraction_recall:.3f}, MAE: {mae:.4f}"
        )

        return metrics

    def benchmark_against_finqa(self) -> dict[str, float]:
        """
        Benchmark against FinQA dataset expectations.

        Expected performance (Program Aggregation Engine):
        - Numeric Accuracy: ~73%
        - Top teams: ~85-88%
        - Arithmetic operations support: +, -, *, /, %, comparisons
        - Program synthesis: 30% of cases use multi-step reasoning

        Returns:
            Benchmark metrics
        """
        benchmarks = {
            "numeric_accuracy_bottom": 0.73,
            "numeric_accuracy_top": 0.88,
            "expected_accuracy": 0.80,
            "multi_step_percentage": 0.30,
        }

        self.logger.info("Using FinQA benchmarks...")
        return benchmarks

    def evaluate_calculation_correctness(
        self,
        calculation_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate correctness of derived calculations.

        Args:
            calculation_steps: List of calculation steps with operands and operators

        Returns:
            Verification metrics
        """
        self.logger.info(f"Evaluating {len(calculation_steps)} calculation steps...")

        correct_calculations = 0

        for step in calculation_steps:
            operands = step.get("operands", [])
            operator = step.get("operator", "+")
            expected_result = step.get("expected_result", None)

            try:
                # Perform calculation safely
                if operator == "+":
                    result = sum(operands)
                elif operator == "-":
                    result = operands[0] - sum(operands[1:]) if operands else 0
                elif operator == "*":
                    result = 1
                    for op in operands:
                        result *= op
                elif operator == "/":
                    result = operands[0]
                    for op in operands[1:]:
                        result /= op if op != 0 else 1
                elif operator == "%":
                    result = operands[0] * operands[1] / 100 if len(operands) > 1 else operands[0]
                else:
                    result = None

                if result is not None and expected_result is not None:
                    if abs(result - expected_result) <= self.tolerance * max(1, abs(expected_result)):
                        correct_calculations += 1

            except Exception as e:
                self.logger.warning(f"Calculation error: {e}")

        calculation_accuracy = (
            correct_calculations / len(calculation_steps) if calculation_steps else 0.0
        )

        return {
            "calculation_accuracy": calculation_accuracy,
            "correct_steps": correct_calculations,
            "total_steps": len(calculation_steps),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Test-set format (see evaluation/_cli.py for the accepted container shapes).
# Each sample:
#
#     {
#       "text": "Total net sales of 394328 ...",     # required for --run-agent
#       "reference_kpis": {"revenue": 394328.0},     # ground truth
#       "extracted_kpis": {"revenue": 394328.0},     # optional
#       "calculation_steps": [                       # optional, scored separately
#         {"operands": [169148, 394328], "operator": "/", "expected_result": 0.4289}
#       ]
#     }
#
# "gold_kpis" is accepted as an alias for "reference_kpis"; "predicted_kpis" and
# "predictions" as aliases for "extracted_kpis".


def _coerce_kpi_values(kpis: dict[str, Any] | None, *, context: str) -> dict[str, float]:
    """
    Coerce a KPI mapping to {name: float}, dropping entries that are not numeric.

    Args:
        kpis: Raw KPI mapping from a test set
        context: Description used in warnings

    Returns:
        Mapping of KPI name to float value
    """
    coerced: dict[str, float] = {}

    for name, value in (kpis or {}).items():
        try:
            coerced[name] = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Dropping non-numeric KPI {name}={value!r} in {context}")

    return coerced


def _predict_with_agent(samples: list[dict[str, Any]]) -> None:
    """
    Populate each sample's "extracted_kpis" by running KPIAgent over its text.

    The agent returns every regex hit per KPI type; the evaluator scores a single
    value per type, so the first match is taken.

    Args:
        samples: Test-set samples, mutated in place
    """
    from src.agents.kpi_agent import KPIAgent

    agent = KPIAgent()

    for index, sample in enumerate(samples):
        text = sample.get("text", "")

        if not text:
            logger.warning(f"Sample {index} has no 'text'; extracting no KPIs")
            sample["extracted_kpis"] = {}
            continue

        matches = agent._extract_kpi_patterns(text)
        sample["extracted_kpis"] = {
            kpi_type: entries[0]["value"] for kpi_type, entries in matches.items() if entries
        }

    logger.info(f"Generated KPI extractions for {len(samples)} samples")


def main() -> None:
    """CLI entry point for KPI evaluation."""
    parser = build_parser(
        description="Evaluate the KPI agent against a FinQA style test set",
        agent_help="Load KPIAgent and extract KPIs from each sample's 'text'",
    )
    args = parser.parse_args()

    samples = load_samples(args)

    if args.run_agent:
        _predict_with_agent(samples)

    evaluator = KPIEvaluator()

    per_sample: list[KPIMetrics] = []
    per_type_values: dict[str, list[float]] = {}
    calculation_steps: list[dict[str, Any]] = []
    without_extractions = 0

    for index, sample in enumerate(samples):
        reference = sample.get("reference_kpis", sample.get("gold_kpis"))

        if reference is None:
            raise SystemExit(
                f"Sample {index} has no 'reference_kpis' (or 'gold_kpis') key. "
                "Every sample needs ground-truth KPI values."
            )

        extracted = sample.get(
            "extracted_kpis", sample.get("predicted_kpis", sample.get("predictions"))
        )

        if extracted is None:
            without_extractions += 1
            extracted = {}

        metrics = evaluator.evaluate(
            _coerce_kpi_values(extracted, context=f"sample {index} extraction"),
            _coerce_kpi_values(reference, context=f"sample {index} ground truth"),
        )
        per_sample.append(metrics)

        for kpi_type, accuracy in metrics.per_type_accuracy.items():
            per_type_values.setdefault(kpi_type, []).append(accuracy)

        calculation_steps.extend(sample.get("calculation_steps", []))

    if without_extractions:
        logger.warning(
            f"{without_extractions}/{len(samples)} samples carried no 'extracted_kpis' "
            "and were scored as complete misses. Pass --run-agent to generate them."
        )

    payload: dict[str, Any] = {
        "aggregation": "macro (mean across samples)",
        "samples_evaluated": len(samples),
        "numeric_accuracy": mean([m.numeric_accuracy for m in per_sample]),
        "extraction_recall": mean([m.extraction_recall for m in per_sample]),
        "type_coverage": mean([m.type_coverage for m in per_sample]),
        "mean_absolute_error": mean([m.mean_absolute_error for m in per_sample]),
        "mean_rmse": mean([m.rmse for m in per_sample]),
        "per_type_accuracy": {
            kpi_type: mean(values) for kpi_type, values in sorted(per_type_values.items())
        },
        "total_reference_kpis": sum(m.total_kpis for m in per_sample),
    }

    if calculation_steps:
        payload["calculations"] = evaluator.evaluate_calculation_correctness(calculation_steps)

    if args.benchmark:
        payload["finqa_benchmark"] = evaluator.benchmark_against_finqa()

    print_metrics("KPI EVALUATION (FinQA)", payload)
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
