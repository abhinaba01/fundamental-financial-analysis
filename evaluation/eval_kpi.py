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
