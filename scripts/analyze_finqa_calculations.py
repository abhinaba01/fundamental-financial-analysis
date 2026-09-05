"""
Break down eval_kpi's calculation accuracy on FinQA by operator, and test one
specific hypothesis about the failures.

Headline calculation accuracy on the converted FinQA set is ~0.47, which reads
like the arithmetic is broken. It is not. Almost all of the loss sits in one
operator and one convention: FinQA states the answer to a ratio question as a
percentage ("14.0"), while `divide` returns the ratio ("0.1446"). The two
differ by exactly 100x, and no amount of arithmetic correctness closes that
gap.

This script quantifies the split so the claim is reproducible rather than
asserted:

    python scripts/analyze_finqa_calculations.py

It deliberately reports the percentage-adjusted figure *alongside* the raw one
rather than rescaling the test set. Rescaling would make the metric pass by
editing the ground truth, which is the wrong direction - the mismatch is a real
property of converting FinQA into this harness's format, and hiding it would
misrepresent what the harness measures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Mirrors evaluate_calculation_correctness in evaluation/eval_kpi.py.
TOLERANCE = 0.01


def _apply(operands: list[float], operator: str) -> float | None:
    if operator == "+":
        return sum(operands)
    if operator == "-":
        return operands[0] - sum(operands[1:]) if operands else 0.0
    if operator == "*":
        result = 1.0
        for value in operands:
            result *= value
        return result
    if operator == "/":
        result = operands[0]
        for value in operands[1:]:
            result /= value if value != 0 else 1
        return result
    return None


def _within_tolerance(result: float, expected: float) -> bool:
    return abs(result - expected) <= TOLERANCE * max(1.0, abs(expected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", default="data/eval/finqa_test.json")
    args = parser.parse_args()

    path = Path(args.test_set)
    if not path.exists():
        print(f"Not found: {path}\nRun: python scripts/prepare_eval_datasets.py --dataset finqa")
        return 1

    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]

    # per operator: [correct, total, fixed_by_percentage_scaling]
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for sample in samples:
        for step in sample.get("calculation_steps", []):
            operands = step["operands"]
            operator = step["operator"]
            expected = step["expected_result"]

            result = _apply(operands, operator)
            if result is None:
                continue

            entry = stats[operator]
            entry[1] += 1

            if _within_tolerance(result, expected):
                entry[0] += 1
            elif _within_tolerance(result * 100, expected):
                entry[2] += 1

    total = sum(entry[1] for entry in stats.values())
    correct = sum(entry[0] for entry in stats.values())
    pct_fixed = sum(entry[2] for entry in stats.values())

    print(f"Test set: {path}  ({len(samples)} samples, {total} calculation steps)\n")
    print(f"{'op':<4}{'correct':>9}{'total':>8}{'accuracy':>10}{'x100 match':>12}")
    for operator, (ok, seen, scaled) in sorted(stats.items()):
        print(f"{operator:<4}{ok:>9}{seen:>8}{ok / seen:>10.3f}{scaled:>12}")

    print("\n" + "=" * 58)
    print(f"raw calculation accuracy:        {correct / total:.3f}  ({correct}/{total})")
    print(
        f"explained by percent convention: {pct_fixed / total:.3f}  ({pct_fixed}/{total})"
    )
    print(
        f"percentage-adjusted accuracy:    {(correct + pct_fixed) / total:.3f}  "
        f"({correct + pct_fixed}/{total})"
    )
    print("=" * 58)
    print(
        "\nThe gap between the first and last line is a unit convention in the\n"
        "benchmark, not an arithmetic error in the evaluator."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
