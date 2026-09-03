"""
Shared CLI plumbing for the evaluation modules.

Every ``evaluation/eval_*.py`` module builds its ``main()`` on the helpers here,
so the commands documented in the README actually run:

    python -m evaluation.eval_ner --test-set data/eval/finer139_test.json

Test sets are JSON files. All four modules accept the same three shapes:

    [{...}, {...}]                  # bare list of samples
    {"samples": [{...}, {...}]}     # samples under a key
    {...}                           # a single sample object

The per-sample keys differ by evaluator and are documented in each module's
docstring. Samples may carry model predictions inline (offline scoring, no
models loaded) or only ground truth plus the source text, in which case
``--run-agent`` loads the corresponding agent and generates predictions first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keys under which a test-set file may nest its list of samples.
SAMPLE_KEYS = ("samples", "data", "examples")


def build_parser(
    description: str,
    *,
    supports_agent: bool = True,
    agent_help: str = "Load the agent and generate predictions from each sample's text",
) -> argparse.ArgumentParser:
    """
    Build the argument parser shared by every evaluation CLI.

    Args:
        description: Parser description shown in --help
        supports_agent: Whether this evaluator can generate live predictions
        agent_help: Help text for --run-agent

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--test-set",
        required=True,
        help="Path to the JSON test set",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the computed metrics to this JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also print the published reference numbers for this dataset",
    )

    if supports_agent:
        parser.add_argument(
            "--run-agent",
            action="store_true",
            help=agent_help,
        )

    return parser


def load_test_set(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Load a test set and normalise it to a list of sample dicts.

    Args:
        path: Path to the JSON test set
        limit: Keep only the first N samples

    Returns:
        List of sample dictionaries

    Raises:
        FileNotFoundError: Test set does not exist
        ValueError: File is not JSON, or is not a recognised test-set shape
    """
    test_path = Path(path)

    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")

    try:
        with open(test_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Test set is not valid JSON: {test_path} ({exc})") from exc

    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict):
        samples = next(
            (payload[key] for key in SAMPLE_KEYS if isinstance(payload.get(key), list)),
            [payload],
        )
    else:
        raise ValueError(
            f"Unsupported test-set shape in {test_path}: expected a list or object, "
            f"got {type(payload).__name__}"
        )

    non_dict = [i for i, sample in enumerate(samples) if not isinstance(sample, dict)]
    if non_dict:
        raise ValueError(
            f"Samples must be JSON objects; entries at index {non_dict[:5]} are not"
        )

    if limit is not None:
        samples = samples[:limit]

    logger.info(f"Loaded {len(samples)} samples from {test_path}")

    return samples


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    Load the test set named on the command line, exiting cleanly on bad input.

    Args:
        args: Parsed arguments carrying .test_set and .limit

    Returns:
        Non-empty list of sample dictionaries

    Raises:
        SystemExit: Test set is missing, malformed, or empty
    """
    try:
        samples = load_test_set(args.test_set, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if not samples:
        raise SystemExit(f"error: test set contains no samples: {args.test_set}")

    return samples


def to_dict(metrics: Any) -> dict[str, Any]:
    """Convert a metrics dataclass (or plain dict) to a JSON-serialisable dict."""
    if is_dataclass(metrics) and not isinstance(metrics, type):
        return asdict(metrics)
    if isinstance(metrics, dict):
        return metrics
    raise TypeError(f"Cannot serialise metrics of type {type(metrics).__name__}")


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    """
    Print a metrics dict as an indented report.

    Args:
        title: Section heading
        metrics: Metrics to print (nested dicts are indented)
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    _print_block(metrics, indent=0)


def _print_block(block: dict[str, Any], indent: int) -> None:
    """Recursively print a metrics block with aligned keys."""
    pad = " " * indent

    for key, value in block.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_block(value, indent + 2)
        elif isinstance(value, float):
            print(f"{pad}{key:<32} {value:.4f}")
        else:
            print(f"{pad}{key:<32} {value}")


def write_output(path: str | Path | None, payload: dict[str, Any]) -> None:
    """
    Write metrics to a JSON file if an output path was given.

    Args:
        path: Destination path, or None to skip
        payload: Metrics payload to serialise
    """
    if path is None:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    logger.info(f"Metrics written to: {output_path}")
    print(f"\nMetrics written to: {output_path}")
