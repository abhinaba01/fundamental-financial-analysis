"""
Download the real benchmark datasets and convert them into the test-set format
the evaluation harnesses read.

The files in data/eval/*_example.json are hand-written worked examples that
prove the scoring code runs. They are not benchmarks. This script produces the
real thing, into data/eval/<name>_test.json.

    python scripts/prepare_eval_datasets.py --dataset phrasebank
    python scripts/prepare_eval_datasets.py --all --limit 200

Three caveats that matter when reading the resulting numbers, each of which is
a property of the dataset rather than of this code:

1. **Financial PhraseBank is training data for ProsusAI/finbert.** The model
   was fine-tuned on it, so scoring finbert here measures memorization as much
   as generalization. It is still the number the literature quotes, so it is
   worth reproducing - but it is not evidence the model generalizes.

2. **FinanceBench retrieval is simplified here.** The real benchmark retrieves
   from full filing PDFs. This builds the corpus out of the evidence passages
   pooled across all samples, so retrieval means "find the right passage among
   all passages" - a real but easier task than searching whole filings.

3. **FinQA does not map onto KPIAgent's task at all.** FinQA is multi-step
   numerical reasoning; KPIAgent is a regex extractor for eight named KPI
   types. The conversion therefore targets the harness's *calculation* metric
   and leaves reference_kpis empty, so the extraction metrics come back as a
   meaningless zero over zero references. Read `calculations`, not
   `numeric_accuracy`, from that run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("data/eval")

PHRASEBANK_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# FinQA program operators -> the operator strings evaluate_calculation_correctness
# understands. Anything outside this set (exp, greater, table ops) is skipped.
FINQA_OPERATORS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

# A single-operation FinQA program over two literal numbers, e.g.
# "subtract(206588, 181001)". Programs referencing an earlier step ("#0") are
# multi-step and deliberately not converted - see _parse_finqa_program.
_SINGLE_OP = re.compile(
    r"^(add|subtract|multiply|divide)\(\s*(-?[\d.,]+)\s*,\s*(-?[\d.,]+)\s*\)$"
)


def _to_float(raw: str) -> float | None:
    """Parse a FinQA numeric literal, tolerating commas, %, and $."""
    cleaned = str(raw).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _write(name: str, samples: list[dict], note: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_test.json"
    path.write_text(
        json.dumps({"_comment": note, "samples": samples}, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {len(samples)} samples -> {path}")
    return path


def prepare_phrasebank(limit: int | None, config: str = "sentences_allagree") -> None:
    """Financial PhraseBank -> eval_sentiment format."""
    from datasets import load_dataset

    print(f"[phrasebank] loading takala/financial_phrasebank ({config})")
    rows = load_dataset("takala/financial_phrasebank", config, trust_remote_code=True)["train"]

    samples = [
        {"text": row["sentence"], "sentiment": PHRASEBANK_LABELS[row["label"]]}
        for row in rows
    ]
    if limit:
        samples = samples[:limit]

    _write(
        "phrasebank",
        samples,
        f"Financial PhraseBank ({config}), converted for eval_sentiment. NOTE: "
        "ProsusAI/finbert was fine-tuned on this dataset, so scores here reflect "
        "training data, not held-out generalization.",
    )


def prepare_financebench(limit: int | None) -> None:
    """FinanceBench -> eval_rag format."""
    from datasets import load_dataset

    print("[financebench] loading PatronusAI/financebench")
    rows = load_dataset("PatronusAI/financebench", split="train")

    samples = []
    for row in rows:
        question, answer = row.get("question"), row.get("answer")
        if not question or not answer:
            continue

        # `evidence` is a list of dicts carrying the passage text under one of a
        # couple of key names depending on the dataset revision.
        chunks = []
        for item in row.get("evidence") or []:
            if isinstance(item, dict):
                text = item.get("evidence_text") or item.get("text")
            else:
                text = item
            if text:
                chunks.append(text)

        if not chunks:
            continue

        samples.append({
            "question": question,
            "answer": str(answer),
            "gold_chunks": [
                {"chunk_id": f"{row.get('financebench_id', len(samples))}-{i}", "text": text}
                for i, text in enumerate(chunks)
            ],
        })

    if limit:
        samples = samples[:limit]

    _write(
        "financebench",
        samples,
        "FinanceBench converted for eval_rag. Retrieval corpus is the pooled "
        "evidence passages, not the full source PDFs - an easier retrieval task "
        "than the published benchmark.",
    )


def _parse_finqa_program(program: str, gold_answer: str) -> dict | None:
    """
    Convert a single-operation FinQA program into a calculation step.

    expected_result comes from FinQA's gold answer, never from evaluating the
    program here. evaluate_calculation_correctness recomputes the result from
    the operands, so deriving expected_result the same way would make the
    metric trivially perfect and measure nothing.

    Multi-step programs (those referencing "#0") are skipped rather than
    flattened, since the harness scores one operation per step.

    Args:
        program: FinQA program string, e.g. "subtract(206588, 181001)"
        gold_answer: FinQA's ground-truth answer

    Returns:
        A calculation-step dict, or None if this program is not convertible
    """
    match = _SINGLE_OP.match((program or "").strip())
    if not match:
        return None

    name, left, right = match.groups()
    operands = [_to_float(left), _to_float(right)]
    expected = _to_float(gold_answer)

    if expected is None or any(value is None for value in operands):
        return None

    return {
        "operands": operands,
        "operator": FINQA_OPERATORS[name],
        "expected_result": expected,
    }


def prepare_finqa(limit: int | None) -> None:
    """FinQA -> eval_kpi format, targeting the calculation metric only."""
    from datasets import load_dataset

    # ibm/finqa rather than dreamerdeo/finqa: the latter drops `program_re`,
    # keeping only question/answer/evidence. Without the gold program there is
    # no honest way to build a calculation step - inferring the operator by
    # testing which one reproduces the answer would guarantee a perfect score
    # and measure nothing.
    print("[finqa] loading ibm/finqa")
    rows = load_dataset("ibm/finqa", split="test", trust_remote_code=True)

    print(f"  columns: {list(rows.features)}")

    samples = []
    skipped_multistep = 0

    for row in rows:
        program = row.get("program_re") or row.get("program") or ""
        answer = row.get("answer") or row.get("final_result") or ""

        step = _parse_finqa_program(program, str(answer))
        if step is None:
            skipped_multistep += 1
            continue

        context = " ".join(
            part if isinstance(part, str) else " ".join(map(str, part))
            for part in (row.get("pre_text") or [], row.get("post_text") or [])
            if part
        )

        samples.append({
            "text": context[:4000],
            "question": row.get("question", ""),
            # Intentionally empty: FinQA has no typed-KPI ground truth. The
            # extraction metrics from this file are 0/0 and mean nothing.
            "reference_kpis": {},
            "calculation_steps": [step],
        })

    if limit:
        samples = samples[:limit]

    print(f"  skipped {skipped_multistep} multi-step/unsupported programs")
    _write(
        "finqa",
        samples,
        "FinQA single-operation subset, converted for eval_kpi's calculation "
        "metric. reference_kpis is empty by design - FinQA has no typed-KPI "
        "ground truth, so numeric_accuracy/extraction_recall from this file are "
        "zero over zero references and should be ignored. Read `calculations`.",
    )


DATASETS = {
    "phrasebank": prepare_phrasebank,
    "financebench": prepare_financebench,
    "finqa": prepare_finqa,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Single dataset to prepare")
    parser.add_argument("--all", action="store_true", help="Prepare every dataset")
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per dataset")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("pass --dataset NAME or --all")

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("The `datasets` package is required: pip install -e '.[eval]'")
        return 1

    targets = sorted(DATASETS) if args.all else [args.dataset]

    failed = []
    for name in targets:
        try:
            DATASETS[name](args.limit)
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
        return 1

    print("\nDone. Point the harnesses at the generated files, e.g.:")
    print("  python -m evaluation.eval_sentiment --test-set data/eval/phrasebank_test.json --run-agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
