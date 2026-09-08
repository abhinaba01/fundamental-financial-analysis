"""
Index a converted evaluation test set's gold passages into their own ChromaDB
collection, so eval_rag --run-agent has a corpus to retrieve from.

The retrieval corpus is every gold passage pooled across every sample. A
question's own evidence is therefore one passage among all of them, and
retrieval means picking it out of that pool. That is a real retrieval task but
an easier one than the published FinanceBench setting, which searches whole
filing PDFs - worth stating plainly next to any number this produces.

Writes to a separate collection and store path by default so benchmark data
never mixes into the working vector store used by `python -m src.main`.

    python scripts/index_eval_corpus.py --test-set data/eval/financebench_test.json
    python -m evaluation.eval_rag --test-set data/eval/financebench_test.json \\
        --run-agent --collection financebench_eval --vector-store data/eval_vector_store
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.document import DocumentChunk, ParsedDocument  # noqa: E402
from src.preprocessing.embedder import EmbeddingPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", default="data/eval/financebench_test.json")
    parser.add_argument("--collection", default="financebench_eval")
    parser.add_argument("--vector-store", default="data/eval_vector_store")
    parser.add_argument("--gpu", action="store_true", help="Embed on CUDA")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Embedding model to index with (default: the pipeline's own). Accepts "
            "a Hub id or a local directory, so a fine-tuned model can be indexed "
            "into its own collection and compared against the stock one."
        ),
    )
    args = parser.parse_args()

    path = Path(args.test_set)
    if not path.exists():
        print(f"Not found: {path}\nRun scripts/prepare_eval_datasets.py first.")
        return 1

    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]

    # Pool every gold passage, de-duplicated by chunk id - samples about the
    # same filing often cite overlapping evidence.
    chunks: dict[str, DocumentChunk] = {}
    for sample in samples:
        for gold in sample.get("gold_chunks", []):
            if isinstance(gold, dict):
                chunk_id, text = gold.get("chunk_id"), gold.get("text")
            else:
                chunk_id, text = None, gold
            if not text:
                continue
            chunk_id = str(chunk_id or f"chunk-{len(chunks)}")
            chunks.setdefault(chunk_id, DocumentChunk(chunk_id=chunk_id, text=text))

    if not chunks:
        print("No gold_chunks found in the test set - nothing to index.")
        return 1

    print(f"Indexing {len(chunks)} passages from {len(samples)} samples")
    print(f"  collection: {args.collection}")
    print(f"  store:      {args.vector_store}")

    document = ParsedDocument(
        doc_id=path.stem,
        chunks=list(chunks.values()),
        cleaned_text=" ".join(chunk.text for chunk in chunks.values()),
    )

    overrides = {"model_name": args.embedding_model} if args.embedding_model else {}
    if args.embedding_model:
        print(f"  model:      {args.embedding_model}")

    embedder = EmbeddingPipeline(
        collection_name=args.collection,
        vector_store_path=args.vector_store,
        device="cuda" if args.gpu else "cpu",
        **overrides,
    )
    embedder.embed_and_index(document)

    print(f"\nIndexed. Now run:\n"
          f"  python -m evaluation.eval_rag --test-set {args.test_set} --run-agent \\\n"
          f"      --collection {args.collection} --vector-store {args.vector_store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
