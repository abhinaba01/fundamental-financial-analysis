"""
Main Entry Point: Financial Fundamentals Analysis Pipeline.

Example usage:
    python -m src.main --document data/raw/AAPL_10K_2023.pdf --query "What are the key risks?"

Or programmatically:
    from src.main import run_analysis
    report = run_analysis("path/to/document.pdf", "What is the revenue trend?")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing.parser import DocumentParser
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.chunker import SemanticChunker
from src.preprocessing.embedder import EmbeddingPipeline
from src.graph.builder import create_default_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_analysis(
    document_path: str | Path,
    query: str,
    use_gpu: bool = True,
) -> dict:
    """
    Run complete financial analysis pipeline on a document.

    Args:
        document_path: Path to PDF/HTML/TXT document
        query: User query about the document
        use_gpu: Whether to use GPU for embeddings

    Returns:
        Structured report with all analysis results
    """
    document_path = Path(document_path)

    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    logger.info(f"Starting analysis pipeline for: {document_path}")

    # Stage 1: Parse
    logger.info("Stage 1: Parsing document...")
    parser = DocumentParser()
    parsed_doc = parser.parse(document_path)
    logger.info(f"Parsed: {len(parsed_doc.chunks)} chunks, {len(parsed_doc.tables)} tables")

    # Stage 2: Clean
    logger.info("Stage 2: Cleaning text...")
    cleaner = DocumentCleaner()
    cleaned_doc = cleaner.clean(parsed_doc)
    logger.info(f"Cleaned: {len(cleaned_doc.chunks)} chunks")

    # Stage 3: Chunk
    logger.info("Stage 3: Semantic chunking...")
    chunker = SemanticChunker()
    chunked_doc = chunker.chunk(cleaned_doc)
    logger.info(f"Chunked: {len(chunked_doc.chunks)} semantic chunks")

    # Stage 4: Embed
    logger.info("Stage 4: Embedding and indexing...")
    embedder = EmbeddingPipeline(device="cuda" if use_gpu else "cpu")
    embedder.embed_and_index(chunked_doc)
    logger.info(f"Indexed: {len(chunked_doc.chunks)} embeddings")

    # Stage 5: Build and run analysis pipeline
    logger.info("Stage 5: Running analysis pipeline...")
    graph = create_default_pipeline(embedding_pipeline=embedder)

    # Initialize state
    initial_state = {
        "document": chunked_doc,
        "query": query,
        "retrieved_chunks": [],
        "retrieval_score": 0.0,
        "retry_count": 0,
        "ner_results": {},
        "sentiment_results": {},
        "kpi_results": {},
        "cot_reasoning": "",
        "final_answer": "",
        "report": {},
    }

    # Execute pipeline
    logger.info("Executing LangGraph pipeline...")
    final_state = graph.invoke(initial_state)

    report = final_state.get("report", {})
    logger.info("Analysis complete")

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Financial Fundamentals Analysis System"
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Path to document (PDF/HTML/TXT)",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Analysis query",
    )
    parser.add_argument(
        "--output",
        default="report.json",
        help="Output file for report (default: report.json)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )

    args = parser.parse_args()

    # Run analysis
    report = run_analysis(
        document_path=args.document,
        query=args.query,
        use_gpu=not args.cpu,
    )

    # Save report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Report saved to: {output_path}")
    if isinstance(report, dict):
        print(f"Summary: {report.get('summary', 'N/A')}")


if __name__ == "__main__":
    main()
