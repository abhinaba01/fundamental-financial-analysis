"""
LangGraph GraphState definition for the financial analysis pipeline.

This TypedDict defines the complete state shared across all agent nodes.
Each field represents either input data or output from a pipeline stage.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.preprocessing.document import DocumentChunk, ParsedDocument


class GraphState(TypedDict):
    """
    State dictionary for LangGraph pipeline execution.

    Fields:
        document: The ParsedDocument being analyzed (input to pipeline)
        query: User question or request (used in RAG mode)
        retrieved_chunks: Chunks returned by the retriever
        retrieval_score: Average cosine similarity of retrieved chunks
        retry_count: Retries performed so far (0 during the first attempt).
            Read by the routing edge to enforce the retry budget, and reported
            as "retries_performed".
        rag_attempts: Total times the retrieve node has run, including the
            first. Tracked separately from retry_count so the loop terminates
            even when an attempt returns zero chunks.
        ner_results: Named entity extraction results
        sentiment_results: Sentiment analysis results dictionary
        kpi_results: Key performance indicator extraction results
        cot_reasoning: Chain-of-thought reasoning trace for transparency
        final_answer: Synthesized answer from the RAG agent
        report: Final structured report with all findings
    """

    document: ParsedDocument
    query: str
    retrieved_chunks: list[DocumentChunk]
    retrieval_score: float
    retry_count: int
    rag_attempts: int
    ner_results: list[dict[str, Any]]
    sentiment_results: dict[str, Any]
    kpi_results: dict[str, Any]
    cot_reasoning: str
    final_answer: str
    report: dict[str, Any]
