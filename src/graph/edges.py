"""
Graph edge functions: Conditional routing logic for LangGraph.

Defines transitions between nodes based on state conditions:
- Re-query trigger: When <3 chunks AND similarity <0.75, re-query
- Sequential flow: Parser → Cleaner → Chunker → Embedder → Agents → Synthesis

Input: GraphState after each node
Output: Next node name (str) or END
"""

from __future__ import annotations

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration for edge conditions
SIMILARITY_THRESHOLD = 0.75
MIN_CHUNKS_THRESHOLD = 3
MAX_RETRIES = 2


def should_retrieve_again(state: GraphState) -> str:
    """
    Determine if RAG retrieval should be retried.

    Condition: Retry if (<3 chunks) AND (similarity <0.75) AND (retries <max)

    Args:
        state: Current GraphState

    Returns:
        "retrieve" if should re-query, "synthesize" to proceed to synthesis
    """
    retrieved_chunks = state.get("retrieved_chunks", [])
    retrieval_score = state.get("retrieval_score", 0.0)
    retry_count = state.get("retry_count", 0)

    # Check conditions for re-query
    low_chunk_count = len(retrieved_chunks) < MIN_CHUNKS_THRESHOLD
    low_similarity = retrieval_score < SIMILARITY_THRESHOLD
    within_max_retries = retry_count < MAX_RETRIES

    should_retry = low_chunk_count and low_similarity and within_max_retries

    if should_retry:
        logger.info(
            f"Re-query triggered: chunks={len(retrieved_chunks)}, "
            f"similarity={retrieval_score:.3f}, retries={retry_count}"
        )
        return "retrieve"

    logger.info("Retrieval sufficient. Proceeding to synthesis.")
    return "synthesize"


def determine_next_agent(state: GraphState) -> str:
    """
    Determine next agent in pipeline.

    Sequential flow: NER → Sentiment → KPI → RAG → Synthesis

    Args:
        state: Current GraphState

    Returns:
        Next agent node name
    """
    ner_done = state.get("ner_results")
    sentiment_done = state.get("sentiment_results")
    kpi_done = state.get("kpi_results")
    rag_done = state.get("final_answer")

    current_phase = (
        bool(ner_done),
        bool(sentiment_done),
        bool(kpi_done),
        bool(rag_done),
    )

    # Define transition sequence
    if not ner_done:
        logger.debug("Next: NER agent")
        return "ner"
    elif not sentiment_done:
        logger.debug("Next: Sentiment agent")
        return "sentiment"
    elif not kpi_done:
        logger.debug("Next: KPI agent")
        return "kpi"
    elif not rag_done:
        logger.debug("Next: RAG agent")
        return "rag"
    else:
        logger.debug("Next: Synthesis")
        return "synthesis"


def route_after_synthesis(state: GraphState) -> str:
    """
    Route after synthesis to END.

    Args:
        state: Final GraphState

    Returns:
        END to complete graph traversal
    """
    logger.info("Analysis pipeline complete. Exiting.")
    return "__end__"
