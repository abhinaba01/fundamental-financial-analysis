"""
Graph edge functions: Conditional routing logic for LangGraph.

Defines the one conditional transition in the pipeline: after a retrieval
attempt, either loop back and re-query, or move on to generation.

Input: GraphState after the retrieve node
Output: Next node key ("retrieve" or "generate")
"""

from __future__ import annotations

from src.agents.rag_agent import RETRIEVAL_SIMILARITY_FLOOR
from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Routing thresholds. These decide whether a completed retrieval was good
# enough, which is a different question from RAGAgent's
# RETRIEVAL_SIMILARITY_FLOOR - that one filters individual chunks *during*
# retrieval.
#
# SIMILARITY_THRESHOLD must stay strictly above that floor. Retrieval discards
# every chunk scoring below the floor, so the average of the survivors is
# always >= the floor; a routing threshold at or below it can never fire, and
# the similarity half of the condition becomes dead code. This is exactly the
# bug that used to be here (both were 0.5), which is why the re-query loop
# never triggered on similarity.
#
# BGE relevant-chunk similarities cluster in 0.5-0.65, so 0.6 asks for "better
# than merely admissible" while staying reachable. Above ~0.7 nearly every
# query would retry regardless of retrieval quality.
SIMILARITY_THRESHOLD = 0.6
MIN_CHUNKS_THRESHOLD = 3
MAX_RETRIES = 2

assert SIMILARITY_THRESHOLD > RETRIEVAL_SIMILARITY_FLOOR, (
    "Routing threshold must exceed the retrieval floor or it can never fire "
    f"({SIMILARITY_THRESHOLD} <= {RETRIEVAL_SIMILARITY_FLOOR})"
)


def should_retrieve_again(state: GraphState) -> str:
    """
    Decide whether to re-query or proceed to generation.

    Retries when the evidence looks weak on *either* axis - too few chunks, or
    a weak average similarity across them - and the retry budget is not spent.
    The two signals are independent failure modes: a single highly relevant
    chunk is thin evidence, and five barely-admissible ones are weak evidence.
    Requiring both to be bad (which this previously did) meant that in practice
    neither triggered.

    Args:
        state: Current GraphState

    Returns:
        "retrieve" to re-query, "generate" to proceed
    """
    retrieved_chunks = state.get("retrieved_chunks", [])
    retrieval_score = state.get("retrieval_score", 0.0)
    retry_count = state.get("retry_count", 0)

    low_chunk_count = len(retrieved_chunks) < MIN_CHUNKS_THRESHOLD
    low_similarity = retrieval_score < SIMILARITY_THRESHOLD
    within_max_retries = retry_count < MAX_RETRIES

    should_retry = (low_chunk_count or low_similarity) and within_max_retries

    if should_retry:
        reasons = []
        if low_chunk_count:
            reasons.append(f"only {len(retrieved_chunks)} chunks")
        if low_similarity:
            reasons.append(f"avg similarity {retrieval_score:.3f} < {SIMILARITY_THRESHOLD}")
        logger.info(f"Re-query triggered ({'; '.join(reasons)}), retries so far: {retry_count}")
        return "retrieve"

    if low_chunk_count or low_similarity:
        logger.info(
            f"Retrieval still weak after {retry_count} retries "
            f"({len(retrieved_chunks)} chunks, avg similarity {retrieval_score:.3f}). "
            "Retry budget exhausted; generating from what was found."
        )
    else:
        logger.info(
            f"Retrieval sufficient ({len(retrieved_chunks)} chunks, "
            f"avg similarity {retrieval_score:.3f}). Proceeding to generation."
        )

    return "generate"
