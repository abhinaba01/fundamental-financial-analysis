"""
RAG Agent: Retrieval-Augmented Generation with Chain-of-Thought.

Uses:
- BAAI/bge-large-en-v1.5 for retrieval
- gpt-4o for generation with chain-of-thought (falls back to extractive
  synthesis if OPENAI_API_KEY is not set or the API call fails)

Performs:
- Query-based document chunk retrieval
- Chain-of-thought synthesis of answers
- Evidence citation

Retrieval and generation are exposed as two separate graph nodes. The re-query
loop runs around retrieval alone, so a low-confidence retry re-searches with a
reformulated query and only pays for one generation call once the loop settles,
instead of regenerating an answer on every attempt. `__call__` runs both in
sequence for standalone use (e.g. evaluation/eval_rag.py).

Retry *policy* lives in src/graph/edges.py. This module counts attempts but
does not decide when to stop.

Input: GraphState with document, query, chunks, embeddings available
Output: GraphState with retrieved_chunks, cot_reasoning, final_answer populated
"""

from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration.
# RETRIEVAL_SIMILARITY_FLOOR is a *filter*: chunks scoring below it are dropped
# at retrieval time. It is deliberately not the same knob as edges.py's
# SIMILARITY_THRESHOLD, which is a *routing* decision about the surviving
# chunks' average. Because this floor discards everything below 0.5, the
# average of what comes back is always >= 0.5 - so a routing threshold set to
# 0.5 or lower can never fire. edges.py must stay strictly above this value;
# see the comment there.
#
# BAAI/bge-large-en-v1.5 cosine similarities for genuinely relevant chunks
# typically land around 0.5-0.65, not near 1.0.
RETRIEVAL_TOP_K = 5
RETRIEVAL_SIMILARITY_FLOOR = 0.5
GENERATION_MODEL = "gpt-4o"

# Dropped when a retry strips a question down to content words, so that the
# search text reads like filing prose rather than like a question.
_QUESTION_STOPWORDS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "or",
})
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 1024

# System prompt for CoT reasoning
COT_SYSTEM_PROMPT = """You are a financial analysis expert assistant. When answering questions about financial documents:

1. First, analyze the retrieved evidence chunks carefully
2. Identify key financial metrics, dates, and relationships
3. Show your reasoning step-by-step (Chain-of-Thought)
4. Synthesize the evidence into a clear, well-reasoned answer
5. Always cite the document sections you used

Format your response as:
REASONING: [Step-by-step analysis]
ANSWER: [Synthesized answer with citations]
"""


class RAGAgent:
    """RAG agent with retrieval, chain-of-thought reasoning, and synthesis."""

    def __init__(self, embedding_pipeline=None, generation_model: str = GENERATION_MODEL):
        """
        Initialize the RAG agent.

        Args:
            embedding_pipeline: EmbeddingPipeline instance for retrieval
            generation_model: OpenAI chat model used for CoT generation
        """
        self.logger = logger
        self.embedding_pipeline = embedding_pipeline
        self.generation_model = generation_model

        self.llm_client = None
        api_key = os.environ.get("OPENAI_API_KEY")
        if OpenAI is None:
            self.logger.warning("openai package not installed. RAG will use extractive synthesis only.")
        elif not api_key:
            self.logger.warning(
                "OPENAI_API_KEY not set. RAG will fall back to extractive synthesis "
                "instead of LLM-generated chain-of-thought answers."
            )
        else:
            self.llm_client = OpenAI(api_key=api_key)

        self.logger.info("RAG agent initialized successfully")

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """
        Run retrieval then generation in one pass.

        This is the standalone entry point (evaluation/eval_rag.py uses it).
        The graph wires `retrieve` and `generate` as separate nodes instead, so
        that the re-query loop can iterate on retrieval without regenerating an
        answer each time around.

        Args:
            state: GraphState with document, query, and retrieved_chunks

        Returns:
            State delta from both stages
        """
        delta = self.retrieve(state)
        delta.update(self.generate({**state, **delta}))
        return delta

    def retrieve(self, state: GraphState) -> dict[str, Any]:
        """
        Retrieve chunks for the query. Graph node - may run more than once.

        Retrieval runs on every entry, not just the first. The graph loops back
        here when routing judges confidence too low, and an attempt that reused
        the previous attempt's chunks would search for nothing new - the retry
        would be a no-op. Each attempt therefore reformulates the query.

        This node counts attempts but does not decide when to stop; that bound
        is enforced by should_retrieve_again in src/graph/edges.py. A previous
        version also retried internally here, giving two independent retry
        mechanisms with different trigger conditions.

        Args:
            state: GraphState with query and document

        Returns:
            State delta with retrieved_chunks, retrieval_score, attempt counters
        """
        query = state.get("query")
        document = state.get("document")
        attempts = state.get("rag_attempts", 0)
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not query:
            self.logger.warning("No query in state. Skipping retrieval.")
            return {}

        delta: dict[str, Any] = {
            "rag_attempts": attempts + 1,
            # Retries performed *before* this attempt - i.e. what the report
            # means by "retries_performed". First pass is not a retry.
            "retry_count": attempts,
        }

        if attempts == 0:
            search_query = query
            self.logger.info(f"Running RAG for query: {query[:100]}...")
        else:
            search_query = self._reformulate_query(query, attempts)
            self.logger.info(f"Re-query attempt {attempts}: {search_query[:100]}...")

        # First pass honors chunks a caller supplied directly; a retry always
        # re-retrieves, or looping back here would change nothing.
        if self.embedding_pipeline is not None and (attempts > 0 or not retrieved_chunks):
            chunks, retrieval_score = self._retrieve_chunks(search_query, document)
            delta["retrieved_chunks"] = chunks
            delta["retrieval_score"] = retrieval_score

        return delta

    def generate(self, state: GraphState) -> dict[str, Any]:
        """
        Generate the chain-of-thought answer. Graph node - runs once, after
        the re-query loop has settled on a set of chunks.

        Args:
            state: GraphState with query and retrieved_chunks

        Returns:
            State delta with cot_reasoning and final_answer
        """
        query = state.get("query")
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not query:
            self.logger.warning("No query in state. Skipping generation.")
            return {}

        if not retrieved_chunks:
            self.logger.warning("No chunks retrieved. Unable to generate answer.")
            return {
                "cot_reasoning": "No relevant chunks found in document.",
                "final_answer": "Unable to answer based on available documents.",
            }

        cot_reasoning, final_answer = self._generate_answer(query, retrieved_chunks)

        self.logger.info("RAG synthesis complete")

        return {"cot_reasoning": cot_reasoning, "final_answer": final_answer}

    def _retrieve_chunks(
        self, query: str, document
    ) -> tuple[list, float]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            document: ParsedDocument with ticker for filtering

        Returns:
            Tuple of (retrieved_chunks, average_similarity_score)
        """
        if not self.embedding_pipeline:
            self.logger.warning("No embedding pipeline available. Skipping retrieval.")
            return [], 0.0

        try:
            chunks = self.embedding_pipeline.retrieve(
                query,
                top_k=RETRIEVAL_TOP_K,
                filter_ticker=document.ticker if document else None,
                similarity_threshold=RETRIEVAL_SIMILARITY_FLOOR,
            )

            if chunks:
                scores = [
                    chunk.metadata.get("similarity", 0.0)
                    for chunk in chunks
                ]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                self.logger.info(
                    f"Retrieved {len(chunks)} chunks with avg similarity: {avg_score:.3f}"
                )
                return chunks, avg_score
            else:
                self.logger.warning("No chunks above similarity threshold")
                return [], 0.0

        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return [], 0.0

    def _reformulate_query(self, original_query: str, attempt: int) -> str:
        """
        Reformulate the query for a re-retrieval attempt.

        Each attempt widens the search differently. Returning the same string
        every time would make the second and third attempts re-run an identical
        vector search and retrieve identical chunks, so the reformulation is
        indexed by attempt number.

        Args:
            original_query: The user's original query, unchanged across attempts
            attempt: 1-based retry number (attempt 0 uses the original query)

        Returns:
            Reformulated query
        """
        strategies = [
            # 1st retry: ask for elaboration, which pulls in surrounding context.
            f"Provide detailed information about: {original_query}",
            # 2nd retry: drop question framing down to content words, which
            # embeds closer to declarative filing prose than a question does.
            " ".join(
                word
                for word in original_query.replace("?", "").split()
                if word.lower() not in _QUESTION_STOPWORDS
            )
            or original_query,
        ]

        return strategies[min(attempt, len(strategies)) - 1]

    def _generate_answer(
        self, query: str, chunks
    ) -> tuple[str, str]:
        """
        Generate answer with chain-of-thought reasoning.

        Uses the configured LLM when available; otherwise falls back to
        extractive synthesis directly from the retrieved chunks.

        Args:
            query: Query text
            chunks: Retrieved DocumentChunk objects

        Returns:
            Tuple of (cot_reasoning, final_answer)
        """
        context = self._build_context(chunks)

        if self.llm_client is not None:
            try:
                return self._generate_llm_answer(query, context)
            except Exception as e:
                self.logger.error(f"LLM generation failed, falling back to extractive synthesis: {e}")

        cot_reasoning = self._build_reasoning(query, chunks)
        final_answer = self._synthesize_answer(chunks)
        return cot_reasoning, final_answer

    def _generate_llm_answer(self, query: str, context: str) -> tuple[str, str]:
        """
        Generate a chain-of-thought answer via the configured OpenAI chat model.

        Args:
            query: Query text
            context: Formatted context string built from retrieved chunks

        Returns:
            Tuple of (cot_reasoning, final_answer)
        """
        response = self.llm_client.chat.completions.create(
            model=self.generation_model,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_TOKENS,
            messages=[
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nRetrieved evidence:\n{context}",
                },
            ],
        )

        response_text = response.choices[0].message.content or ""
        return self._parse_response(response_text)

    def _build_context(self, chunks) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            similarity = chunk.metadata.get("similarity", 0.0)
            section = chunk.metadata.get("section", "general")

            context_parts.append(
                f"[Chunk {i}] ({section}, similarity: {similarity:.3f})\n"
                f"{chunk.text[:500]}...\n"
            )

        return "\n".join(context_parts)

    def _build_reasoning(self, query: str, chunks) -> str:
        """
        Build step-by-step reasoning from chunks (extractive fallback).

        Args:
            query: Original query
            chunks: Retrieved chunks

        Returns:
            Reasoning string
        """
        reasoning = f"Question: {query}\n\n"
        reasoning += f"Analyzing {len(chunks)} relevant document sections:\n"

        for i, chunk in enumerate(chunks, 1):
            similarity = chunk.metadata.get("similarity", 0.0)
            reasoning += f"{i}. Section discusses: {chunk.text[:100]}... (relevance: {similarity:.2%})\n"

        return reasoning

    def _synthesize_answer(self, chunks) -> str:
        """
        Synthesize final answer from chunks (extractive fallback).

        Args:
            chunks: Retrieved chunks

        Returns:
            Final answer string
        """
        if not chunks:
            return "No information found to answer this query."

        answer_parts = [chunk.text[:200] for chunk in chunks]
        answer = " ".join(answer_parts)

        if len(answer) > 500:
            answer = answer[:500] + "..."

        return answer

    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """
        Parse LLM response into reasoning and answer.

        Args:
            response_text: Full response from LLM

        Returns:
            Tuple of (reasoning, answer)
        """
        # Look for REASONING: and ANSWER: sections
        if "REASONING:" in response_text and "ANSWER:" in response_text:
            parts = response_text.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
            return reasoning, answer
        else:
            # Fallback: treat entire response as answer
            return "", response_text
