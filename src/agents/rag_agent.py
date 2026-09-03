"""
RAG Agent: Retrieval-Augmented Generation with Chain-of-Thought.

Uses:
- BAAI/bge-large-en-v1.5 for retrieval
- gpt-4o for generation with chain-of-thought (falls back to extractive
  synthesis if OPENAI_API_KEY is not set or the API call fails)

Performs:
- Query-based document chunk retrieval
- Re-querying if retrieval confidence is low
- Chain-of-thought synthesis of answers
- Evidence citation

Input: GraphState with document, query, chunks, embeddings available
Output: GraphState with retrieved_chunks, cot_reasoning, final_answer populated
"""

from __future__ import annotations

import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration.
# BAAI/bge-large-en-v1.5 cosine similarities for genuinely relevant chunks
# typically land around 0.5-0.65, not near 1.0 - keep this in sync with
# src/graph/edges.py's SIMILARITY_THRESHOLD.
RETRIEVAL_TOP_K = 5
SIMILARITY_THRESHOLD = 0.5
MIN_CHUNKS_THRESHOLD = 3
MAX_RETRIES = 2
GENERATION_MODEL = "gpt-4o"
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

    def __call__(self, state: GraphState) -> GraphState:
        """
        Execute RAG on the query using document chunks.

        Args:
            state: GraphState with document, query, and retrieved_chunks

        Returns:
            Updated GraphState with cot_reasoning and final_answer populated
        """
        query = state.get("query")
        document = state.get("document")
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not query:
            self.logger.warning("No query in state. Skipping RAG.")
            return state

        self.logger.info(f"Running RAG for query: {query[:100]}...")

        # Retrieve chunks if not already provided
        if not retrieved_chunks and self.embedding_pipeline:
            retrieved_chunks, retrieval_score = self._retrieve_chunks(
                query, document
            )
            state["retrieved_chunks"] = retrieved_chunks
            state["retrieval_score"] = retrieval_score

            # Re-query if needed
            if len(retrieved_chunks) < MIN_CHUNKS_THRESHOLD:
                self.logger.info(
                    f"Low retrieval confidence ({len(retrieved_chunks)} chunks). Re-querying..."
                )
                retry_query = self._generate_retry_query(query)
                retrieved_chunks, retrieval_score = self._retrieve_chunks(
                    retry_query, document
                )
                state["retrieved_chunks"] = retrieved_chunks
                state["retrieval_score"] = retrieval_score
                state["retry_count"] = state.get("retry_count", 0) + 1

        # Generate answer with chain-of-thought
        if retrieved_chunks:
            cot_reasoning, final_answer = self._generate_answer(query, retrieved_chunks)
            state["cot_reasoning"] = cot_reasoning
            state["final_answer"] = final_answer
        else:
            self.logger.warning("No chunks retrieved. Unable to generate answer.")
            state["cot_reasoning"] = "No relevant chunks found in document."
            state["final_answer"] = "Unable to answer based on available documents."

        self.logger.info("RAG synthesis complete")

        return state

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
                similarity_threshold=SIMILARITY_THRESHOLD,
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

    def _generate_retry_query(self, original_query: str) -> str:
        """
        Generate a reformulated query for re-retrieval.

        Args:
            original_query: Original query text

        Returns:
            Reformulated query
        """
        return f"Provide detailed information about: {original_query}"

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
