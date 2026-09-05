"""
Evaluation: Retrieval-Augmented Generation (RAG) Performance.

Evaluates RAGAgent performance using:
- FinanceBench dataset (financial QA pairs with evidence)
- Metrics: Exact Match (EM), ROUGE-L, BLEU, MRR, NDCG
- Retrieval: Top-1, Top-5 Hit Rate
- Generation: BERT-Score, semantic similarity

Benchmarks:
- RetrieverGPT: ~75% EM on FinanceBench
- Top-5 Retrieval: ~85% Hit Rate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation._cli import (
    build_parser,
    load_samples,
    mean,
    print_metrics,
    to_dict,
    write_output,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGMetrics:
    """RAG evaluation metrics."""

    exact_match: float
    rouge_l: float
    bleu_score: float
    mrr: float
    ndcg: float
    retrieval_hit_rate_top1: float
    retrieval_hit_rate_top5: float
    bert_score: float
    total_questions: int


class RAGEvaluator:
    """Evaluator for RAG agent performance."""

    def __init__(self):
        """Initialize RAG evaluator."""
        self.logger = logger

    def evaluate_retrieval(
        self,
        retrieved_chunks: list[dict[str, Any]],
        gold_chunks: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Evaluate retrieval quality.

        Args:
            retrieved_chunks: Retrieved document chunks
            gold_chunks: Ground-truth relevant chunks

        Returns:
            Retrieval metrics (hit@1, hit@5, MRR, NDCG)
        """
        self.logger.info(f"Evaluating retrieval: {len(retrieved_chunks)} retrieved vs {len(gold_chunks)} gold")

        # Calculate hit rates
        hit_top1 = 1.0 if any(chunk in gold_chunks for chunk in retrieved_chunks[:1]) else 0.0
        hit_top5 = 1.0 if any(chunk in gold_chunks for chunk in retrieved_chunks[:5]) else 0.0

        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks[:5], 1):
            if chunk in gold_chunks:
                mrr = 1.0 / i
                break

        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        # Simplified: assume binary relevance
        dcg = sum(1.0 / (i + 1) for i, chunk in enumerate(retrieved_chunks) if chunk in gold_chunks)
        ideal_dcg = sum(1.0 / (i + 1) for i in range(len(gold_chunks)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        metrics = {
            "hit_rate_top1": hit_top1,
            "hit_rate_top5": hit_top5,
            "mrr": mrr,
            "ndcg": ndcg,
        }

        self.logger.info(f"Retrieval Metrics - Hit@1: {hit_top1:.3f}, Hit@5: {hit_top5:.3f}, MRR: {mrr:.3f}")

        return metrics

    def evaluate_generation(
        self,
        generated_answer: str,
        reference_answer: str,
    ) -> dict[str, float]:
        """
        Evaluate generated answer quality.

        Args:
            generated_answer: Generated text
            reference_answer: Ground-truth answer

        Returns:
            Generation metrics (EM, ROUGE-L, BLEU)
        """
        self.logger.info("Evaluating generation quality...")

        # Exact Match
        exact_match = 1.0 if generated_answer.lower().strip() == reference_answer.lower().strip() else 0.0

        # Simple ROUGE-L approximation
        rouge_l = self._calculate_simple_rouge_l(generated_answer, reference_answer)

        # Simple BLEU approximation
        bleu = self._calculate_simple_bleu(generated_answer, reference_answer)

        metrics = {
            "exact_match": exact_match,
            "rouge_l": rouge_l,
            "bleu": bleu,
        }

        self.logger.info(f"Generation Metrics - EM: {exact_match:.3f}, ROUGE-L: {rouge_l:.3f}, BLEU: {bleu:.3f}")

        return metrics

    def _calculate_simple_rouge_l(self, generated: str, reference: str) -> float:
        """
        Calculate simplified ROUGE-L (LCS-based).

        Args:
            generated: Generated text
            reference: Reference text

        Returns:
            ROUGE-L score (0-1)
        """
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 0.0

        # Count common consecutive words
        lcs_len = sum(
            1 for i, word in enumerate(gen_words)
            if i < len(ref_words) and word == ref_words[i]
        )

        recall = lcs_len / len(ref_words)
        precision = lcs_len / len(gen_words) if gen_words else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    def _calculate_simple_bleu(self, generated: str, reference: str) -> float:
        """
        Calculate simplified BLEU (bigram-based).

        Args:
            generated: Generated text
            reference: Reference text

        Returns:
            BLEU score (0-1)
        """
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 0.0

        # Count bigram matches
        gen_bigrams = set(zip(gen_words, gen_words[1:]))
        ref_bigrams = set(zip(ref_words, ref_words[1:]))

        if not ref_bigrams:
            return 0.0

        matches = len(gen_bigrams & ref_bigrams)
        total = len(ref_bigrams)

        return matches / total if total > 0 else 0.0

    def evaluate_end_to_end(
        self,
        retrieved_chunks: list[dict[str, Any]],
        generated_answer: str,
        gold_chunks: list[dict[str, Any]],
        reference_answer: str,
    ) -> RAGMetrics:
        """
        Evaluate complete RAG pipeline.

        Args:
            retrieved_chunks: Retrieved chunks
            generated_answer: Generated answer
            gold_chunks: Ground-truth chunks
            reference_answer: Ground-truth answer

        Returns:
            Combined RAG metrics
        """
        self.logger.info("Running end-to-end RAG evaluation...")

        retrieval_metrics = self.evaluate_retrieval(retrieved_chunks, gold_chunks)
        generation_metrics = self.evaluate_generation(generated_answer, reference_answer)

        metrics = RAGMetrics(
            exact_match=generation_metrics.get("exact_match", 0.0),
            rouge_l=generation_metrics.get("rouge_l", 0.0),
            bleu_score=generation_metrics.get("bleu", 0.0),
            mrr=retrieval_metrics.get("mrr", 0.0),
            ndcg=retrieval_metrics.get("ndcg", 0.0),
            retrieval_hit_rate_top1=retrieval_metrics.get("hit_rate_top1", 0.0),
            retrieval_hit_rate_top5=retrieval_metrics.get("hit_rate_top5", 0.0),
            bert_score=0.0,  # Would need transformers library
            total_questions=1,
        )

        self.logger.info(f"End-to-End RAG Metrics - EM: {metrics.exact_match:.3f}, MRR: {metrics.mrr:.3f}")

        return metrics

    def benchmark_against_financebench(self) -> dict[str, float]:
        """
        Benchmark against FinanceBench expectations.

        Expected performance (RetrieverGPT):
        - Exact Match: ~75%
        - ROUGE-L: ~0.82
        - Top-1 Hit: ~62%
        - Top-5 Hit: ~85%
        - MRR: ~0.71

        Returns:
            Benchmark metrics
        """
        benchmarks = {
            "exact_match": 0.75,
            "rouge_l": 0.82,
            "top1_hit": 0.62,
            "top5_hit": 0.85,
            "mrr": 0.71,
        }

        self.logger.info("Using FinanceBench benchmarks...")
        return benchmarks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Test-set format (see evaluation/_cli.py for the accepted container shapes).
# Each sample:
#
#     {
#       "question": "What were FY2023 net sales?",
#       "answer": "Total net sales were $383.3 billion.",   # reference answer
#       "gold_chunks": ["...evidence text..."],             # ground-truth evidence
#       "retrieved_chunks": ["...evidence text..."],        # optional
#       "generated_answer": "..."                           # optional
#     }
#
# Chunks may be plain strings or objects with "chunk_id" and/or "text".
# "query" aliases "question"; "reference_answer" aliases "answer"; "gold" and
# "evidence" alias "gold_chunks".


def _chunk_identity(chunk: Any) -> dict[str, str]:
    """
    Reduce a chunk to a single-key identity dict.

    The evaluator matches chunks with ``chunk in gold_chunks``, i.e. full dict
    equality, so a retrieved chunk carrying extra metadata (similarity scores,
    page numbers) would never match its gold counterpart. Collapsing both sides
    to one stable key makes that comparison behave as identity matching.

    Args:
        chunk: A string, or an object with "chunk_id" and/or "text"

    Returns:
        Single-key dict identifying the chunk
    """
    if isinstance(chunk, str):
        return {"id": " ".join(chunk.split()).lower()}

    if isinstance(chunk, dict):
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if chunk_id:
            return {"id": str(chunk_id)}
        return {"id": " ".join(str(chunk.get("text", "")).split()).lower()}

    return {"id": " ".join(str(chunk).split()).lower()}


def _predict_with_agent(samples: list[dict[str, Any]]) -> None:
    """
    Populate each sample's retrieval and generation output by running RAGAgent.

    Retrieval runs against the persistent ChromaDB collection, so the corpus
    must already be indexed (e.g. via ``python -m src.main``). Building the
    EmbeddingPipeline needs OPENAI_API_KEY.

    Args:
        samples: Test-set samples, mutated in place
    """
    from src.agents.rag_agent import RAGAgent
    from src.preprocessing.embedder import EmbeddingPipeline

    try:
        embedder = EmbeddingPipeline()
    except (ImportError, ValueError) as exc:
        raise SystemExit(
            f"Cannot run the RAG agent: {exc}\n"
            "Retrieval needs an indexed vector store and OPENAI_API_KEY. "
            "Index a corpus first with `python -m src.main`, or score a test set "
            "that already carries 'retrieved_chunks' and 'generated_answer'."
        ) from exc

    agent = RAGAgent(embedding_pipeline=embedder)

    for index, sample in enumerate(samples):
        question = sample.get("question", sample.get("query", ""))

        if not question:
            logger.warning(f"Sample {index} has no 'question'; skipping retrieval")
            sample["retrieved_chunks"] = []
            sample["generated_answer"] = ""
            continue

        state = {
            "document": None,
            "query": question,
            "retrieved_chunks": [],
            "retrieval_score": 0.0,
            "retry_count": 0,
            "rag_attempts": 0,
            "ner_results": {},
            "sentiment_results": {},
            "kpi_results": {},
            "cot_reasoning": "",
            "final_answer": "",
            "report": {},
        }

        result = agent(state)

        sample["retrieved_chunks"] = [
            {"chunk_id": chunk.chunk_id, "text": chunk.text}
            for chunk in result.get("retrieved_chunks", [])
        ]
        sample["generated_answer"] = result.get("final_answer", "")

    logger.info(f"Generated RAG output for {len(samples)} samples")


def main() -> None:
    """CLI entry point for RAG evaluation."""
    parser = build_parser(
        description="Evaluate the RAG agent against a FinanceBench style test set",
        agent_help=(
            "Load RAGAgent and answer each sample's 'question' against the indexed "
            "vector store (needs OPENAI_API_KEY)"
        ),
    )
    args = parser.parse_args()

    samples = load_samples(args)

    if args.run_agent:
        _predict_with_agent(samples)

    evaluator = RAGEvaluator()

    per_sample: list[RAGMetrics] = []
    without_output = 0

    for index, sample in enumerate(samples):
        reference_answer = sample.get("answer", sample.get("reference_answer"))

        if reference_answer is None:
            raise SystemExit(
                f"Sample {index} has no 'answer' (or 'reference_answer') key. "
                "Every sample needs a ground-truth answer."
            )

        gold = sample.get("gold_chunks", sample.get("gold", sample.get("evidence", [])))
        retrieved = sample.get("retrieved_chunks")
        generated = sample.get("generated_answer")

        if retrieved is None and generated is None:
            without_output += 1

        per_sample.append(
            evaluator.evaluate_end_to_end(
                retrieved_chunks=[_chunk_identity(chunk) for chunk in (retrieved or [])],
                generated_answer=generated or "",
                gold_chunks=[_chunk_identity(chunk) for chunk in gold],
                reference_answer=str(reference_answer),
            )
        )

    if without_output:
        logger.warning(
            f"{without_output}/{len(samples)} samples carried neither 'retrieved_chunks' "
            "nor 'generated_answer' and scored zero. Pass --run-agent to generate them."
        )

    payload: dict[str, Any] = {
        "aggregation": "macro (mean across samples)",
        "samples_evaluated": len(samples),
        "exact_match": mean([m.exact_match for m in per_sample]),
        "rouge_l": mean([m.rouge_l for m in per_sample]),
        "bleu_score": mean([m.bleu_score for m in per_sample]),
        "mrr": mean([m.mrr for m in per_sample]),
        "ndcg": mean([m.ndcg for m in per_sample]),
        "retrieval_hit_rate_top1": mean([m.retrieval_hit_rate_top1 for m in per_sample]),
        "retrieval_hit_rate_top5": mean([m.retrieval_hit_rate_top5 for m in per_sample]),
        "unavailable_metrics": ["bert_score (not implemented in RAGEvaluator)"],
    }

    if args.benchmark:
        payload["financebench_benchmark"] = evaluator.benchmark_against_financebench()

    print_metrics("RAG EVALUATION (FinanceBench)", payload)
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
