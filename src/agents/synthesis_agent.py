"""
Synthesis Agent: Combine all agent outputs into a structured report.

Combines outputs from:
- NER Agent: Named entities (companies, people, regulations)
- Sentiment Agent: Document-level and chunk-level sentiment
- KPI Agent: Extracted financial metrics and calculations
- RAG Agent: Evidence-based answer with chain-of-thought

Produces: GraphState.report with structured analysis suitable for downstream consumption

Input: GraphState with ner_results, sentiment_results, kpi_results, cot_reasoning, final_answer
Output: GraphState.report as structured dict with all findings
"""

from __future__ import annotations

from typing import Any
from datetime import datetime

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesisAgent:
    """Synthesis agent for combining all analysis results into structured report."""

    def __init__(self):
        """Initialize synthesis agent."""
        self.logger = logger

    def __call__(self, state: GraphState) -> GraphState:
        """
        Synthesize all agent outputs into final report.

        Args:
            state: GraphState with all agent results

        Returns:
            Updated GraphState with report dict populated
        """
        self.logger.info("Synthesizing analysis results into final report...")

        report = self._build_report(state)
        state["report"] = report

        self.logger.info("Report synthesis complete")

        return state

    def _build_report(self, state: GraphState) -> dict[str, Any]:
        """
        Build comprehensive report from all agent outputs.

        Args:
            state: Complete GraphState

        Returns:
            Structured report dictionary
        """
        document = state.get("document")
        query = state.get("query", "")
        ner_results = state.get("ner_results", {})
        sentiment_results = state.get("sentiment_results", {})
        kpi_results = state.get("kpi_results", {})
        cot_reasoning = state.get("cot_reasoning", "")
        final_answer = state.get("final_answer", "")
        retrieved_chunks = state.get("retrieved_chunks", [])
        retrieval_score = state.get("retrieval_score", 0.0)
        retry_count = state.get("retry_count", 0)

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "document_id": getattr(document, "doc_id", "unknown") if document else None,
                "ticker": getattr(document, "ticker", "unknown") if document else None,
                "doc_type": getattr(document, "doc_type", "unknown") if document else None,
                "fiscal_period": getattr(document, "fiscal_period", "unknown") if document else None,
                "query": query,
            },
            "retrieval": {
                "chunks_retrieved": len(retrieved_chunks),
                "average_similarity": retrieval_score,
                "retries_performed": retry_count,
                "chunks": self._format_chunks(retrieved_chunks),
            },
            "named_entities": self._format_ner_results(ner_results),
            "sentiment_analysis": self._format_sentiment_results(sentiment_results),
            "financial_metrics": self._format_kpi_results(kpi_results),
            "reasoning": {
                "chain_of_thought": cot_reasoning,
                "final_answer": final_answer,
            },
            "summary": self._generate_summary(
                ner_results, sentiment_results, kpi_results, final_answer
            ),
        }

        return report

    def _format_chunks(self, chunks) -> list[dict[str, Any]]:
        """
        Format retrieved chunks for report.

        Args:
            chunks: Retrieved DocumentChunk objects

        Returns:
            List of formatted chunk dicts
        """
        formatted = []

        for chunk in chunks:
            formatted.append(
                {
                    "text": chunk.text[:200] + "..."
                    if len(chunk.text) > 200
                    else chunk.text,
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type.value
                    if hasattr(chunk.chunk_type, "value")
                    else str(chunk.chunk_type),
                    "page_number": chunk.metadata.get("page_number", -1),
                    "similarity": chunk.metadata.get("similarity", 0.0),
                }
            )

        return formatted

    def _format_ner_results(self, ner_results: dict[str, Any]) -> dict[str, Any]:
        """
        Format NER results for report.

        Args:
            ner_results: Raw NER results from agent

        Returns:
            Formatted NER section
        """
        if not ner_results:
            return {"total_entities": 0, "entity_types": {}, "sample_entities": []}

        doc_entities = ner_results.get("document_entities", [])
        entity_types = ner_results.get("entity_types", {})

        return {
            "total_entities": ner_results.get("total_entities", 0),
            "entity_types": entity_types,
            "sample_entities": doc_entities[:10],
            "unique_entity_types": len(entity_types),
        }

    def _format_sentiment_results(self, sentiment_results: dict[str, Any]) -> dict[str, Any]:
        """
        Format sentiment results for report.

        Args:
            sentiment_results: Raw sentiment results from agent

        Returns:
            Formatted sentiment section
        """
        if not sentiment_results:
            return {
                "overall_sentiment": "unknown",
                "confidence": 0.0,
                "distribution": {},
            }

        return {
            "overall_sentiment": sentiment_results.get("overall_sentiment", "unknown"),
            "confidence_score": sentiment_results.get("overall_score", 0.0),
            "sentiment_distribution": sentiment_results.get("sentiment_distribution", {}),
            "tone_analysis": sentiment_results.get("tone_analysis", {}),
            "is_positive": sentiment_results.get("is_positive", False),
        }

    def _format_kpi_results(self, kpi_results: dict[str, Any]) -> dict[str, Any]:
        """
        Format KPI results for report.

        Args:
            kpi_results: Raw KPI results from agent

        Returns:
            Formatted KPI section
        """
        if not kpi_results:
            return {
                "total_kpis": 0,
                "extracted_kpis": {},
                "calculated_kpis": {},
            }

        return {
            "total_kpis": kpi_results.get("total_kpis", 0),
            "extracted_kpis": kpi_results.get("extracted_kpis", {}),
            "calculated_kpis": kpi_results.get("calculated_kpis", {}),
            "kpi_count": len(kpi_results.get("extracted_kpis", {})),
        }

    def _generate_summary(
        self,
        ner_results: dict[str, Any],
        sentiment_results: dict[str, Any],
        kpi_results: dict[str, Any],
        final_answer: str,
    ) -> str:
        """
        Generate brief text summary of analysis.

        Args:
            ner_results: NER analysis results
            sentiment_results: Sentiment analysis results
            kpi_results: KPI analysis results
            final_answer: Final RAG answer

        Returns:
            Summary string
        """
        summary_parts = []

        # Entity summary
        if ner_results.get("total_entities", 0) > 0:
            summary_parts.append(
                f"Identified {ner_results.get('total_entities', 0)} entities "
                f"across {len(ner_results.get('entity_types', {}))} categories."
            )

        # Sentiment summary
        if sentiment_results:
            sentiment = sentiment_results.get("overall_sentiment", "neutral")
            score = sentiment_results.get("overall_score", 0.0)
            summary_parts.append(
                f"Document has {sentiment} sentiment (score: {score:.3f})."
            )

        # KPI summary
        if kpi_results.get("total_kpis", 0) > 0:
            kpi_count = len(kpi_results.get("extracted_kpis", {}))
            calc_count = len(kpi_results.get("calculated_kpis", {}))
            summary_parts.append(
                f"Extracted {kpi_count} KPIs with {calc_count} derived calculations."
            )

        # Answer summary
        if final_answer:
            answer_preview = final_answer[:100]
            if len(final_answer) > 100:
                answer_preview += "..."
            summary_parts.append(f"Answer: {answer_preview}")

        return " ".join(summary_parts) if summary_parts else "Analysis incomplete."

    def _build_kpi_section(
        self, kpi_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build KPI section.

        Args:
            kpi_results: KPI extraction results

        Returns:
            KPI section dictionary
        """
        extracted = kpi_results.get("extracted_kpis", {})
        calculated = kpi_results.get("calculated_kpis", {})

        return {
            "total_kpis": kpi_results.get("total_kpis", 0),
            "extracted_metrics": extracted,
            "calculated_metrics": calculated,
        }

    def _build_rag_section(
        self,
        retrieved_chunks,
        cot_reasoning: str,
        final_answer: str,
    ) -> dict[str, Any]:
        """
        Build RAG/Q&A section.

        Args:
            retrieved_chunks: Retrieved document chunks
            cot_reasoning: Chain-of-thought reasoning
            final_answer: Generated answer

        Returns:
            RAG section dictionary
        """
        chunk_summary = [
            {
                "id": chunk.chunk_id,
                "type": chunk.chunk_type.value,
                "similarity": chunk.metadata.get("similarity", 0.0),
                "preview": chunk.text[:200],
            }
            for chunk in retrieved_chunks
        ]

        return {
            "chunks_retrieved": len(retrieved_chunks),
            "chunk_summary": chunk_summary,
            "reasoning": cot_reasoning,
            "answer": final_answer,
        }

    def _build_findings(
        self,
        sentiment_results: dict[str, Any],
        ner_results: dict[str, Any],
        kpi_results: dict[str, Any],
    ) -> list[str]:
        """
        Build key findings list.

        Args:
            sentiment_results: Sentiment analysis results
            ner_results: NER results
            kpi_results: KPI results

        Returns:
            List of finding strings
        """
        findings = []

        # Sentiment findings
        sentiment = sentiment_results.get("overall_sentiment", "")
        if sentiment == "positive":
            findings.append("Document exhibits a positive sentiment overall")
        elif sentiment == "negative":
            findings.append("Document exhibits a negative sentiment overall")
        else:
            findings.append("Document exhibits a neutral sentiment overall")

        # Entity findings
        entity_types = ner_results.get("entity_types", {})
        if entity_types:
            top_type = max(entity_types.items(), key=lambda x: x[1])
            findings.append(
                f"Most common entity type: {top_type[0]} "
                f"({top_type[1]} occurrences)"
            )

        # KPI findings
        extracted = kpi_results.get("extracted_kpis", {})
        if extracted:
            findings.append(f"Identified {len(extracted)} types of KPIs in document")

        # Tone findings
        tone_prog = sentiment_results.get("tone_analysis", {}).get("tone_progression", 0)
        if tone_prog > 0:
            findings.append("Management tone becomes more positive from opening to closing")
        elif tone_prog < 0:
            findings.append("Management tone becomes more negative from opening to closing")

        return findings if findings else ["Analysis complete with no major findings"]
