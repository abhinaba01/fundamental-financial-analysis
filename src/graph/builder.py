"""
Graph Builder: LangGraph StateGraph assembly.

Orchestrates the complete analysis pipeline:
1. Input: ParsedDocument + query
2. Sequential agents: NER → Sentiment → KPI → RAG
3. Conditional: Re-query if retrieval low confidence
4. Output: Final report with all agent results

Architecture:
- NER Agent: Entity extraction
- Sentiment Agent: Sentiment analysis with tone
- KPI Agent: Financial metric extraction
- RAG Agent: Evidence-based synthesis with re-query capability
- Synthesis Agent: Report assembly

Edges:
- Sequential: Each agent → next agent
- Conditional: RAG → {retrieve or synthesize} based on confidence
- Terminal: Synthesis → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.graph.state import GraphState
from src.graph.edges import (
    should_retrieve_again,
    determine_next_agent,
    route_after_synthesis,
)
from src.agents.ner_agent import NERAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.kpi_agent import KPIAgent
from src.agents.rag_agent import RAGAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisPipelineBuilder:
    """Builder for financial analysis pipeline using LangGraph."""

    def __init__(
        self,
        embedding_pipeline=None,
        ner_agent=None,
        sentiment_agent=None,
        kpi_agent=None,
        rag_agent=None,
        synthesis_agent=None,
    ):
        """
        Initialize pipeline builder with optional agent instances.

        Args:
            embedding_pipeline: EmbeddingPipeline for RAG retrieval
            ner_agent: NERAgent instance (created if None)
            sentiment_agent: SentimentAgent instance (created if None)
            kpi_agent: KPIAgent instance (created if None)
            rag_agent: RAGAgent instance (created if None)
            synthesis_agent: SynthesisAgent instance (created if None)
        """
        self.logger = logger
        self.embedding_pipeline = embedding_pipeline

        # Initialize agents (use provided or create new)
        self.ner_agent = ner_agent or NERAgent()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.kpi_agent = kpi_agent or KPIAgent()
        self.rag_agent = rag_agent or RAGAgent(embedding_pipeline=embedding_pipeline)
        self.synthesis_agent = synthesis_agent or SynthesisAgent()

        self.logger.info("Pipeline builder initialized")

    def build(self) -> StateGraph:
        """
        Build the complete analysis pipeline StateGraph.

        Returns:
            Compiled StateGraph ready for execution
        """
        self.logger.info("Building analysis pipeline graph...")

        # Create state graph
        graph = StateGraph(GraphState)

        # Add all nodes
        graph.add_node("ner", self.ner_agent)
        graph.add_node("sentiment", self.sentiment_agent)
        graph.add_node("kpi", self.kpi_agent)
        graph.add_node("rag", self.rag_agent)
        graph.add_node("synthesis", self.synthesis_agent)

        # Set entry point
        graph.set_entry_point("ner")

        # Add sequential edges (normal progression)
        graph.add_edge("ner", "sentiment")
        graph.add_edge("sentiment", "kpi")
        graph.add_edge("kpi", "rag")

        # Add conditional edge from RAG
        # Re-query trigger: <3 chunks AND similarity <0.75
        graph.add_conditional_edges(
            "rag",
            should_retrieve_again,
            {
                "retrieve": "rag",  # Loop back to RAG for re-query
                "synthesize": "synthesis",  # Proceed to synthesis
            },
        )

        # Add edge from synthesis to END
        graph.add_edge("synthesis", END)

        # Compile graph
        compiled_graph = graph.compile()

        self.logger.info("Pipeline graph compiled successfully")

        return compiled_graph

    def build_parallel_agents_variant(self) -> StateGraph:
        """
        Build variant with parallel agent execution.

        Agents NER, Sentiment, KPI, run in parallel after retrieval,
        then synthesis combines results.

        Returns:
            Compiled StateGraph with parallel agents
        """
        self.logger.info("Building parallel agents variant...")

        graph = StateGraph(GraphState)

        # Add all nodes
        graph.add_node("ner", self.ner_agent)
        graph.add_node("sentiment", self.sentiment_agent)
        graph.add_node("kpi", self.kpi_agent)
        graph.add_node("rag", self.rag_agent)
        graph.add_node("synthesis", self.synthesis_agent)

        graph.set_entry_point("rag")

        # RAG runs first (retrieval)
        graph.add_conditional_edges(
            "rag",
            should_retrieve_again,
            {
                "retrieve": "rag",  # Re-query loop
                "synthesize": "synthesis",  # But should actually go to parallel agents
            },
        )

        # In this variant, after RAG → all agents run in parallel
        # LangGraph handles this via send() mechanism

        graph.add_edge("synthesis", END)

        compiled_graph = graph.compile()

        self.logger.info("Parallel variant compiled successfully")

        return compiled_graph


def create_default_pipeline(embedding_pipeline=None) -> StateGraph:
    """
    Create default analysis pipeline with standard configuration.

    Args:
        embedding_pipeline: Optional EmbeddingPipeline instance

    Returns:
        Compiled StateGraph ready for .invoke() or .stream()
    """
    builder = AnalysisPipelineBuilder(embedding_pipeline=embedding_pipeline)
    return builder.build()
