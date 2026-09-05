"""
Graph Builder: LangGraph StateGraph assembly.

Topology:

    START ─┬─→ ner ───────┐
           ├─→ sentiment ─┼─→ retrieve ──(conditional)──┐
           └─→ kpi ───────┘      ↑                      │
                                 └──── re-query ────────┤
                                                        ↓
                                        generate → synthesis → END

Two things are worth noting about this shape.

**The three analysis agents fan out.** NER, sentiment and KPI extraction read
the same document and write disjoint state keys, so nothing forces them into a
sequence. They are independent branches that fan back in at `retrieve`, which
LangGraph runs only once all three have finished. Each returns a state delta
rather than the whole state - concurrent branches all returning full state
would mean several writes to every key in one superstep, which LangGraph
rejects without a reducer.

**Retrieval and generation are separate nodes.** The re-query loop wraps
retrieval alone, so a low-confidence retry re-searches with a reformulated
query and generation runs once, after the loop settles - rather than paying
for a gpt-4o call on every attempt. Retry policy lives in edges.py; the RAG
agent only counts attempts.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.state import GraphState
from src.graph.edges import should_retrieve_again
from src.agents.ner_agent import NERAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.kpi_agent import KPIAgent
from src.agents.rag_agent import RAGAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Nodes that run concurrently as independent branches off START.
PARALLEL_ANALYSIS_NODES = ("ner", "sentiment", "kpi")


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
        device: str = "cpu",
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
            device: Device used for any agent created here ('cpu' or 'cuda') -
                ignored for agents passed in explicitly
        """
        self.logger = logger
        self.embedding_pipeline = embedding_pipeline

        # Initialize agents (use provided or create new)
        self.ner_agent = ner_agent or NERAgent(device=device)
        self.sentiment_agent = sentiment_agent or SentimentAgent(device=device)
        self.kpi_agent = kpi_agent or KPIAgent()
        self.rag_agent = rag_agent or RAGAgent(embedding_pipeline=embedding_pipeline)
        self.synthesis_agent = synthesis_agent or SynthesisAgent()

        self.logger.info("Pipeline builder initialized")

    def build(self) -> StateGraph:
        """
        Build the complete analysis pipeline StateGraph.

        Returns:
            Compiled StateGraph ready for .invoke() or .stream()
        """
        self.logger.info("Building analysis pipeline graph...")

        graph = StateGraph(GraphState)

        graph.add_node("ner", self.ner_agent)
        graph.add_node("sentiment", self.sentiment_agent)
        graph.add_node("kpi", self.kpi_agent)
        graph.add_node("retrieve", self.rag_agent.retrieve)
        graph.add_node("generate", self.rag_agent.generate)
        graph.add_node("synthesis", self.synthesis_agent)

        # Fan out: the three analysis agents are independent, so they start
        # together rather than in an arbitrary sequence.
        for node in PARALLEL_ANALYSIS_NODES:
            graph.add_edge(START, node)

        # Fan in: retrieve waits for all three branches to finish.
        for node in PARALLEL_ANALYSIS_NODES:
            graph.add_edge(node, "retrieve")

        # Re-query loop around retrieval only.
        graph.add_conditional_edges(
            "retrieve",
            should_retrieve_again,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            },
        )

        graph.add_edge("generate", "synthesis")
        graph.add_edge("synthesis", END)

        compiled_graph = graph.compile()

        self.logger.info("Pipeline graph compiled successfully")

        return compiled_graph


def create_default_pipeline(embedding_pipeline=None, device: str = "cpu") -> StateGraph:
    """
    Create default analysis pipeline with standard configuration.

    Args:
        embedding_pipeline: Optional EmbeddingPipeline instance
        device: Device used for the NER/sentiment models ('cpu' or 'cuda')

    Returns:
        Compiled StateGraph ready for .invoke() or .stream()
    """
    builder = AnalysisPipelineBuilder(embedding_pipeline=embedding_pipeline, device=device)
    return builder.build()
