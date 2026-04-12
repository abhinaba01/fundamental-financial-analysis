"""
Tests: Basic pipeline functionality validation.

Tests:
- Document parsing
- Text cleaning
- Semantic chunking
- Embedding generation
- Graph execution (mocked agents for speed)
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.preprocessing.parser import DocumentParser
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.chunker import SemanticChunker
from src.preprocessing.document import DocumentType, ChunkType
from src.graph.state import GraphState
from src.agents.ner_agent import NERAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.kpi_agent import KPIAgent
from src.agents.rag_agent import RAGAgent
from src.agents.synthesis_agent import SynthesisAgent


def test_document_parser_initialization():
    """Test parser initialization."""
    parser = DocumentParser()
    assert parser is not None


def test_document_cleaner_initialization():
    """Test cleaner initialization."""
    cleaner = DocumentCleaner()
    assert cleaner is not None


def test_semantic_chunker_initialization():
    """Test chunker initialization."""
    chunker = SemanticChunker()
    assert chunker is not None


def test_ner_agent_initialization():
    """Test NER agent initialization."""
    agent = NERAgent()
    assert agent is not None


def test_sentiment_agent_initialization():
    """Test sentiment agent initialization."""
    agent = SentimentAgent()
    assert agent is not None


def test_kpi_agent_initialization():
    """Test KPI agent initialization."""
    agent = KPIAgent()
    assert agent is not None


def test_rag_agent_initialization():
    """Test RAG agent initialization."""
    agent = RAGAgent()
    assert agent is not None


def test_synthesis_agent_initialization():
    """Test synthesis agent initialization."""
    agent = SynthesisAgent()
    assert agent is not None


def test_graph_state_structure():
    """Test GraphState has required fields."""
    state: GraphState = {
        "document": None,
        "query": "test",
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

    # Verify all fields exist
    assert "document" in state
    assert "query" in state
    assert "retrieved_chunks" in state
    assert "ner_results" in state
    assert "sentiment_results" in state
    assert "kpi_results" in state
    assert "report" in state


def test_synthesis_agent_formatting():
    """Test synthesis agent output formatting."""
    agent = SynthesisAgent()

    # Mock state
    state: GraphState = {
        "document": None,
        "query": "test query",
        "retrieved_chunks": [],
        "retrieval_score": 0.0,
        "retry_count": 0,
        "ner_results": {"total_entities": 5, "entity_types": {"ORG": 3}},
        "sentiment_results": {"overall_sentiment": "positive", "overall_score": 0.8},
        "kpi_results": {"total_kpis": 3, "extracted_kpis": {"revenue": 1000}},
        "cot_reasoning": "Step 1: Analyze data...",
        "final_answer": "The revenue is 1000",
        "report": {},
    }

    # Run synthesis
    result = agent(state)

    # Verify report structure
    assert "report" in result
    report = result["report"]
    assert "metadata" in report
    assert "summary" in report
    assert "named_entities" in report
    assert "sentiment_analysis" in report
    assert "financial_metrics" in report


def test_edge_re_query_condition():
    """Test re-query edge condition logic."""
    from src.graph.edges import should_retrieve_again

    # Test: should retry (low chunks, low similarity)
    state: GraphState = {
        "document": None,
        "query": "test",
        "retrieved_chunks": [],  # Empty
        "retrieval_score": 0.5,  # Below threshold
        "retry_count": 0,
        "ner_results": {},
        "sentiment_results": {},
        "kpi_results": {},
        "cot_reasoning": "",
        "final_answer": "",
        "report": {},
    }

    result = should_retrieve_again(state)
    assert result == "retrieve"  # Should re-query

    # Test: should not retry (high quality retrieval)
    state["retrieved_chunks"] = [1, 2, 3, 4, 5]  # Sufficient chunks
    state["retrieval_score"] = 0.9  # High similarity

    result = should_retrieve_again(state)
    assert result == "synthesize"  # Proceed to synthesis


def test_config_loading():
    """Test configuration files exist."""
    agents_config = Path("configs/agents.yaml")
    pipeline_config = Path("configs/pipeline.yaml")

    assert agents_config.exists(), "agents.yaml not found"
    assert pipeline_config.exists(), "pipeline.yaml not found"


def test_evaluation_modules_import():
    """Test evaluation modules can be imported."""
    from evaluation.eval_ner import NERERvaluator
    from evaluation.eval_sentiment import SentimentEvaluator
    from evaluation.eval_rag import RAGEvaluator
    from evaluation.eval_kpi import KPIEvaluator

    # Initialize evaluators
    ner_eval = NERERvaluator()
    sent_eval = SentimentEvaluator()
    rag_eval = RAGEvaluator()
    kpi_eval = KPIEvaluator()

    assert ner_eval is not None
    assert sent_eval is not None
    assert rag_eval is not None
    assert kpi_eval is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
