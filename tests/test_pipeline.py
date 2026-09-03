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

import inspect

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
        "retrieval_score": 0.2,  # Below threshold
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


def test_kpi_agent_extracts_realistic_phrasing():
    """Regression: KPI_KEYWORDS values are themselves capturing groups, and
    wrapping them in another () shifted every group index by one, so
    match.group(2) was the keyword text (not the number) and float()
    silently failed for every sample. Also covers the "revenue of $X"
    connector case, which the original regex could not match at all."""
    agent = KPIAgent()
    text = (
        "Apple Inc. reported revenue of $119.60B for Q1 2024. "
        "The company's gross margin was 38.2%. "
        "Net income reached $24.20B, up 10% year-over-year."
    )

    kpis = agent._extract_kpi_patterns(text)

    assert kpis["revenue"][0]["value"] == 119.6
    assert kpis["revenue"][0]["unit"] == "B"
    assert kpis["gross_margin"][0]["value"] == 38.2
    assert kpis["net_income"][0]["value"] == 24.2


def test_kpi_agent_does_not_treat_percentage_margin_as_dollar_amount():
    """Regression: _perform_calculations divided the extracted gross_margin
    value by revenue as if it were a dollar gross-profit figure, but the
    regex captures gross margin as a percentage already - producing a
    nonsense "calculated" margin instead of skipping the calculation."""
    agent = KPIAgent()
    extracted = {
        "revenue": [{"name": "revenue", "value": 119.6, "unit": "B"}],
        "gross_margin": [{"name": "gross margin", "value": 38.2, "unit": "%"}],
    }

    calculated = agent._perform_calculations(extracted)

    assert calculated == {}


def test_cleaner_normalizes_smart_quotes():
    """Regression: _normalize_unicode's replace() calls used literal curly-quote
    characters that had been corrupted to mojibake by a prior bad save, so
    smart quotes silently passed through unchanged."""
    cleaner = DocumentCleaner()
    text = "He said “hello” and it’s fine."

    normalized = cleaner._normalize_unicode(text)

    assert all(ord(ch) < 128 for ch in normalized)
    assert '"' in normalized


def test_cleaner_financial_notation_preserves_word_boundary():
    """Regression: the billion/million/thousand patterns consumed their
    trailing boundary character (space/punctuation) instead of just checking
    for it, so "$1.2 billion for" became "$1.20Bfor" (words jammed together)."""
    cleaner = DocumentCleaner()

    assert cleaner._normalize_financial_notation(
        "revenue of $119.6 billion for Q1 2024"
    ) == "revenue of $119.60B for Q1 2024"

    assert cleaner._normalize_financial_notation(
        "Net income reached $24.2 billion, up 10%."
    ) == "Net income reached $24.20B, up 10%."


def test_synthesis_agent_formats_retrieved_chunks():
    """Regression: _format_chunks read chunk.section, but DocumentChunk has no
    such attribute (section lives in chunk.metadata) - this raised
    AttributeError as soon as any chunk was retrieved."""
    from src.preprocessing.document import DocumentChunk, ChunkType

    chunk = DocumentChunk(
        text="Revenue grew 8%.",
        chunk_type=ChunkType.MDA,
        metadata={"similarity": 0.91, "section": "mda"},
    )

    formatted = SynthesisAgent()._format_chunks([chunk])

    assert formatted[0]["section"] == "mda"
    assert formatted[0]["chunk_type"] == "mda"


def test_embedding_pipeline_accepts_device_argument():
    """Regression: src.main.run_analysis calls EmbeddingPipeline(device=...),
    which raised TypeError once the constructor's signature changed and
    dropped the device parameter."""
    from src.preprocessing.embedder import EmbeddingPipeline

    sig = inspect.signature(EmbeddingPipeline.__init__)
    assert "device" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
