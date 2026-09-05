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


def _routing_state(**overrides) -> GraphState:
    """Minimal state for exercising should_retrieve_again."""
    state = {
        "document": None,
        "query": "test",
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
    state.update(overrides)
    return state


def test_edge_re_query_condition():
    """Test re-query edge condition logic."""
    from src.graph.edges import should_retrieve_again

    # Weak on both axes -> retry.
    assert should_retrieve_again(_routing_state(retrieved_chunks=[], retrieval_score=0.2)) == "retrieve"

    # Strong on both axes -> proceed.
    strong = _routing_state(retrieved_chunks=[1, 2, 3, 4, 5], retrieval_score=0.9)
    assert should_retrieve_again(strong) == "generate"


def test_edge_retries_on_weak_similarity_alone():
    """Regression: plenty of chunks but a weak average must still retry.

    The condition was previously `low_chunks AND low_similarity`, so a
    retrieval that returned five barely-relevant chunks was treated as good
    evidence. Each signal is an independent failure mode, so either alone is
    grounds for re-querying.
    """
    from src.graph.edges import SIMILARITY_THRESHOLD, should_retrieve_again

    state = _routing_state(
        retrieved_chunks=[1, 2, 3, 4, 5],
        retrieval_score=SIMILARITY_THRESHOLD - 0.05,
    )

    assert should_retrieve_again(state) == "retrieve"


def test_edge_retries_on_few_chunks_alone():
    """One highly relevant chunk is thin evidence and must still retry."""
    from src.graph.edges import should_retrieve_again

    assert should_retrieve_again(_routing_state(retrieved_chunks=[1], retrieval_score=0.95)) == "retrieve"


def test_routing_threshold_must_exceed_retrieval_floor():
    """Regression: the routing threshold has to sit above the retrieval floor.

    _retrieve_chunks discards every chunk scoring below
    RETRIEVAL_SIMILARITY_FLOOR, so the average of the survivors is always at
    least that floor. When both values were 0.5, `retrieval_score <
    SIMILARITY_THRESHOLD` was unsatisfiable and the similarity half of the
    routing condition was dead code that could never fire.
    """
    from src.agents.rag_agent import RETRIEVAL_SIMILARITY_FLOOR
    from src.graph.edges import SIMILARITY_THRESHOLD

    assert SIMILARITY_THRESHOLD > RETRIEVAL_SIMILARITY_FLOOR


def test_edge_stops_retrying_at_budget():
    """The retry budget bounds the loop even while retrieval stays weak."""
    from src.graph.edges import MAX_RETRIES, should_retrieve_again

    exhausted = _routing_state(retrieved_chunks=[], retrieval_score=0.0, retry_count=MAX_RETRIES)

    assert should_retrieve_again(exhausted) == "generate"


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


def test_chunker_terminates_on_document_needing_multiple_chunks():
    """Regression: once the sliding-window loop's chunk_end_token_idx got
    clamped to len(tokens) (i.e. on the final chunk of any document long
    enough to need more than one), chunk_start_token_idx = chunk_end_token_idx
    - OVERLAP_TOKENS became a fixed value no longer dependent on the current
    chunk_start_token_idx. The old post-hoc safety check (comparing the new
    start index to len(tokens) - 1) could never be satisfied, so the loop
    re-processed the same final chunk forever - a genuine infinite loop on
    every real document over ~512 tokens, not merely a slow path."""
    import threading

    from src.preprocessing.document import ParsedDocument

    # ~8000 tokens of repeated sentences - comfortably more than the 512
    # token chunk size, so this must go through the sliding-window branch
    # and reach the tail-clamping condition that used to hang forever.
    long_text = "Apple Inc. reported quarterly revenue growth. " * 800
    doc = ParsedDocument(cleaned_text=long_text)

    chunker = SemanticChunker()
    result = {}

    def run():
        result["chunked"] = chunker.chunk(doc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive(), "chunker.chunk() did not terminate within 30s - infinite loop regression"
    assert len(result["chunked"].chunks) > 1


def test_parallel_agents_return_delta_not_full_state():
    """Regression: each parallel branch must return only the key it owns.

    LangGraph merges what concurrent branches return. A branch handing back the
    whole state would write every key in the same superstep as its siblings,
    which LangGraph rejects for keys with no reducer. This contract is what
    makes the fan-out legal, so it is pinned rather than assumed. The branch
    must also leave the shared input state untouched.
    """
    from src.preprocessing.document import ParsedDocument

    doc = ParsedDocument(cleaned_text="Total revenue was $383,285 million.")
    state = {"document": doc, "query": "q"}

    delta = KPIAgent()(state)

    assert set(delta) == {"kpi_results"}
    assert "kpi_results" not in state, "branch mutated state shared with its siblings"


def test_rag_retrieve_re_searches_on_retry_instead_of_reusing_chunks():
    """Regression: a retry has to actually re-query.

    Retrieval used to be guarded by `if not retrieved_chunks`, so when the
    graph looped back the node skipped retrieval entirely and regenerated from
    the same evidence - the loop could never change its own outcome.
    """
    from src.agents.rag_agent import RAGAgent

    searched = []

    class StubEmbedder:
        def retrieve(self, query, **kwargs):
            searched.append(query)
            return []

    agent = RAGAgent(embedding_pipeline=StubEmbedder())
    original = "What is the revenue?"

    agent.retrieve({
        "query": original,
        "document": None,
        "rag_attempts": 1,
        "retrieved_chunks": ["chunk from the previous attempt"],
    })

    assert len(searched) == 1, "retry reused previous chunks instead of re-searching"
    assert searched[0] != original, "retry re-ran the identical query"


def test_rag_reformulation_differs_per_attempt():
    """Each retry widens the search differently.

    A constant reformulation would make the second and third attempts re-run
    an identical vector search and retrieve identical chunks.
    """
    from src.agents.rag_agent import RAGAgent

    agent = RAGAgent(embedding_pipeline=None)
    original = "What are the primary risk factors?"

    first = agent._reformulate_query(original, 1)
    second = agent._reformulate_query(original, 2)

    assert first != second
    assert original not in (first, second)


def test_graph_fans_out_analysis_agents_and_generates_once():
    """Topology regression for the parallel build.

    Covers three properties at once: all three analysis agents run and their
    results merge (fan-out then fan-in), the re-query loop is bounded by the
    retry budget, and generation happens exactly once after the loop settles.
    That last one is why retrieval and generation are separate nodes - when
    they were one node, every retry attempt also paid for a gpt-4o call.

    Stub agents keep this a test of the wiring, not of the models.
    """
    from src.graph.builder import AnalysisPipelineBuilder
    from src.graph.edges import MAX_RETRIES

    calls = []

    class StubAgent:
        def __init__(self, key):
            self.key = key

        def __call__(self, state):
            calls.append(self.key)
            return {self.key: {"ran": True}}

    class StubRag:
        """Always reports weak retrieval, forcing the full retry budget."""

        def retrieve(self, state):
            attempts = state.get("rag_attempts", 0)
            calls.append(f"retrieve{attempts}")
            return {
                "rag_attempts": attempts + 1,
                "retry_count": attempts,
                "retrieved_chunks": [1],
                "retrieval_score": 0.55,
            }

        def generate(self, state):
            calls.append("generate")
            return {"cot_reasoning": "reasoning", "final_answer": "answer"}

    graph = AnalysisPipelineBuilder(
        ner_agent=StubAgent("ner_results"),
        sentiment_agent=StubAgent("sentiment_results"),
        kpi_agent=StubAgent("kpi_results"),
        rag_agent=StubRag(),
        synthesis_agent=StubAgent("report"),
    ).build()

    final = graph.invoke(_routing_state())

    # Every branch ran, and every branch's result survived the merge.
    for key in ("ner_results", "sentiment_results", "kpi_results", "report"):
        assert final[key] == {"ran": True}, f"{key} missing after fan-in"

    # The loop is bounded, and generation waited for it to finish.
    assert final["retry_count"] == MAX_RETRIES
    assert calls.count("generate") == 1, "generation ran per attempt instead of once"
    last_retrieve = max(i for i, c in enumerate(calls) if c.startswith("retrieve"))
    assert calls.index("generate") > last_retrieve


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
