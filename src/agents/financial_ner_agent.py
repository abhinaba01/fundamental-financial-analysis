"""
Financial NER Agent: XBRL financial-figure tagging via a fine-tuned model.

Fine-tunes nlpaueb/sec-bert-base on nlpaueb/finer-139 (FiNER-139): tags
numeric tokens in SEC filing text with 139 XBRL accounting concepts (e.g.
is this figure "Revenues" or "OperatingLeaseLiability"). This is NOT a
company/person/location task - see src/agents/ner_agent.py for that. The
base sec-bert-base variant was used deliberately, despite sibling models
(sec-bert-num, sec-bert-shape) existing specifically to make numeric tokens
easier to tag; see Train_FinBERT_NER_Colab.ipynb for the training run.

Until a fine-tuned model has actually been trained and pushed to the Hub
(see Train_FinBERT_NER_Colab.ipynb), this agent has nothing to load and
degrades to an empty result - there is no "simpler working alternative" to
fall back to the way SentimentAgent falls back from its tone model to its
primary model, so the correct behavior on load failure is empty output, not
a substitute model. KPIAgent only reads state["financial_entities"] for its
"revenue" KPI type - FiNER-139 has no tag for gross margin, operating
income, net income, EPS, EBITDA, ROA, or ROE, so those stay regex-only.

Input: GraphState with document, chunks populated
Output: GraphState with financial_entities populated (possibly empty)
"""

from __future__ import annotations

import re
from typing import Any

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Placeholder until Train_FinBERT_NER_Colab.ipynb has been run and the
# resulting model pushed to the Hub - update this to the real repo name
# (e.g. "<your-hf-username>/sec-bert-finer139") once that exists. Until
# then, loading this repo will fail and the agent degrades to empty output.
MODEL_NAME = "YOUR_HF_USERNAME/sec-bert-finer139"

_NUMERIC_CHARS = re.compile(r"[^\d.\-]")


class FinancialNERAgent:
    """Tags financial figures with XBRL accounting concepts."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        """
        Initialize the financial entity tagger.

        Args:
            model_name: HuggingFace repo of the fine-tuned FiNER-139 model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.logger = logger
        self.model_name = model_name
        self.pipeline = None

        if pipeline is None:
            self.logger.warning("transformers not installed. Financial entity tagging disabled.")
            return

        try:
            self.logger.info(f"Loading financial entity model: {model_name} on {device}")
            self.pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=device,
            )
            self.logger.info("Financial entity model loaded successfully")
        except Exception as e:
            self.logger.warning(
                f"Financial entity model '{model_name}' not available ({e}). "
                "This is expected until Train_FinBERT_NER_Colab.ipynb has been run "
                "and its output model pushed to the Hub. Financial entity tagging "
                "will produce empty results; KPI extraction falls back to regex."
            )
            self.pipeline = None

    def __call__(self, state: GraphState) -> GraphState:
        """
        Tag financial figures in the document's chunks with XBRL concepts.

        Args:
            state: GraphState with document populated

        Returns:
            Updated GraphState with financial_entities populated
        """
        document = state.get("document")

        if self.pipeline is None or not document:
            state["financial_entities"] = []
            return state

        chunk_texts = [chunk.text for chunk in document.chunks if chunk.text.strip()]

        if not chunk_texts:
            state["financial_entities"] = []
            return state

        self.logger.info(f"Running financial entity tagging on document: {document.doc_id}")

        per_chunk_entities = self.tag_texts(chunk_texts)
        entities = [entity for chunk_entities in per_chunk_entities for entity in chunk_entities]

        state["financial_entities"] = entities

        self.logger.info(f"Financial entity tagging complete: {len(entities)} tagged")

        return state

    def tag_texts(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """
        Tag a batch of texts with XBRL financial entities in one pipeline
        call. Public so evaluation/eval_finer.py can reuse it directly
        instead of duplicating pipeline-invocation logic.

        Args:
            texts: List of text strings to tag

        Returns:
            List (same order/length as `texts`) of entity-dict lists -
            empty lists throughout if the model isn't loaded or the batch
            is empty, never raises.
        """
        if self.pipeline is None or not texts:
            return [[] for _ in texts]

        try:
            # One batched call, not a per-text loop - matters a lot on GPU
            # and for documents/eval sets with many texts.
            batched_results = self.pipeline(texts)
        except Exception as e:
            self.logger.error(f"Error during financial entity tagging: {e}")
            return [[] for _ in texts]

        return [self._spans_to_entities(spans) for spans in batched_results]

    def _spans_to_entities(self, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert raw pipeline span dicts into this agent's entity shape."""
        entities = []
        for span in spans:
            text = span.get("word", "")
            entities.append({
                "text": text,
                "tag": span.get("entity_group", "unknown"),
                "value": self._parse_numeric_value(text),
                "score": float(span.get("score", 0.0)),
                "start": span.get("start"),
                "end": span.get("end"),
            })
        return entities

    def _parse_numeric_value(self, text: str) -> float | None:
        """
        Parse a tagged span's text into a numeric value where possible.

        Args:
            text: The tagged span's raw text (e.g. "$ 416,161" or "7.49")

        Returns:
            Parsed float, or None if the span isn't a plain number
        """
        cleaned = _NUMERIC_CHARS.sub("", text)

        if not cleaned or cleaned in ("-", "."):
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None
