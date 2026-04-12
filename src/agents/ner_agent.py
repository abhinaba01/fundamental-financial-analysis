"""
NER Agent: Named Entity Recognition for financial documents.

Uses nlpaueb/sec-bert-base fine-tuned on FiNER-139 to extract:
- Company names, ticker symbols
- Financial instruments (stocks, bonds, derivatives)
- Financial metrics and measures
- XBRL tags and accounting concepts

Input: GraphState with document, chunks populated
Output: GraphState with ner_results populated
"""

from __future__ import annotations

from typing import Any

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Model configuration
MODEL_NAME = "dslim/bert-base-NER"


class NERAgent:
    """Named Entity Recognition agent for financial documents."""

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the NER agent.

        Args:
            model_name: HuggingFace model ID for NER
        """
        if pipeline is None:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.logger = logger
        self.model_name = model_name

        self.logger.info(f"Loading NER model: {model_name}")
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
        )
        self.logger.info(f"NER model loaded successfully")

    def __call__(self, state: GraphState) -> GraphState:
        """
        Execute NER on the document in the state.

        Args:
            state: GraphState with document populated

        Returns:
            Updated GraphState with ner_results populated
        """
        document = state.get("document")

        if not document:
            self.logger.warning("No document in state. Skipping NER.")
            return state

        self.logger.info(f"Running NER on document: {document.doc_id}")

        # Extract entities from cleaned text (using full document text)
        entities = self._extract_entities(document.cleaned_text)

        # Also extract from individual chunks for span information
        chunk_entities = self._extract_chunk_entities(document.chunks)

        ner_results = {
            "document_entities": entities,
            "chunk_entities": chunk_entities,
            "total_entities": len(entities),
            "entity_types": self._summarize_entity_types(entities),
        }

        state["ner_results"] = ner_results

        self.logger.info(
            f"NER complete: {len(entities)} entities extracted, "
            f"{len(chunk_entities)} chunks processed"
        )

        return state

    def _extract_entities(self, text: str, max_length: int = 512) -> list[dict[str, Any]]:
        """
        Extract named entities from text using the NER model.

        Args:
            text: Text to analyze
            max_length: Maximum text length for model (chunks if needed)

        Returns:
            List of entity dictionaries with entity, label, start, end, score
        """
        if not text or len(text.strip()) == 0:
            return []

        try:
            # Chunk text if it's too long
            if len(text) > max_length * 4:
                chunks = [
                    text[i : i + max_length * 4]
                    for i in range(0, len(text), max_length * 4)
                ]
                all_entities = []
                for i, chunk in enumerate(chunks):
                    chunk_ents = self.ner_pipeline(chunk)
                    all_entities.extend(chunk_ents)
                return all_entities
            else:
                entities = self.ner_pipeline(text)
                return entities

        except Exception as e:
            self.logger.error(f"Error during NER extraction: {e}")
            return []

    def _extract_chunk_entities(self, chunks) -> dict[str, list[dict[str, Any]]]:
        """
        Extract entities from individual document chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary mapping chunk_id to list of entities
        """
        chunk_entities = {}

        for chunk in chunks:
            try:
                entities = self._extract_entities(chunk.text)
                if entities:
                    chunk_entities[chunk.chunk_id] = entities
            except Exception as e:
                self.logger.debug(
                    f"Error extracting entities from chunk {chunk.chunk_id}: {e}"
                )

        return chunk_entities

    def _summarize_entity_types(
        self, entities: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        Summarize extracted entities by type.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary mapping entity types to counts
        """
        entity_type_counts = {}

        for entity in entities:
            entity_type = entity.get("entity_group", "unknown")
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        return entity_type_counts
