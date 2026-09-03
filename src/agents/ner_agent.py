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

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        """
        Initialize the NER agent.

        Args:
            model_name: HuggingFace model ID for NER
            device: Device to run the model on ('cpu' or 'cuda')
        """
        if pipeline is None:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.logger = logger
        self.model_name = model_name

        self.logger.info(f"Loading NER model: {model_name} on {device}")
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=device,
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

        ner_results = {
            "document_entities": entities,
            "total_entities": len(entities),
            "entity_types": self._summarize_entity_types(entities),
        }

        state["ner_results"] = ner_results

        self.logger.info(f"NER complete: {len(entities)} entities extracted")

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
                slices = [
                    text[i : i + max_length * 4]
                    for i in range(0, len(text), max_length * 4)
                ]
                # Pass the whole batch to the pipeline in one call instead of
                # looping one slice at a time - the pipeline batches these
                # internally, which matters a lot on GPU and for documents
                # with many slices (e.g. a full 10-K).
                batched_results = self.ner_pipeline(slices)
                all_entities = []
                for entities_for_slice in batched_results:
                    all_entities.extend(entities_for_slice)
                return all_entities
            else:
                entities = self.ner_pipeline(text)
                return entities

        except Exception as e:
            self.logger.error(f"Error during NER extraction: {e}")
            return []

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
