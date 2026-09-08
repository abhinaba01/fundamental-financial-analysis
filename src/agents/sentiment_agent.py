"""
Sentiment Agent: Financial sentiment analysis using FinBERT.

Uses:
- Primary: ProsusAI/finbert (trained on Financial PhraseBank)
- Secondary: yiyanghkust/finbert-tone (for management tone scoring)

Performs:
- Document-level sentiment classification (positive/negative/neutral)
- Tone analysis from earnings calls
- Quantitative sentiment scoring (-1.0 to 1.0)

Input: GraphState with document, chunks populated
Output: GraphState with sentiment_results populated
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
PRIMARY_MODEL = "ProsusAI/finbert"
SECONDARY_MODEL = "yiyanghkust/finbert-tone"

# Sentiment mapping
SENTIMENT_LABEL_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}

TONE_LABEL_MAP = {
    "Negative": -1.0,
    "negative": -1.0,
    "Neutral": 0.0,
    "neutral": 0.0,
    "Positive": 1.0,
    "positive": 1.0,
}


class SentimentAgent:
    """Sentiment analysis agent for financial documents."""

    def __init__(
        self,
        primary_model: str = PRIMARY_MODEL,
        secondary_model: str = SECONDARY_MODEL,
        device: str = "cpu",
    ):
        """
        Initialize the Sentiment agent.

        Args:
            primary_model: Primary sentiment model (Financial PhraseBank trained)
            secondary_model: Secondary tone model (for earnings calls)
            device: Device to run the models on ('cpu' or 'cuda')
        """
        if pipeline is None:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.logger = logger
        self.primary_model = primary_model
        self.secondary_model = secondary_model

        self.logger.info(f"Loading sentiment model: {primary_model} on {device}")
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=primary_model,
            device=device,
        )

        self.logger.info(f"Loading tone model: {secondary_model} on {device}")
        try:
            self.tone_pipeline = pipeline(
                "text-classification",
                model=secondary_model,
                device=device,
            )
        except Exception as e:
            self.logger.warning(
                f"Tone model {secondary_model} failed to load ({e}). "
                "Falling back to primary sentiment model for tone analysis."
            )
            self.tone_pipeline = self.sentiment_pipeline

        self.logger.info("Sentiment and tone models loaded successfully")

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """
        Execute sentiment analysis on the document in the state.

        Runs as a parallel branch, so it returns only the key it owns and
        treats `state` as read-only - see NERAgent.__call__ for why.

        Args:
            state: GraphState with document populated

        Returns:
            State delta containing sentiment_results
        """
        document = state.get("document")

        if not document:
            self.logger.warning("No document in state. Skipping sentiment analysis.")
            return {}

        self.logger.info(f"Running sentiment analysis on document: {document.doc_id}")

        # Perform sentiment analysis
        doc_sentiment = self._analyze_document_sentiment(document.cleaned_text)
        chunk_sentiments = self._analyze_chunk_sentiments(document.chunks)

        # Perform tone analysis (if earnings call)
        tone_analysis = {}
        if document.doc_type.value == "earnings_call":
            tone_analysis = self._analyze_tone(document.cleaned_text)

        # Aggregate results
        sentiment_results = {
            "overall_sentiment": doc_sentiment.get("label", "neutral"),
            "overall_score": doc_sentiment.get("score", 0.0),
            "sentiment_distribution": self._aggregate_sentiments(chunk_sentiments),
            "chunk_sentiments": chunk_sentiments,
            "tone_analysis": tone_analysis,
            "is_positive": doc_sentiment.get("score", 0.0) > 0.5,
        }

        self.logger.info(
            f"Sentiment analysis complete: {sentiment_results['overall_sentiment']} "
            f"({sentiment_results['overall_score']:.3f})"
        )

        return {"sentiment_results": sentiment_results}

    def _analyze_document_sentiment(self, text: str) -> dict[str, Any]:
        """
        Analyze overall sentiment of document.

        Args:
            text: Document text

        Returns:
            Dictionary with label and aggregated score
        """
        if not text or len(text.strip()) == 0:
            return {"label": "neutral", "score": 0.0}

        try:
            # Split text into sentences and analyze them as a single batch
            # instead of one pipeline call per sentence - matters a lot for
            # documents with many sentences, especially on GPU.
            sentences = text.split(".")[:50]
            valid_sentences = [s[:512] for s in sentences if len(s.strip()) >= 5]

            sentiments = []
            if valid_sentences:
                results = self.sentiment_pipeline(valid_sentences)
                for result in results:
                    label = result["label"].lower()
                    score = result["score"]

                    # Convert to numerical scale
                    sentiment_value = SENTIMENT_LABEL_MAP.get(label, 0.0)
                    sentiments.append(sentiment_value * score)

            # Aggregate
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                # Determine overall label
                if avg_sentiment > 0.2:
                    overall_label = "positive"
                elif avg_sentiment < -0.2:
                    overall_label = "negative"
                else:
                    overall_label = "neutral"

                return {"label": overall_label, "score": avg_sentiment}
            else:
                return {"label": "neutral", "score": 0.0}

        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}")
            return {"label": "neutral", "score": 0.0}

    def _analyze_chunk_sentiments(self, chunks) -> dict[str, dict[str, Any]]:
        """
        Analyze sentiment of individual chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary mapping chunk_id to sentiment results
        """
        chunk_sentiments = {}

        # Batch all chunks into a single pipeline call instead of looping one
        # at a time - for a document with hundreds of chunks this is the
        # difference between hundreds of forward passes and a handful of
        # batched ones, especially on GPU.
        valid_chunks = [c for c in chunks if len(c.text.strip()) >= 5]

        if not valid_chunks:
            return chunk_sentiments

        try:
            texts = [chunk.text[:512] for chunk in valid_chunks]
            results = self.sentiment_pipeline(texts)

            for chunk, result in zip(valid_chunks, results):
                label = result["label"].lower()
                score = result["score"]
                sentiment_value = SENTIMENT_LABEL_MAP.get(label, 0.0)

                chunk_sentiments[chunk.chunk_id] = {
                    "label": label,
                    "score": sentiment_value * score,
                    "confidence": score,
                }
        except Exception as e:
            self.logger.debug(f"Error analyzing chunk sentiments: {e}")

        return chunk_sentiments

    def _analyze_tone(self, text: str) -> dict[str, Any]:
        """
        Analyze management tone (for earnings calls).

        Args:
            text: Document text (typically earnings call transcript)

        Returns:
            Dictionary with tone analysis results
        """
        if not text or len(text.strip()) == 0:
            return {}

        try:
            # Analyze opening and closing remarks for tone
            sentences = text.split(".")
            opening = " ".join(sentences[:10])
            closing = " ".join(sentences[-10:])

            opening_tone = self.tone_pipeline(opening[:512])
            closing_tone = self.tone_pipeline(closing[:512])

            opening_label = opening_tone[0]["label"] if opening_tone else "Neutral"
            opening_label = opening_label.title() if isinstance(opening_label, str) else opening_label
            opening_score = TONE_LABEL_MAP.get(opening_label, TONE_LABEL_MAP.get(str(opening_label).lower(), 0.0))

            closing_label = closing_tone[0]["label"] if closing_tone else "Neutral"
            closing_label = closing_label.title() if isinstance(closing_label, str) else closing_label
            closing_score = TONE_LABEL_MAP.get(closing_label, TONE_LABEL_MAP.get(str(closing_label).lower(), 0.0))

            return {
                "opening_tone": opening_label,
                "opening_score": opening_score,
                "closing_tone": closing_label,
                "closing_score": closing_score,
                "tone_progression": closing_score - opening_score,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing tone: {e}")
            return {}

    def _aggregate_sentiments(
        self, chunk_sentiments: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """
        Aggregate sentiment distribution across chunks.

        Args:
            chunk_sentiments: Dictionary of chunk sentiments

        Returns:
            Distribution of sentiment labels
        """
        distribution = {"positive": 0, "neutral": 0, "negative": 0}

        for chunk_result in chunk_sentiments.values():
            label = chunk_result.get("label", "neutral")
            distribution[label] = distribution.get(label, 0) + 1

        # Convert to percentages if any chunks exist
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        return distribution
