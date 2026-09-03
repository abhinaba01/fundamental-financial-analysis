"""
SemanticChunker: Section-aware semantic chunking stage.

Performs:
- Token-based chunking (512 tokens max per chunk, tiktoken cl100k_base)
- Sliding overlap (64 tokens from previous chunk)
- Section-aware splitting (never merge across section boundaries)
- Sentence-level safe split points (spaCy sentencizer)
- Automatic chunk type assignment based on section

Output: ParsedDocument with chunks populated
"""

from __future__ import annotations

import tiktoken
from dataclasses import replace

try:
    import spacy
except ImportError:
    spacy = None

from src.preprocessing.document import ChunkType, DocumentChunk, ParsedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
MAX_TOKENS_PER_CHUNK = 512
OVERLAP_TOKENS = 64
ENCODING_NAME = "cl100k_base"

# Section to chunk type mapping
SECTION_TYPE_MAPPING = {
    "mda": ChunkType.MDA,
    "risk_factors": ChunkType.RISK_FACTORS,
    "financial_statements": ChunkType.FINANCIALS,
}


class SemanticChunker:
    """Chunk documents with semantic awareness and proper overlap."""

    def __init__(self):
        """Initialize the SemanticChunker with tokenizer and spacy model."""
        self.logger = logger
        self.encoding = tiktoken.get_encoding(ENCODING_NAME)

        # Load spaCy for sentence boundaries only - _get_sentence_boundaries
        # reads nothing but sent.end_char, so the tagger/parser/ner/lemmatizer
        # components (the expensive part of en_core_web_sm) are unnecessary
        # and are excluded in favor of the lightweight rule-based sentencizer.
        # On a full 10-K (~270K chars), the full pipeline takes ~50s; this
        # takes a fraction of a second.
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load(
                    "en_core_web_sm",
                    exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
                )
                self.nlp.add_pipe("sentencizer")
            except OSError:
                self.logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )

    def chunk(self, doc: ParsedDocument) -> ParsedDocument:
        """
        Chunk a ParsedDocument's cleaned_text into semantic chunks.

        Args:
            doc: ParsedDocument with cleaned_text populated

        Returns:
            ParsedDocument with chunks populated
        """
        self.logger.info(f"Chunking document: {doc.doc_id}")

        if not doc.cleaned_text:
            self.logger.warning("Document has no cleaned_text. Returning empty chunks.")
            return replace(doc, chunks=[])

        # Build chunks section-aware
        chunks = self._chunk_with_sections(doc)

        # Set chunk IDs and token counts
        for chunk in chunks:
            chunk.token_count = len(self.encoding.encode(chunk.text))

        chunked_doc = replace(doc, chunks=chunks)

        self.logger.info(f"Created {len(chunks)} chunks with {OVERLAP_TOKENS} token overlap")

        return chunked_doc

    def _chunk_with_sections(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk text respecting section boundaries.

        Args:
            doc: ParsedDocument with section_map

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        text = doc.cleaned_text

        # If no sections, chunk the entire document
        if not doc.section_map:
            chunks.extend(self._chunk_text(text, ChunkType.GENERAL))
            return chunks

        # Process each section separately
        for section_name, (start, end) in doc.section_map.items():
            section_text = text[start:end]
            chunk_type = SECTION_TYPE_MAPPING.get(section_name, ChunkType.GENERAL)

            section_chunks = self._chunk_text(
                section_text, chunk_type, section_name=section_name
            )
            
            # Update chunk metadata with section info
            for chunk in section_chunks:
                chunk.metadata["section"] = section_name
                chunk.metadata["start_offset"] = start + int(chunk.metadata.get("local_start", 0))

            chunks.extend(section_chunks)

        return chunks

    def _chunk_text(
        self,
        text: str,
        chunk_type: ChunkType = ChunkType.GENERAL,
        section_name: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a text string using token-aware sliding window.

        Args:
            text: Text to chunk
            chunk_type: Type to assign to chunks
            section_name: Optional section name for metadata

        Returns:
            List of DocumentChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []

        # Tokenize
        tokens = self.encoding.encode(text)

        if len(tokens) <= MAX_TOKENS_PER_CHUNK:
            # Text fits in one chunk
            return [
                DocumentChunk(
                    text=text,
                    chunk_type=chunk_type,
                    metadata={"section": section_name} if section_name else {},
                )
            ]

        # Find sentence boundaries using spaCy
        sentence_boundaries = self._get_sentence_boundaries(text)

        chunks = []
        chunk_start_token_idx = 0

        while chunk_start_token_idx < len(tokens):
            # Find the end token index for this chunk
            chunk_end_token_idx = min(
                chunk_start_token_idx + MAX_TOKENS_PER_CHUNK,
                len(tokens),
            )

            # Find nearest sentence boundary before or at chunk_end_token_idx
            char_end = self._find_char_position_for_token_idx(
                text, tokens, chunk_end_token_idx
            )

            # Adjust to sentence boundary if possible
            if sentence_boundaries:
                # Find the nearest sentence boundary before char_end
                best_boundary = 0
                for boundary in sentence_boundaries:
                    if boundary <= char_end:
                        best_boundary = boundary
                    else:
                        break

                if best_boundary > self._find_char_position_for_token_idx(
                    text, tokens, chunk_start_token_idx
                ):
                    char_end = best_boundary

            # Extract chunk text
            char_start = self._find_char_position_for_token_idx(
                text, tokens, chunk_start_token_idx
            )
            chunk_text = text[char_start:char_end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        chunk_type=chunk_type,
                        metadata={
                            "section": section_name,
                            "local_start": char_start,
                        }
                        if section_name
                        else {"local_start": char_start},
                    )
                )

            # This was the last chunk - stop here. Checking this before
            # advancing (rather than checking the new start index afterward)
            # matters once chunk_end_token_idx has been clamped to len(tokens):
            # chunk_end_token_idx - OVERLAP_TOKENS then stays fixed forever,
            # since it no longer depends on chunk_start_token_idx, which made
            # the old post-hoc check (comparing the new start index to
            # len(tokens) - 1) unreachable and looped forever on any document
            # needing more than one sliding-window chunk.
            if chunk_end_token_idx >= len(tokens):
                break

            # Move to next chunk with overlap
            chunk_start_token_idx = chunk_end_token_idx - OVERLAP_TOKENS

        return chunks

    def _get_sentence_boundaries(self, text: str) -> list[int]:
        """
        Get character positions of sentence boundaries using spaCy.

        Args:
            text: Text to analyze

        Returns:
            List of character positions where sentences end
        """
        if self.nlp is None:
            return []

        try:
            doc = self.nlp(text)
            boundaries = [sent.end_char for sent in doc.sents]
            return sorted(set(boundaries))
        except Exception as e:
            self.logger.debug(f"Error detecting sentence boundaries: {e}")
            return []

    def _find_char_position_for_token_idx(
        self, text: str, tokens: list[int], token_idx: int
    ) -> int:
        """
        Find the character position corresponding to a token index.

        Args:
            text: Original text
            tokens: Encoded token list
            token_idx: Index in token list

        Returns:
            Character position in text
        """
        # Clamp token_idx
        token_idx = min(token_idx, len(tokens))

        # Decode tokens up to token_idx: since tokens came from encoding this
        # same text, the decoded prefix always starts at position 0 - no need
        # to search for it (text.find() here was an O(n) scan for something
        # always found at 0, which made this function - called repeatedly
        # with a growing prefix - the dominant cost on large documents).
        tokens_up_to = tokens[:token_idx]
        decoded = self.encoding.decode(tokens_up_to)

        if text.startswith(decoded):
            return len(decoded)

        # Fallback: estimate based on token count
        estimated_chars_per_token = len(text) / len(tokens) if tokens else 1
        return min(int(token_idx * estimated_chars_per_token), len(text))

