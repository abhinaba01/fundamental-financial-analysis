from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentType(str, Enum):
    SEC_10K        = "10-K"
    SEC_10Q        = "10-Q"
    EARNINGS_CALL  = "earnings_call"
    FINANCIAL_NEWS = "financial_news"
    UNKNOWN        = "unknown"


class ChunkType(str, Enum):
    MDA          = "mda"
    RISK_FACTORS = "risk_factors"
    FINANCIALS   = "financial_statements"
    GENERAL      = "general"


@dataclass
class DocumentChunk:
    """A single semantic unit of text produced by the chunker."""

    chunk_id:    str            = field(default_factory=lambda: str(uuid.uuid4()))
    text:        str            = ""
    token_count: int            = 0
    chunk_type:  ChunkType      = ChunkType.GENERAL
    page_number: int | None     = None
    embedding:   list | None    = None
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"DocumentChunk(type={self.chunk_type.value!r}, "
            f"tokens={self.token_count}, "
            f"preview={preview!r}...)"
        )


@dataclass
class ParsedDocument:
    """
    The canonical intermediate representation for any financial document
    in the pipeline. Every agent operates on this object — never on raw
    files or strings.
    """

    doc_id:        str                       = field(default_factory=lambda: str(uuid.uuid4()))
    source_path:   str                       = ""
    doc_type:      DocumentType              = DocumentType.UNKNOWN
    ticker:        str | None                = None
    fiscal_period: str | None                = None
    raw_text:      str                       = ""
    cleaned_text:  str                       = ""
    chunks:        list[DocumentChunk]       = field(default_factory=list)
    tables:        list[dict[str, Any]]      = field(default_factory=list)
    section_map:   dict[str, tuple[int,int]] = field(default_factory=dict)
    metadata:      dict[str, Any]            = field(default_factory=dict)

    def get_section(self, section_name: str) -> str | None:
        """Return the cleaned text for a named section, e.g. 'mda'."""
        span = self.section_map.get(section_name.lower())
        if span is None:
            return None
        start, end = span
        return self.cleaned_text[start:end]

    def get_embedded_chunks(self) -> list[DocumentChunk]:
        """Return only chunks that have been embedded."""
        return [c for c in self.chunks if c.embedding is not None]

    def summary(self) -> dict[str, Any]:
        """Quick-glance summary — useful for notebook debugging."""
        return {
            "doc_id":         self.doc_id,
            "ticker":         self.ticker,
            "type":           self.doc_type.value,
            "fiscal_period":  self.fiscal_period,
            "source":         self.source_path,
            "raw_chars":      len(self.raw_text),
            "cleaned_chars":  len(self.cleaned_text),
            "num_chunks":     len(self.chunks),
            "num_tables":     len(self.tables),
            "sections_found": list(self.section_map.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"ParsedDocument(ticker={self.ticker!r}, "
            f"type={self.doc_type.value!r}, "
            f"period={self.fiscal_period!r}, "
            f"chunks={len(self.chunks)})"
        )