from __future__ import annotations

import re
import uuid
from dataclasses import dataclass,field

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

class DocumentType(str,Enum):
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    EARNINGS_CALL = "earnings_call"
    FINANCIAL_NEWS = "financial_news"
    UNKNOWN = "unknown"

class ChunkType(str,Enum):
    MDA = "mda"
    RISK_FACTORS = "risk_factors"
    FINANCIALS = "financial_statements"
    GENERAL = "general"

@dataclass
class DocumentChunk:

    def get_section(self,section_name:str) -> str | None:
        
        span = self.section_map.get(section_name.lower())
        if span is None:
            return None
        start,end = span
        return self.cleaned_text[start:end]
    

    def get_embedded_chunks(self) -> list[DocumentChunk]:
        return [c for c in self.chunks if c.embedding is not None]
    

    def __rep__(self) -> str:
         return (
            f"ParsedDocument(ticker={self.ticker!r}, "
            f"type={self.doc_type.value!r}, "
            f"period={self.fiscal_period!r}, "
            f"chunks={len(self.chunks)})"
        )
      
