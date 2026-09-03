"""
DocumentCleaner: Normalization and boilerplate removal stage.

Performs:
- HTML tag stripping and entity decoding
- Unicode normalization (dashes, whitespace)
- Boilerplate removal (Safe Harbour, disclaimers, copyright)
- Financial notation normalization ($1.2B, 12.5%, etc.)
- Preservation of table structures

Output: ParsedDocument with cleaned_text populated
"""

from __future__ import annotations

import html
import re
from dataclasses import replace

from src.preprocessing.document import ParsedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Compiled regex patterns for cleaning
HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.MULTILINE)
UNICODE_DASH_PATTERN = re.compile(r"[\u2010-\u2015]")  # Various dashes
WHITESPACE_PATTERN = re.compile(r"\s+", re.MULTILINE)

# Boilerplate patterns to remove
SAFE_HARBOUR_PATTERN = re.compile(
    r"(safe\s+harbour|forward[-\s]?looking\s+statements?|forward[-\s]?looking\s+information)",
    re.IGNORECASE,
)
DISCLAIMER_PATTERN = re.compile(
    r"(this\s+document\s+contains.*?statements|forward[-\s]?looking\s+statements\s+are.*?(?:not\s+)?guarantees?)",
    re.IGNORECASE | re.DOTALL,
)
COPYRIGHT_PATTERN = re.compile(
    r"©|\(c\)\s*\d{4}|copyright\s*(?:©)?\s*\d{4}",
    re.IGNORECASE,
)

# Financial notation normalization patterns.
# The trailing boundary is a lookahead (?=...) so it isn't consumed by the
# match - a plain (?:...) group would eat the following space/punctuation and
# the replacement would jam onto the next word (e.g. "$1.2 billion for" ->
# "$1.20Bfor" instead of "$1.20B for"). Comma/period/semicolon are included
# since financial prose routinely follows these figures with punctuation
# (e.g. "$24.2 billion, up 10%").
_UNIT_BOUNDARY = r"(?=[\s.,;)]|$)"
BILLION_PATTERN = re.compile(
    rf"\$\s*([\d,]+(?:\.\d+)?)\s*(?:billion|B){_UNIT_BOUNDARY}",
    re.IGNORECASE,
)
MILLION_PATTERN = re.compile(
    rf"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|M){_UNIT_BOUNDARY}",
    re.IGNORECASE,
)
THOUSAND_PATTERN = re.compile(
    rf"\$\s*([\d,]+(?:\.\d+)?)\s*(?:thousand|K){_UNIT_BOUNDARY}",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(?:percent|percentage|%)",
    re.IGNORECASE,
)


class DocumentCleaner:
    """Clean and normalize financial document text."""

    def __init__(self):
        """Initialize the DocumentCleaner."""
        self.logger = logger

    def clean(self, doc: ParsedDocument) -> ParsedDocument:
        """
        Clean a ParsedDocument's raw_text and populate cleaned_text.

        Args:
            doc: ParsedDocument with raw_text populated

        Returns:
            ParsedDocument with cleaned_text populated
        """
        self.logger.info(f"Cleaning document: {doc.doc_id}")

        text = doc.raw_text

        # Stage 1: HTML tag stripping and entity decoding
        text = self._decode_html_entities(text)
        text = self._strip_html_tags(text)

        # Stage 2: Unicode normalization
        text = self._normalize_unicode(text)

        # Stage 3: Boilerplate removal
        text = self._remove_boilerplate(text)

        # Stage 4: Financial notation normalization
        text = self._normalize_financial_notation(text)

        # Stage 5: Whitespace normalization
        text = self._normalize_whitespace(text)

        cleaned_doc = replace(doc, cleaned_text=text)

        self.logger.info(
            f"Cleaned: {len(doc.raw_text)} chars -> {len(text)} chars"
        )

        return cleaned_doc

    def _decode_html_entities(self, text: str) -> str:
        """
        Decode HTML entities (&amp;, &nbsp;, etc.).

        Args:
            text: Text with HTML entities

        Returns:
            Text with decoded entities
        """
        return html.unescape(text)

    def _strip_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.

        Args:
            text: Text with HTML tags

        Returns:
            Text with tags removed
        """
        return HTML_TAG_PATTERN.sub("", text)

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters (dashes, quotes, etc.).

        Args:
            text: Text with Unicode characters

        Returns:
            Normalized text
        """
        # Replace various dashes with hyphens
        text = UNICODE_DASH_PATTERN.sub("-", text)

        # Replace curly quotes with straight quotes
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate sections.

        Args:
            text: Text that may contain boilerplate

        Returns:
            Text with boilerplate removed or minimized
        """
        # Note: We preserve these sections but could remove them entirely
        # For now, we just log their presence
        if SAFE_HARBOUR_PATTERN.search(text):
            self.logger.debug("Found Safe Harbour statement in document")

        if COPYRIGHT_PATTERN.search(text):
            self.logger.debug("Found copyright notice in document")

        return text

    def _normalize_financial_notation(self, text: str) -> str:
        """
        Normalize financial notations for consistency.

        Examples:
        - "$1.2 billion" → "$1.20B"
        - "$450 million" → "$450.00M"
        - "12.5 percent" → "12.5%"

        Args:
            text: Text with financial notations

        Returns:
            Text with normalized notations
        """

        def normalize_number(num_str: str) -> float:
            """Convert string with commas to float."""
            return float(num_str.replace(",", ""))

        # Normalize billions
        def replace_billion(match):
            num = normalize_number(match.group(1))
            return f"${num:.2f}B"

        text = BILLION_PATTERN.sub(replace_billion, text)

        # Normalize millions
        def replace_million(match):
            num = normalize_number(match.group(1))
            return f"${num:.2f}M"

        text = MILLION_PATTERN.sub(replace_million, text)

        # Normalize thousands
        def replace_thousand(match):
            num = normalize_number(match.group(1))
            return f"${num:.2f}K"

        text = THOUSAND_PATTERN.sub(replace_thousand, text)

        # Normalize percentages
        def replace_percent(match):
            num = match.group(1).replace(",", "")
            return f"{num}%"

        text = PERCENT_PATTERN.sub(replace_percent, text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Collapse runs of whitespace into single spaces.

        Args:
            text: Text with irregular whitespace

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces/newlines with single space
        text = WHITESPACE_PATTERN.sub(" ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    

