"""
DocumentParser: Ingestion stage for multiple financial document formats.

Handles:
- PDF files (via pdfplumber)
- HTML files including SEC EDGAR HTML (via BeautifulSoup4)
- Plain text files (including earnings call transcripts)
- JSON files containing financial document data
- SEC filing section detection and mapping (10-K/10-Q Items)

Output: ParsedDocument with raw_text, tables, section_map, and metadata populated
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from src.preprocessing.document import DocumentType, ParsedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Compiled regex patterns for metadata extraction
TICKER_PATTERN = re.compile(r"^([A-Z]{1,5})[\s_-]", re.IGNORECASE)
FISCAL_PERIOD_PATTERN = re.compile(
    r"(FY|Q1|Q2|Q3|Q4)[\s_-]?(\d{4})",
    re.IGNORECASE,
)
DOC_TYPE_PATTERN = re.compile(
    r"(10-K|10-Q|10K|10Q)",
    re.IGNORECASE,
)

# SEC filing section headers (Item numbers for 10-K and 10-Q)
SEC_ITEM_MAPPING = {
    "item 1a": "risk_factors",
    "item 1b": "unresolved_staff_comments",
    "item 7": "mda",
    "item 8": "financial_statements",
}


class DocumentParser:
    """Parse financial documents in multiple formats and extract structured content."""

    def __init__(self):
        """Initialize the DocumentParser."""
        self.logger = logger

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a financial document from file path and return ParsedDocument.

        Args:
            file_path: Path to document file (.txt, .htm, .html, .pdf, .json)

        Returns:
            ParsedDocument with raw_text, tables, section_map populated

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        self.logger.info(f"Parsing {file_path.name} (format: {suffix})")

        # Route to appropriate parser
        if suffix == ".pdf":
            doc = self._parse_pdf(file_path)
        elif suffix in (".htm", ".html"):
            doc = self._parse_html(file_path)
        elif suffix == ".txt":
            doc = self._parse_txt(file_path)
        elif suffix == ".json":
            doc = self._parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Extract metadata from filename
        doc = self._extract_metadata_from_filename(doc, file_path.stem)
        doc = replace(doc, source_path=str(file_path))

        self.logger.info(f"Parsed: {doc.summary()}")
        return doc

    def _parse_pdf(self, file_path: Path) -> ParsedDocument:
        """
        Parse PDF file using pdfplumber.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with raw_text and tables extracted
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber not installed. Install with: pip install pdfplumber")

        raw_text_parts = []
        tables_list = []
        page_map = {}

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text from page
                    text = page.extract_text()
                    if text:
                        raw_text_parts.append(text)
                        page_map[f"page_{page_num}"] = len(
                            "".join(raw_text_parts)
                        )

                    # Extract tables from page
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            tables_list.append(
                                {
                                    "page": page_num,
                                    "data": table,
                                    "metadata": {"source": "pdf"},
                                }
                            )

        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

        raw_text = "\n".join(raw_text_parts)

        return ParsedDocument(
            raw_text=raw_text,
            tables=tables_list,
            metadata={"page_map": page_map},
        )

    def _parse_html(self, file_path: Path) -> ParsedDocument:
        """
        Parse HTML file using BeautifulSoup4.

        For SEC filings, identifies Item sections and builds section_map.

        Args:
            file_path: Path to HTML file

        Returns:
            ParsedDocument with raw_text, tables, and section_map populated
        """
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4 lxml")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading HTML file {file_path}: {e}")
            raise

        soup = BeautifulSoup(html_content, "lxml")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        raw_text = soup.get_text(separator="\n", strip=True)

        # Extract tables
        tables_list = self._extract_tables_from_soup(soup)

        # Detect SEC filing and build section_map
        section_map = self._detect_sec_sections_html(raw_text)

        return ParsedDocument(
            raw_text=raw_text,
            tables=tables_list,
            section_map=section_map,
        )

    def _parse_txt(self, file_path: Path) -> ParsedDocument:
        """
        Parse plain text file.

        For earnings call transcripts, splits by speaker turns and creates sections.

        Args:
            file_path: Path to text file

        Returns:
            ParsedDocument with raw_text and section_map populated
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception as e:
            self.logger.error(f"Error reading text file {file_path}: {e}")
            raise

        # Detect if this is an earnings call transcript
        is_earnings_call = self._detect_earnings_call(raw_text)
        section_map = {}

        if is_earnings_call:
            section_map = self._parse_earnings_call_sections(raw_text)

        return ParsedDocument(
            raw_text=raw_text,
            doc_type=DocumentType.EARNINGS_CALL if is_earnings_call else DocumentType.UNKNOWN,
            tables=[],
            section_map=section_map,
        )

    def _parse_json(self, file_path: Path) -> ParsedDocument:
        """
        Parse JSON file containing financial document data.

        Expects JSON with optional 'text' or 'content' field containing the document text.
        Falls back to converting the entire JSON to string if no text field found.

        Args:
            file_path: Path to JSON file

        Returns:
            ParsedDocument with raw_text populated from JSON
        """
        import json

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                json_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {e}")
            raise

        # Extract text content from JSON
        raw_text = ""
        if isinstance(json_data, dict):
            # Try common text fields
            for field in ["text", "content", "body", "document", "data"]:
                if field in json_data and isinstance(json_data[field], str):
                    raw_text = json_data[field]
                    break
            # If no text field found, convert entire JSON to formatted string
            if not raw_text:
                raw_text = json.dumps(json_data, indent=2)
        elif isinstance(json_data, str):
            raw_text = json_data
        else:
            # Convert to string representation
            raw_text = str(json_data)

        return ParsedDocument(
            raw_text=raw_text,
            tables=[],
            section_map={},
        )

    def _extract_tables_from_soup(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """
        Extract all tables from BeautifulSoup object.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of table dictionaries with data and metadata
        """
        tables_list = []

        for table_elem in soup.find_all("table"):
            rows = []
            for tr in table_elem.find_all("tr"):
                cells = []
                for td in tr.find_all(["td", "th"]):
                    cells.append(td.get_text(strip=True))
                if cells:
                    rows.append(cells)

            if rows:
                tables_list.append(
                    {
                        "data": rows,
                        "metadata": {"source": "html_table"},
                    }
                )

        return tables_list

    def _detect_sec_sections_html(self, raw_text: str) -> dict[str, tuple[int, int]]:
        """
        Detect SEC filing sections (Items) in text and return character spans.

        Args:
            raw_text: Raw text from HTML

        Returns:
            Dictionary mapping section names to (start, end) character offsets
        """
        section_map = {}

        # Convert text to lowercase for matching
        text_lower = raw_text.lower()

        # Search for each SEC Item
        for item_name, section_key in SEC_ITEM_MAPPING.items():
            match = re.search(rf"\b{re.escape(item_name)}\b", text_lower)
            if match:
                start = match.start()
                # Find the next Item header or end of text
                next_item_match = None
                for check_item in SEC_ITEM_MAPPING.keys():
                    if check_item != item_name:
                        m = re.search(
                            rf"\b{re.escape(check_item)}\b",
                            text_lower[start + len(item_name) :],
                        )
                        if m:
                            if next_item_match is None or m.start() < next_item_match.start():
                                next_item_match = m

                if next_item_match:
                    end = start + len(item_name) + next_item_match.start()
                else:
                    end = len(raw_text)

                section_map[section_key] = (start, end)

        return section_map

    def _detect_earnings_call(self, raw_text: str) -> bool:
        """
        Heuristic to detect if text is an earnings call transcript.

        Args:
            raw_text: Raw text to check

        Returns:
            True if appears to be earnings call transcript
        """
        # Common patterns in earnings call transcripts
        patterns = [
            r"\b(Operator|Moderator):",
            r"\b(Analysts?|Participants?|Speakers?):",
            r"(Q&A|Question and Answer)",
            r"\b(earnings call|earnings? (conference )?call)\b",
        ]

        text_sample = raw_text[:5000].lower()
        return sum(1 for pattern in patterns if re.search(pattern, text_sample)) >= 2

    def _parse_earnings_call_sections(self, raw_text: str) -> dict[str, tuple[int, int]]:
        """
        Parse earnings call transcript and split into speaker sections.

        Args:
            raw_text: Raw earnings call text

        Returns:
            Dictionary mapping speaker/section names to (start, end) character offsets
        """
        section_map = {}

        # Find all speaker patterns (e.g., "John Smith:", "Analyst:", "Operator:")
        speaker_pattern = re.compile(r"^([A-Za-z\s\-\.]+?):\s*", re.MULTILINE)

        matches = list(speaker_pattern.finditer(raw_text))

        for i, match in enumerate(matches):
            speaker_name = match.group(1).strip().lower()
            start = match.start()

            # End is the start of the next speaker or end of text
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

            # Use speaker name as section key, avoid duplicates
            section_key = speaker_name
            counter = 1
            while section_key in section_map:
                section_key = f"{speaker_name}_{counter}"
                counter += 1

            section_map[section_key] = (start, end)

        return section_map

    def _extract_metadata_from_filename(
        self, doc: ParsedDocument, filename_stem: str
    ) -> ParsedDocument:
        """
        Extract ticker, fiscal_period, and doc_type from filename.

        Expected patterns:
        - "AAPL_10K_FY2023"
        - "MSFT_10Q_Q2_2024"
        - "GOOG_10-K_FY2024"

        Args:
            doc: ParsedDocument to update
            filename_stem: Filename without extension

        Returns:
            Updated ParsedDocument with ticker, fiscal_period, doc_type
        """
        ticker = None
        fiscal_period = None
        doc_type = None

        # Extract ticker
        ticker_match = TICKER_PATTERN.search(filename_stem)
        if ticker_match:
            ticker = ticker_match.group(1).upper()

        # Extract fiscal period (e.g., FY2024, Q2_2024)
        period_match = FISCAL_PERIOD_PATTERN.search(filename_stem)
        if period_match:
            fiscal_period = f"{period_match.group(1).upper()}{period_match.group(2)}"

        # Extract document type
        type_match = DOC_TYPE_PATTERN.search(filename_stem)
        if type_match:
            try:
                doc_type = DocumentType(type_match.group(1).upper())
            except ValueError:
                pass

        return replace(
            doc,
            ticker=ticker,
            fiscal_period=fiscal_period,
            doc_type=doc_type or doc.doc_type,
        )

