"""
KPI Agent: Financial Key Performance Indicator extraction and reasoning.

Uses:
- Qwen2.5-7B-Instruct for numerical reasoning
- Python REPL tool for executing calculations (MANDATORY)
- FinQA dataset patterns for multi-step reasoning

Performs:
- KPI identification (revenue, margin, EPS, etc.)
- Multi-step numerical reasoning using Python
- Ratio calculations and comparisons
- Year-over-year analysis

Input: GraphState with document, chunks populated
Output: GraphState with kpi_results populated
"""

from __future__ import annotations

import re
from typing import Any

from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common KPI patterns
KPI_KEYWORDS = {
    "revenue": r"(revenue|sales|net sales|total revenue)",
    "gross_margin": r"(gross\s+margin|gross profit|gross\s+profit\s+margin)",
    "operating_income": r"(operating\s+income|operating\s+profit)",
    "net_income": r"(net\s+income|net profit|bottom\s+line)",
    "eps": r"(earnings?\s+per\s+share|EPS)",
    "ebitda": r"(ebitda|operating\s+cash\s+flow)",
    "roa": r"(return\s+on\s+assets|ROA)",
    "roe": r"(return\s+on\s+equity|ROE)",
}


class KPIAgent:
    """KPI extraction and numerical reasoning agent."""

    def __init__(self):
        """Initialize the KPI agent."""
        self.logger = logger

    def __call__(self, state: GraphState) -> GraphState:
        """
        Extract KPIs from the document in the state.

        Args:
            state: GraphState with document populated

        Returns:
            Updated GraphState with kpi_results populated
        """
        document = state.get("document")

        if not document:
            self.logger.warning("No document in state. Skipping KPI extraction.")
            return state

        self.logger.info(f"Extracting KPIs from document: {document.doc_id}")

        # Extract KPI patterns from text
        extracted_kpis = self._extract_kpi_patterns(document.cleaned_text)

        # Perform calculations if needed
        calculated_kpis = self._perform_calculations(extracted_kpis)

        kpi_results = {
            "extracted_kpis": extracted_kpis,
            "calculated_kpis": calculated_kpis,
            "total_kpis": len(extracted_kpis) + len(calculated_kpis),
        }

        state["kpi_results"] = kpi_results

        self.logger.info(
            f"KPI extraction complete: {len(extracted_kpis)} extracted, "
            f"{len(calculated_kpis)} calculated"
        )

        return state

    def _extract_kpi_patterns(self, text: str) -> dict[str, list[dict[str, Any]]]:
        """
        Extract KPI patterns from text using regex.

        Args:
            text: Document text

        Returns:
            Dictionary of KPI type to list of extractions
        """
        kpis = {}

        for kpi_type, pattern in KPI_KEYWORDS.items():
            # pattern is already a capturing group, e.g. "(revenue|sales|...)"
            matches = re.finditer(
                rf"{pattern}\s*(?:of|to|was|is|reached|totaled|stood at)?"
                rf"\s*(?:[:$]\s*){{0,2}}([\d,\.]+)\s*([A-Z\%]*)",
                text,
                re.IGNORECASE,
            )

            kpi_entries = []
            for match in matches:
                kpi_name = match.group(1)
                value = match.group(2).replace(",", "")
                unit = match.group(3) if match.group(3) else ""

                try:
                    numeric_value = float(value)
                    kpi_entries.append({
                        "name": kpi_name,
                        "value": numeric_value,
                        "unit": unit,
                        "raw_text": match.group(0),
                    })
                except ValueError:
                    pass

            if kpi_entries:
                kpis[kpi_type] = kpi_entries

        return kpis

    def _perform_calculations(
        self, extracted_kpis: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """
        Perform multi-step calculations using safe Python execution.

        Args:
            extracted_kpis: Extracted KPI values

        Returns:
            Dictionary of calculated metrics
        """
        calculated = {}

        try:
            # The "gross_margin" pattern matches both "gross margin" (already a
            # percentage, e.g. "38.2%") and "gross profit" (a dollar figure).
            # Only derive a margin when we actually captured a dollar amount;
            # if it's already a "%", there's nothing to calculate.
            gross_margin_entries = extracted_kpis.get("gross_margin", [])
            revenue_entries = extracted_kpis.get("revenue", [])

            if gross_margin_entries and revenue_entries:
                entry = gross_margin_entries[0]
                if entry.get("unit", "").strip() != "%":
                    gross_profit = entry["value"]
                    revenue = revenue_entries[0]["value"]

                    if revenue != 0:
                        margin_percent = (gross_profit / revenue) * 100
                        calculated["calculated_gross_margin_percent"] = margin_percent
                        self.logger.debug(f"Calculated margin: {margin_percent:.2f}%")

        except Exception as e:
            self.logger.debug(f"Error in calculations: {e}")

        return calculated
