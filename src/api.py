"""
Lightweight FastAPI wrapper around the financial analysis pipeline.

Endpoints:
- POST /analyze

Accepts a document upload and query text, runs the pipeline, and returns the report.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.main import run_analysis
from src.utils.logger import get_logger

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

logger = get_logger(__name__)
app = FastAPI(title="Financial Fundamentals Analysis API")


def _serialize_response(content: Any) -> Any:
    """Convert non-native JSON values to JSON-safe Python types."""
    custom_encoders = {}
    if np is not None:
        custom_encoders[np.generic] = lambda value: value.item()

    return jsonable_encoder(content, custom_encoder=custom_encoders)


@app.post("/analyze")
async def analyze_document(
    query: str = Form(...),
    document: UploadFile = File(...),
    use_gpu: bool = Form(False),
) -> Any:
    """Analyze an uploaded financial document and return a structured report."""
    if not document.filename:
        raise HTTPException(status_code=400, detail="Document filename is required.")

    suffix = Path(document.filename).suffix or ".pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / f"uploaded_document{suffix}"
        try:
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(document.file, buffer)
        except Exception as exc:
            logger.error(f"Failed to save uploaded document: {exc}")
            raise HTTPException(status_code=500, detail="Unable to save uploaded document.")

        try:
            report = run_analysis(document_path=temp_path, query=query, use_gpu=use_gpu)
            return JSONResponse(content=_serialize_response(report))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(f"Pipeline execution failed: {exc}")
            raise HTTPException(status_code=500, detail="Pipeline execution failed.")
