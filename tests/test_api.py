"""
Tests for the FastAPI wrapper in src/api.py.

`run_analysis` is monkeypatched throughout: the pipeline itself is covered by
tests/test_pipeline.py, and running it for real here would mean loading every
model and embedding a document per request. What these tests pin down is the
HTTP contract around it - status codes, upload handling, argument
pass-through, and JSON serialization of numpy values.

This file replaces the old root-level `test_api.py`, which was a manual script
that required a server already running on localhost:8000 and asserted nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api
from src.api import app

client = TestClient(app)


@pytest.fixture
def fake_run_analysis(monkeypatch):
    """Replace run_analysis with a recorder, and hand back the recorded calls.

    The document is written into a TemporaryDirectory that is torn down when
    the request ends, so anything the test wants to assert about the uploaded
    file has to be read here, while the call is still in flight.
    """
    calls = []

    def _fake(document_path, query, use_gpu=None):
        calls.append({
            "suffix": document_path.suffix,
            "content": document_path.read_bytes(),
            "query": query,
            "use_gpu": use_gpu,
        })
        return {"summary": "ok"}

    monkeypatch.setattr(src.api, "run_analysis", _fake)
    return calls


def _post(filename: str = "filing.txt", content: bytes = b"Revenue was $100.", **data):
    """POST a document to /analyze with sensible defaults."""
    return client.post(
        "/analyze",
        files={"document": (filename, content, "text/plain")},
        data={"query": "What is the revenue?", **data},
    )


def test_analyze_returns_report_json(fake_run_analysis):
    """A successful run returns the pipeline's report dict verbatim."""
    response = _post()

    assert response.status_code == 200
    assert response.json() == {"summary": "ok"}


def test_analyze_passes_query_and_uploaded_bytes_through(fake_run_analysis):
    """The uploaded bytes and query reach run_analysis intact."""
    response = _post(content=b"Total net sales were $383,285 million.")

    assert response.status_code == 200
    assert len(fake_run_analysis) == 1
    assert fake_run_analysis[0]["query"] == "What is the revenue?"
    assert fake_run_analysis[0]["content"] == b"Total net sales were $383,285 million."


def test_analyze_preserves_file_extension(fake_run_analysis):
    """The temp file keeps the upload's suffix - DocumentParser dispatches on it,
    so a .pdf arriving as .txt would silently take the wrong parsing path."""
    _post(filename="apple_10k.pdf", content=b"%PDF-1.4 fake")

    assert fake_run_analysis[0]["suffix"] == ".pdf"


def test_analyze_defaults_extensionless_upload_to_pdf(fake_run_analysis):
    """An upload with no extension falls back to .pdf, per src/api.py."""
    _post(filename="filing", content=b"data")

    assert fake_run_analysis[0]["suffix"] == ".pdf"


def test_analyze_omitting_use_gpu_leaves_autodetect(fake_run_analysis):
    """use_gpu must arrive as None when the form field is omitted: None is the
    documented sentinel that lets run_analysis auto-detect CUDA, so coercing it
    to False here would silently disable GPU for every API caller."""
    _post()

    assert fake_run_analysis[0]["use_gpu"] is None


def test_analyze_honors_explicit_use_gpu_false(fake_run_analysis):
    """An explicit use_gpu=false is passed through as False, not None."""
    _post(use_gpu="false")

    assert fake_run_analysis[0]["use_gpu"] is False


def test_analyze_serializes_numpy_values(fake_run_analysis, monkeypatch):
    """Reports carry numpy scalars out of the model layers; the encoder has to
    convert them or FastAPI raises on an unserializable response."""
    def _numpy_report(document_path, query, use_gpu=None):
        return {
            "sentiment_analysis": {
                "confidence_score": np.float32(0.87),
                "count": np.int64(42),
            }
        }

    monkeypatch.setattr(src.api, "run_analysis", _numpy_report)

    response = _post()

    assert response.status_code == 200
    payload = response.json()["sentiment_analysis"]
    assert payload["confidence_score"] == pytest.approx(0.87, abs=1e-6)
    assert payload["count"] == 42


def test_analyze_rejects_empty_filename(fake_run_analysis):
    """A filename-less upload never reaches the pipeline.

    It comes back 422, not the handler's own 400: a multipart part with no
    filename is parsed as a plain form field rather than an UploadFile, so
    FastAPI's `File(...)` validation rejects it before the body of
    analyze_document runs. That makes the `if not document.filename` guard in
    src/api.py unreachable over HTTP - it is kept as a defensive check for
    direct calls, but the status code clients actually observe is 422.
    """
    response = client.post(
        "/analyze",
        files={"document": ("", b"data", "text/plain")},
        data={"query": "What is the revenue?"},
    )

    assert response.status_code == 422
    assert fake_run_analysis == []


def test_analyze_requires_document():
    """The document field is mandatory - FastAPI validation rejects the request."""
    response = client.post("/analyze", data={"query": "What is the revenue?"})

    assert response.status_code == 422


def test_analyze_requires_query():
    """The query field is mandatory."""
    response = client.post(
        "/analyze",
        files={"document": ("filing.txt", b"data", "text/plain")},
    )

    assert response.status_code == 422


def test_analyze_missing_document_returns_404(monkeypatch):
    """FileNotFoundError from the pipeline maps to 404, not a generic 500."""
    def _raise(document_path, query, use_gpu=None):
        raise FileNotFoundError("Document not found: nope.pdf")

    monkeypatch.setattr(src.api, "run_analysis", _raise)

    response = _post()

    assert response.status_code == 404
    assert "nope.pdf" in response.json()["detail"]


def test_analyze_pipeline_failure_returns_500_without_leaking_internals(monkeypatch):
    """An unexpected failure returns 500 with a generic message - the exception
    text goes to the log, not to the HTTP client."""
    def _raise(document_path, query, use_gpu=None):
        raise RuntimeError("CUDA out of memory at 0x7fff in /home/abhin/secret")

    monkeypatch.setattr(src.api, "run_analysis", _raise)

    response = _post()

    assert response.status_code == 500
    assert response.json()["detail"] == "Pipeline execution failed."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
