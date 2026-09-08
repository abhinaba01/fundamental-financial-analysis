"""
Pre-download every model the pipeline loads at runtime.

Running this is optional - each model downloads itself on first use anyway.
It exists so the download cost (~3.5GB, several minutes on a cold cache) can
be paid up front, deliberately, instead of appearing as an unexplained
multi-minute hang in the middle of a first pipeline run.

Models are loaded through the agents' own constructors rather than by
re-specifying model names and loader classes here. That keeps this script from
drifting when a model is swapped, and means it exercises the exact loading
path the pipeline uses - `transformers.pipeline()` handles tokenizer fallbacks
that a bare `AutoTokenizer.from_pretrained()` does not, so loading it any
other way could report a failure the real pipeline never hits.

The agents swallow load errors by design (they degrade to reduced
functionality rather than crashing the graph), so success is determined by
inspecting the loaded pipeline attribute, not by catching exceptions.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-embedding   # skip the 1.3GB BGE model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root, so
# `import src` fails unless the project was installed with `pip install -e .`.
# Adding the repo root keeps the script working under either install method.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPACY_MODEL = "en_core_web_sm"


def _warm_ner() -> bool:
    """Load NERAgent, populating the cache for its checkpoint."""
    from src.agents.ner_agent import MODEL_NAME, NERAgent

    print(f"\n[ner] {MODEL_NAME}")
    try:
        agent = NERAgent()
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False

    if agent.ner_pipeline is None:
        print("  FAILED: model did not load (see the warning above)")
        return False

    print("  cached")
    return True


def _warm_sentiment() -> bool:
    """Load SentimentAgent, populating the cache for both of its checkpoints."""
    from src.agents.sentiment_agent import PRIMARY_MODEL, SECONDARY_MODEL, SentimentAgent

    print(f"\n[sentiment] {PRIMARY_MODEL}")
    print(f"[tone]      {SECONDARY_MODEL}")
    try:
        agent = SentimentAgent()
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False

    if agent.sentiment_pipeline is None:
        print("  FAILED: primary sentiment model did not load")
        return False

    # On a tone-model load failure SentimentAgent assigns the primary pipeline
    # to tone_pipeline, so a None check would report success either way -
    # identity is what actually distinguishes a real load from the fallback.
    if agent.tone_pipeline is agent.sentiment_pipeline:
        # Not fatal: the agent runs with the primary model doing double duty.
        print("  cached (primary only - tone model unavailable, agent falls back)")
        return True

    print("  cached (both)")
    return True


def _warm_embedding() -> bool:
    """Populate the cache for the embedding model.

    Loads SentenceTransformer directly rather than constructing
    EmbeddingPipeline, which would also spin up a persistent ChromaDB client
    and create data/vector_store/ as a side effect of a download-only script.
    """
    from src.preprocessing.embedder import MODEL_NAME

    print(f"\n[embedding] {MODEL_NAME}")
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False

    print("  cached")
    return True


def _warm_spacy() -> bool:
    """Install the spaCy sentence-splitting model used by SemanticChunker."""
    print(f"\n[spacy] {SPACY_MODEL}")
    try:
        import spacy

        spacy.load(SPACY_MODEL, disable=["ner", "tagger", "lemmatizer"])
        print("  already installed")
        return True
    except Exception:
        pass

    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
        check=False,
    )
    if result.returncode != 0:
        print(f"  FAILED: `python -m spacy download {SPACY_MODEL}` exited {result.returncode}")
        return False

    print("  installed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download pipeline models")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip the ~1.3GB embedding model (only needed to index documents)",
    )
    args = parser.parse_args()

    print("Downloading models into the HuggingFace cache (~/.cache/huggingface).")
    print("Already-cached models are a no-op.")

    results = [_warm_ner(), _warm_sentiment()]

    if args.skip_embedding:
        from src.preprocessing.embedder import MODEL_NAME

        print(f"\n[embedding] {MODEL_NAME} - skipped (--skip-embedding)")
    else:
        results.append(_warm_embedding())

    results.append(_warm_spacy())

    failed = results.count(False)
    print("\n" + "=" * 60)
    if failed:
        print(f"{failed} model(s) failed to load. See the errors above.")
        print("Models also download on demand, so the pipeline may still run.")
        return 1

    print("All models cached. First pipeline run will not need to download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
