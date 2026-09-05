# Financial Fundamentals Analysis System

[![tests](https://github.com/abhinaba01/fundamental-financial-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/abhinaba01/fundamental-financial-analysis/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A LangGraph-based NLP pipeline for financial document analysis: it parses a
filing, indexes it for retrieval, and runs a graph of specialized agents (NER,
sentiment, KPI extraction, RAG) to produce a single structured JSON report.

## Overview

This system extracts, analyzes, and synthesizes insights from financial documents using:
- **Named Entity Recognition** (NER): General-purpose entity extraction via `dslim/bert-large-NER` (ORG/PER/LOC/MISC - not finance-tuned; see [Measured Results](#measured-results-this-repo) below for why)
- **Sentiment Analysis**: ProsusAI/finbert, plus a tone pass (the tone model
  currently fails to load and falls back to FinBERT — see [Configuration](#agent-configuration-configsagentsyaml))
- **KPI Extraction**: Regex-based financial metric extraction with derived calculations
- **Retrieval-Augmented Generation** (RAG): Evidence-based synthesis with re-query capability
- **Report Synthesis**: Structured output combining all analysis results

## Architecture

```
Document (PDF/HTML/TXT)
    ↓
Parser → Cleaner → Chunker → Embedder
    ↓                          ↓
ParsedDocument         ChromaDB Vector Store
    ↓
LangGraph Pipeline:
    ├─ NER Agent (dslim/bert-large-NER)
    ├─ Sentiment Agent (ProsusAI/finbert)
    ├─ KPI Agent (regex-based extraction, not an LLM)
    ├─ RAG Agent (BAAI/bge-large-en-v1.5 + gpt-4o)
    │   └─ [Re-query if <3 chunks OR similarity <0.5]
    └─ Synthesis Agent
    ↓
Final Report (JSON)
```

## Installation

### Requirements
- Python 3.10+ (developed and tested on 3.10.10)
- CUDA 12.1+ (optional - the pipeline auto-detects a GPU and falls back to CPU)
- 16GB+ RAM
- ~3.5GB disk for model weights

### Setup

```bash
# Clone repository
git clone https://github.com/abhinaba01/fundamental-financial-analysis.git
cd fundamental-financial-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project (add [dev] for the test suite)
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# Optional: pre-download model weights (~3.5GB) instead of paying
# the download cost mid-run on first use
python scripts/download_models.py
```

For a step-by-step walkthrough from a clean checkout — including timings,
Docker, Colab, and the errors you are likely to hit — see [RUNNING.md](RUNNING.md).

## Quick Start

### CLI Usage

Sample filings ship with the repo under `data/samples/`, so a first run needs
nothing downloaded:

```bash
python -m src.main \
  --document data/samples/AAPL_10K.pdf \
  --query "What are the main business risks?" \
  --output report.json
```

`data/samples/small_filing.txt` is a much smaller document if you just want to
watch the pipeline run end to end quickly.


### Programmatic Usage

```python
from src.main import run_analysis

report = run_analysis(
    document_path="data/samples/AAPL_10K.pdf",
    query="What is the revenue trend over the last 3 years?"
)

print(report["summary"])
```

## HTTP API Server

A lightweight FastAPI wrapper is available in `src/api.py`.

### Run locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "query=What are the key risks?" \
  -F "use_gpu=false" \
  -F "document=@data/samples/AAPL_10K.pdf"
```

### Response

The API returns the same JSON report structure used by the CLI.

## Docker Deployment

A Dockerfile is included for containerized deployment.

### Build image

```bash
docker build -t financial-fundamentals-analysis .
```

### Run container

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$(pwd)/data/vector_store:/app/data/vector_store" \
  financial-fundamentals-analysis
```

The API will be served at `http://0.0.0.0:8000`.

## Configuration

### Pipeline Configuration (`configs/pipeline.yaml`)

- **Chunking**: 512 tokens max, 64 token overlap
- **Embedding Model**: BAAI/bge-large-en-v1.5 (1024-dim)
- **Vector Store**: ChromaDB (persistent at `data/vector_store/`)
- **Retrieval**: Top-5 chunks, 0.5 cosine similarity threshold

### Agent Configuration (`configs/agents.yaml`)

`configs/*.yaml` document intent but aren't loaded at runtime - the active
values are the constants at the top of each agent module (see the
Troubleshooting section below).

- **Models**: Hardcoded for reproducibility
- **NER**: dslim/bert-large-NER (general-purpose, not finance-tuned)
- **Sentiment**: ProsusAI/finbert (main) + yiyanghkust/finbert-tone (tone) —
  in practice the tone model **does not load** on current `transformers`
  versions (its `config.json` has no `model_type` key), so `SentimentAgent`
  logs a warning and falls back to running the primary model for tone as
  well. Tone output is therefore FinBERT sentiment, not a distinct signal,
  until that model is replaced.
- **KPI**: Regex-based extraction (`src/agents/kpi_agent.py`), not an LLM
- **RAG**: gpt-4o (requires OPENAI_API_KEY)

Set `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

## Output Format

### Report Structure

```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "document_id": "AAPL_10K_2023",
    "ticker": "AAPL",
    "doc_type": "10-K",
    "query": "..."
  },
  "retrieval": {
    "chunks_retrieved": 5,
    "average_similarity": 0.82,
    "retries_performed": 1
  },
  "named_entities": {
    "total_entities": 42,
    "entity_types": {"ORG": 15, "PER": 8, ...}
  },
  "sentiment_analysis": {
    "overall_sentiment": "neutral",
    "confidence_score": 0.87,
    "sentiment_distribution": {"positive": 0.35, ...}
  },
  "financial_metrics": {
    "total_kpis": 12,
    "extracted_kpis": {"revenue": 394328000000, ...},
    "calculated_kpis": {"gross_margin_pct": 46.2}
  },
  "reasoning": {
    "chain_of_thought": "Step 1: ...",
    "final_answer": "..."
  },
  "summary": "..."
}
```

## Evaluation

Each agent has a benchmark harness under `evaluation/`. All four expose the same CLI:

```bash
python -m evaluation.eval_ner --test-set data/eval/ner_example.json
```

```bash
python -m evaluation.eval_sentiment --test-set data/eval/sentiment_example.json
```

```bash
python -m evaluation.eval_rag --test-set data/eval/rag_example.json
```

```bash
python -m evaluation.eval_kpi --test-set data/eval/kpi_example.json
```

The files in `data/eval/` are small worked examples of each format, not the real
test splits. Point `--test-set` at a converted FiNER-139 / Financial PhraseBank /
FinanceBench / FinQA split to reproduce the benchmark numbers below.

### Flags

| Flag | Effect |
|------|--------|
| `--test-set PATH` | JSON test set (required) |
| `--output PATH` | Write the computed metrics to a JSON file |
| `--limit N` | Score only the first N samples |
| `--benchmark` | Also print the published reference numbers for the dataset |
| `--run-agent` | Load the agent and generate predictions, instead of reading them from the test set |

Metrics print to stdout; `--output` additionally writes them as JSON. Exit code is
`1` with a one-line message if the test set is missing, malformed, or empty.

### Test-set format

Every module accepts three container shapes — a bare list of samples, `{"samples": [...]}`,
or a single sample object. Sample keys per module:

```jsonc
// eval_ner.py    "entities" aliases "references"
{"text": "...", "references": [{"word": "Apple Inc.", "entity": "ORG"}],
                "predictions": [{"word": "Apple Inc.", "entity": "ORG"}]}

// eval_sentiment.py    "label" aliases "sentiment"; classes: positive|neutral|negative
{"text": "...", "sentiment": "positive", "prediction": "positive"}

// eval_kpi.py    "gold_kpis" aliases "reference_kpis"
{"text": "...", "reference_kpis": {"revenue": 383285.0},
                "extracted_kpis": {"revenue": 383285.0},
                "calculation_steps": [{"operands": [169148, 383285], "operator": "/",
                                       "expected_result": 0.4413}]}

// eval_rag.py    chunks may be strings or {"chunk_id", "text"} objects
{"question": "...", "answer": "...", "gold_chunks": [...],
                    "retrieved_chunks": [...], "generated_answer": "..."}
```

The prediction fields (`predictions`, `prediction`, `extracted_kpis`,
`retrieved_chunks`/`generated_answer`) are optional. Omit them and pass
`--run-agent` to generate predictions live; omit them without `--run-agent` and
those samples score as misses, with a warning.

### Live agent mode

`--run-agent` loads real models, so it has real prerequisites:

- **NER / Sentiment / KPI** — run standalone. NER and Sentiment download their
  HuggingFace checkpoints on first use.
- **RAG** — retrieves from the persistent ChromaDB collection, so the corpus must
  already be indexed (run `python -m src.main` once) and `OPENAI_API_KEY` must be
  set. Without those, score a test set that already carries `retrieved_chunks`
  and `generated_answer`.

Scoring is macro-averaged across samples for `eval_kpi` and `eval_rag`; `eval_ner`
pools entities across samples (namespaced per sample so identical surface forms in
different samples cannot cross-match).

### Measured Results (this repo)

Test suite: **32/32 passed** (`pytest tests/`), split as:

- `tests/test_pipeline.py` (20) — 13 smoke tests plus 7 regression tests, each
  pinning a bug found and fixed during a pipeline audit: KPI regex group
  misalignment, gross-margin/dollar-amount conflation, smart-quote
  normalization, financial-notation word-boundary corruption, retrieved-chunk
  id/type coercion, the `EmbeddingPipeline` constructor signature, and a
  chunker infinite loop on any document over ~512 tokens.
- `tests/test_api.py` (12) — HTTP contract for the FastAPI wrapper: status
  codes, upload handling, argument pass-through, and numpy serialization, with
  `run_analysis` monkeypatched so no models load.

Evaluation CLIs run against the worked examples in `data/eval/` (2-6 samples
each — sanity checks that the harness and agents work end-to-end, **not** a
reproduction of the published benchmarks below, which are measured on the full
external datasets):

| Module | Mode | Result |
|--------|------|--------|
| `eval_ner` | `--run-agent` (live `dslim/bert-large-NER`) | Precision 0.44, Recall 0.67, F1 0.53 |
| `eval_sentiment` | `--run-agent` (live `ProsusAI/finbert`) | Accuracy 1.00, Macro-F1 1.00 |
| `eval_kpi` | `--run-agent` (live regex `KPIAgent`) | Numeric accuracy 0.83, Extraction recall 0.83 |
| `eval_rag` | offline (hand-scored `retrieved_chunks`/`generated_answer`) | EM 0.50, ROUGE-L 0.50, Hit@5 1.00 |

The live NER run scores 0 on the `STOCK_EXCHANGE` entity type regardless of
model size (`bert-base` and `bert-large` score identically on this 3-sample
check) because neither `dslim/bert-*-NER` variant has that label at all — see
`src/agents/ner_agent.py`'s module docstring for why there's no finance-tuned
alternative for this entity set as of writing. `bert-large` does extract
cleaner entity spans on messier real-document text (e.g. correctly capturing
"Apple Inc." as one span instead of splitting it), just not something this
tiny synthetic sample measures.

## Roadmap / Future Work

Where this pipeline currently stands against what the literature achieves on
the same tasks, and what closing each gap would actually require. **Nothing in
this section is a measurement of this repo** — see [Measured Results](#measured-results-this-repo)
above for that.

**1. Financial NER (XBRL figure tagging).** `NERAgent` tags ORG/PER/LOC/MISC
with a general-purpose model. The finance-specific task — tagging *numeric
tokens* with XBRL accounting concepts, e.g. deciding whether a figure is
`Revenues` or `NetIncomeLoss` — is a genuinely different problem, closer to
what `KPIAgent` does than to named entity recognition. The FiNER-139 paper
reports 89.2% F1 for a fine-tuned `sec-bert-base` variant; the published
`sec-bert-base` is a base language model with no NER head, so this needs an
actual fine-tune, not a model swap. Work in progress on the
`feature/financial-ner` branch.

**2. LLM-based KPI reasoning.** `KPIAgent` is regex-based. Multi-step
numerical reasoning over filings (the FinQA task, where models like Qwen2.5-7B
reach 80%+) would need a model in the loop with a calculation tool, replacing
pattern matching for anything beyond direct figure extraction.

**3. Benchmarking on full datasets.** The evaluation harnesses run against
2-6 sample worked examples. Pointing them at converted Financial PhraseBank
(finbert reference: ~97% accuracy) and FinanceBench (gpt-4o + BGE reference:
~75% EM) splits would produce numbers comparable to published results — the
harnesses already accept these formats, the datasets just aren't vendored here.

**4. Config files that actually load.** `configs/*.yaml` currently document
intent only; thresholds live as constants in `src/graph/edges.py` and
`src/agents/rag_agent.py`. Wiring the YAML through would remove a real
footgun (see [Troubleshooting](#re-query-loops)).

## Advanced Usage

### Custom Document Processing

```python
from src.preprocessing.parser import DocumentParser
from src.preprocessing.embedder import EmbeddingPipeline

# Parse
parser = DocumentParser()
doc = parser.parse("document.pdf")

# Embed
embedder = EmbeddingPipeline()
embedder.embed_and_index(doc)

# Query
results = embedder.retrieve("What is the revenue?", top_k=5)
```

### Custom Agent Pipeline

Every agent node can be swapped for your own. Each is a plain callable taking
and returning a `GraphState`, so anything with a `__call__(state) -> state` works;
agents left as `None` are constructed with their defaults.

```python
from src.graph.builder import AnalysisPipelineBuilder

class MyKPIAgent:
    def __call__(self, state):
        state["kpi_results"] = {...}
        return state

builder = AnalysisPipelineBuilder(
    embedding_pipeline=embedder,
    kpi_agent=MyKPIAgent(),   # ner_agent, sentiment_agent, rag_agent,
    device="cpu",             # and synthesis_agent are also overridable
)
graph = builder.build()
report = graph.invoke(initial_state)["report"]
```

## Troubleshooting

### Out of Memory
```bash
# Use CPU instead of GPU
python -m src.main --cpu --document file.pdf --query "..."
```

### Re-query Loops
`configs/*.yaml` are reference documentation only — they are not loaded at
runtime. The active thresholds are the `SIMILARITY_THRESHOLD` and
`MIN_CHUNKS_THRESHOLD` constants at the top of `src/graph/edges.py` and
`src/agents/rag_agent.py` (keep both files in sync if you change them).

The default `SIMILARITY_THRESHOLD` is `0.5`. BAAI/bge-large-en-v1.5 cosine
similarities for genuinely relevant chunks typically land around 0.5-0.65,
not near 1.0, so raising this much above ~0.65 will send most queries into
the retry loop regardless of retrieval quality.

### Slow Embedding
The first run downloads `BAAI/bge-large-en-v1.5` (~1.3GB); run
`python scripts/download_models.py` up front to get that out of the way.
After that, embedding batch size is the `BATCH_SIZE` constant in
`src/preprocessing/embedder.py` — there is no CLI flag for it.

## File Structure

```
├── src/
│   ├── preprocessing/
│   │   ├── parser.py          # PDF/HTML/TXT parsing
│   │   ├── cleaner.py         # Text normalization
│   │   ├── chunker.py         # Semantic chunking
│   │   ├── embedder.py        # Embedding + ChromaDB
│   │   └── document.py        # Data structures
│   ├── agents/
│   │   ├── ner_agent.py       # NER node
│   │   ├── sentiment_agent.py  # Sentiment node
│   │   ├── kpi_agent.py        # KPI node
│   │   ├── rag_agent.py        # RAG node (retrieval + generation)
│   │   └── synthesis_agent.py  # Report assembly
│   ├── graph/
│   │   ├── state.py           # GraphState TypedDict
│   │   ├── edges.py           # Conditional routing
│   │   └── builder.py         # StateGraph assembly
│   ├── utils/
│   │   └── logger.py          # Logging utility
│   └── main.py                # Entry point
├── evaluation/
│   ├── _cli.py                # Shared CLI plumbing / test-set loading
│   ├── eval_ner.py            # FiNER-139 evaluation
│   ├── eval_sentiment.py       # Financial PhraseBank
│   ├── eval_rag.py            # FinanceBench
│   └── eval_kpi.py            # FinQA
├── configs/
│   ├── agents.yaml            # Reference only - not loaded at runtime
│   └── pipeline.yaml          # Reference only - not loaded at runtime
├── scripts/
│   └── download_models.py     # Pre-download model weights
├── tests/
│   ├── test_pipeline.py       # Preprocessing + agent unit/regression tests
│   └── test_api.py            # FastAPI endpoint contract tests
├── .github/workflows/
│   └── tests.yml              # CI: pytest on 3.10 and 3.11
└── data/
    ├── samples/               # Sample filings, committed (see Quick Start)
    ├── raw/                   # Your own input documents (gitignored)
    ├── processed/             # Intermediate (gitignored)
    ├── eval/                  # Example test sets for the eval CLIs
    └── vector_store/          # ChromaDB persistence (gitignored)
```

## Model Lock-in

All models are hardcoded for reproducibility. To modify:

1. **NER Model**: `MODEL_NAME` in `src/agents/ner_agent.py`
2. **Sentiment Models**: `PRIMARY_MODEL` / `SECONDARY_MODEL` in `src/agents/sentiment_agent.py`
3. **Embedding Model**: `MODEL_NAME` in `src/preprocessing/embedder.py`
4. **LLM Model**: `GENERATION_MODEL` in `src/agents/rag_agent.py` — *not*
   `configs/agents.yaml`, which is documentation only and is never loaded at runtime

## API Keys

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."  # For RAG generation
export HF_TOKEN="hf_..."         # For HuggingFace model access
```

## Development

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest tests/ -v
```

CI runs the same suite on Python 3.10 and 3.11 for every push and pull request
to `main` (see [.github/workflows/tests.yml](.github/workflows/tests.yml)).

Branches:
- `main` — everything documented here, tests green
- `feature/financial-ner` — XBRL figure tagging, blocked on training and
  publishing the fine-tuned model (see [Roadmap](#roadmap--future-work))

## License

MIT — see [LICENSE](LICENSE).

## Support

Bug reports and questions: [GitHub Issues](https://github.com/abhinaba01/fundamental-financial-analysis/issues).
