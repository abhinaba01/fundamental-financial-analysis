# Financial Fundamentals Analysis System

A production-ready LangGraph-based NLP pipeline for autonomous financial document analysis.

## Overview

This system extracts, analyzes, and synthesizes insights from financial documents using:
- **Named Entity Recognition** (NER): General-purpose entity extraction via `dslim/bert-large-NER` (ORG/PER/LOC/MISC - not finance-tuned; see [Measured Results](#measured-results-this-repo) below for why)
- **Sentiment Analysis**: ProsusAI/finbert + tone detection
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
- Python 3.11+
- CUDA 12.1+ (recommended for GPU acceleration)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone <repo_url>
cd financial-fundamental-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (first run, ~30GB total)
python scripts/download_models.py
```

## Quick Start

### CLI Usage

```bash
python -m src.main \
  --document data/raw/AAPL_10K_2023.json \
  --query "What are the main business risks?" \
  --output report.json
```


### Programmatic Usage

```python
from src.main import run_analysis

report = run_analysis(
    document_path="data/raw/AAPL_10K_2023.json",
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
  -F "document=@data/raw/AAPL_10K_2023.json"
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
- **Sentiment**: ProsusAI/finbert (main) + yiyanghkust/finbert-tone (tone)
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

Test suite: **19/19 passed** (`pytest tests/`) — 13 smoke tests plus 6 regression
tests added for bugs found and fixed during a pipeline audit (KPI regex group
misalignment, gross-margin/dollar-amount conflation, smart-quote normalization,
financial-notation word-boundary corruption, retrieved-chunk id/type coercion,
and the `EmbeddingPipeline` constructor signature).

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
tiny synthetic sample measures. Separately, `eval_kpi`'s live run uses the
regex-based `KPIAgent` in `src/agents/kpi_agent.py`, not the `Qwen2.5-7B` LLM
described in the benchmark table below - reproducing that table's numbers
requires actually wiring up the models it names, not just pointing
`--test-set` at the real datasets.

## Performance Benchmarks

Published reference numbers for the datasets and models named — **not**
measurements of this repo's current agents. See "Measured Results" above for
what the code as it stands actually produces, and the note above the table
for where the two diverge (NER and KPI use different models than named here).

The NER row is a bigger mismatch than "different model": FiNER-139 tags
*numeric tokens* with 139 XBRL accounting concepts (e.g. is this figure
"Revenue" or "NetIncomeLoss") - a different task from the ORG/PER/LOC/MISC
entity tagging `NERAgent` actually does, closer in spirit to `KPIAgent`'s
job than to named entity recognition. `sec-bert-base` itself is a base
language model with no NER head; the 89.2% figure describes a fine-tuned
variant from the FiNER-139 paper, not a model you can drop in as-is.

| Component | Model | Benchmark | Expected |
|-----------|-------|-----------|----------|
| NER | sec-bert-base | FiNER-139 F1 | 89.2% |
| Sentiment | finbert | Financial PhraseBank Acc | 97% |
| RAG EM | gpt-4o + BGE | FinanceBench | ~75% |
| KPI Accuracy | Qwen2.5-7B | FinQA | 80%+ |

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

```python
from src.graph.builder import AnalysisPipelineBuilder
from src.agents.custom_agent import MyCustomAgent

builder = AnalysisPipelineBuilder(
    custom_agents={"my_agent": MyCustomAgent()}
)
graph = builder.build()
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
```bash
# Reduce batch size
python -m src.main --batch-size 8 --document file.pdf --query "..."
```

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
│   ├── agents.yaml            # Agent configuration
│   └── pipeline.yaml          # Pipeline settings
├── tests/
│   └── test_pipeline.py       # Basic tests
└── data/
    ├── raw/                   # Input documents
    ├── processed/             # Intermediate
    ├── eval/                  # Example test sets for the eval CLIs
    └── vector_store/          # ChromaDB persistence
```

## Model Lock-in

All models are hardcoded for reproducibility. To modify:

1. **NER Model**: Edit `src/agents/ner_agent.py`
2. **Sentiment Models**: Edit `src/agents/sentiment_agent.py`
3. **Embedding Model**: Edit `src/preprocessing/embedder.py`
4. **LLM Models**: Edit `configs/agents.yaml`

## API Keys

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."  # For RAG generation
export HF_TOKEN="hf_..."         # For HuggingFace model access
```

## License

[License type here]

## Contributing

[Contribution guidelines here]

## Citation

[Citation information if applicable]

## Support

For issues or questions:
- Documentation: [Wiki link]
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

---

**Last Updated**: 2024-01-15  
**Status**: Production Ready
