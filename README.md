# Financial Fundamentals Analysis System

A production-ready LangGraph-based NLP pipeline for autonomous financial document analysis.

## Overview

This system extracts, analyzes, and synthesizes insights from financial documents using:
- **Named Entity Recognition** (NER): Entity extraction via `nlpaueb/sec-bert-base`
- **Sentiment Analysis**: ProsusAI/finbert + tone detection
- **KPI Extraction**: Financial metric extraction with safe Python calculations
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
    ├─ NER Agent (nlpaueb/sec-bert-base)
    ├─ Sentiment Agent (ProsusAI/finbert)
    ├─ KPI Agent (Qwen2.5-7B-Instruct)
    ├─ RAG Agent (BAAI/bge-large-en-v1.5 + gpt-4o)
    │   └─ [Re-query if <3 chunks OR similarity <0.75]
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
- **Retrieval**: Top-5 chunks, 0.75 cosine similarity threshold

### Agent Configuration (`configs/agents.yaml`)

- **Models**: Hardcoded for reproducibility
- **NER**: nlpaueb/sec-bert-base
- **Sentiment**: ProsusAI/finbert (main) + yiyanghkust/finbert-tone (tone)
- **KPI**: Qwen2.5-7B-Instruct
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

Run benchmark evaluations against standard datasets:

```bash
# NER evaluation (FiNER-139)
python -m evaluation.eval_ner --test-set data/eval/finer139_test.json

# Sentiment evaluation (Financial PhraseBank)
python -m evaluation.eval_sentiment --test-set data/eval/fpb_test.json

# RAG evaluation (FinanceBench)
python -m evaluation.eval_rag --test-set data/eval/financebench_test.json

# KPI evaluation (FinQA)
python -m evaluation.eval_kpi --test-set data/eval/finqa_test.json
```

## Performance Benchmarks

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
Check `configs/pipeline.yaml` for retrieval thresholds:
```yaml
retrieval:
  similarity_threshold: 0.75  # Increase to reduce re-queries
  min_chunks_threshold: 3     # Reduce to accept fewer chunks
```

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
