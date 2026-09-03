# Running This Project — Step by Step

This is a hands-on walkthrough, written from an actual run on Windows. For
the reference docs (config options, output schema, architecture), see
[README.md](README.md). This file just gets you from a clean checkout to a
working report.

## 1. Prerequisites

- Python 3.10+ (this repo was run and tested on 3.10.10)
- ~2GB free disk for model downloads (NER + sentiment + embedding models,
  all pulled from HuggingFace on first use)
- Windows, macOS, or Linux — commands below show both PowerShell and
  bash/macOS/Linux where they differ

You do **not** need a GPU. Everything in this guide runs on CPU; it's just
slower (see timings in step 5).

## 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash):**
```bash
python -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks the activation script with an execution-policy error,
run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or
just call the venv's Python directly without activating:
`.venv\Scripts\python.exe -m ...` for every command below.

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

This installs `torch`, `transformers`, `chromadb`, `langgraph`, and friends.
Expect this to take several minutes and pull a few GB — `torch` alone is
large. The `spacy` model download is small and separate because spaCy
doesn't ship models as regular pip packages.

## 4. Set up your API key (optional but recommended)

The pipeline runs without any API key — the RAG agent falls back to
extractive synthesis (it stitches together the most relevant retrieved
text instead of generating a real answer). For actual chain-of-thought
answers from `gpt-4o`, add an OpenAI key:

```bash
cp .env.example .env
```

Then open `.env` and fill in:
```
OPENAI_API_KEY=sk-...
```

`.env` is gitignored — it will never be committed. Everything else
(NER, sentiment, KPI extraction, embeddings) runs fully local with no key.

## 5. Run it on the sample document

The repo ships a couple of tiny test documents so you don't need a real
10-K on hand for a first run.

```bash
python -m src.main --document test_small.txt --query "What is the revenue and gross margin?" --output report.json --cpu
```

**What happens, in order, and roughly how long each stage takes on CPU:**

1. **Parse** the document (instant for `.txt`/`.json`; PDFs take longer)
2. **Clean** the text (instant)
3. **Chunk** it into token-bounded pieces (instant to a few seconds,
   depending on length — spaCy sentence splitting runs here)
4. **Embed** each chunk with `BAAI/bge-large-en-v1.5` and index it into a
   local ChromaDB store at `data/vector_store/` — **this is the slow part
   on a first run**: the model itself is ~1.3GB and downloads once, then
   gets cached in `~/.cache/huggingface/hub/`. Budget 3-5 minutes the very
   first time; seconds on every run after
5. **Run the analysis graph**: NER (`dslim/bert-base-NER`), sentiment
   (`ProsusAI/finbert`), KPI extraction (regex-based, no model), then RAG
   (retrieval + generation). NER and sentiment models are smaller and
   download in under a minute the first time
6. **Synthesize** everything into a JSON report

A cold run (nothing cached yet) takes on the order of 5-8 minutes, almost
all of it one-time model downloads. A warm run (models cached) finishes in
under a minute for a short document.

Open `report.json` when it's done — it has `named_entities`,
`sentiment_analysis`, `financial_metrics`, `reasoning` (the RAG answer with
chain-of-thought if you set an API key), and a `summary` string.

## 6. Run it on a real document

Point `--document` at any `.pdf`, `.txt`, `.html`, or `.json` file:

```bash
python -m src.main --document AAPL_10K.pdf --query "What are the primary risk factors?" --output aapl_report.json --cpu
```

PDFs take noticeably longer to parse than text — `pdfplumber` extracts text
and tables page by page. A full 10-K (100+ pages) can take a few minutes
just to parse and chunk, on top of the model time above.

Drop `--cpu` if you have a CUDA GPU set up; the flag defaults to GPU
otherwise, which will fail loudly if no GPU is available.

## 7. Run the HTTP API instead of the CLI

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then, from another terminal:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "query=What are the key risks?" \
  -F "use_gpu=false" \
  -F "document=@test_small.txt"
```

The response is the same JSON structure as the CLI's `report.json`.

## 8. Run the test suite

```bash
pytest tests/ -v
```

Should show `19 passed`. The first run loads the NER and sentiment models
(same one-time download cost as step 5), so it isn't instant, but it
doesn't touch the embedding model or ChromaDB.

## 9. Run the evaluation harnesses

Each agent has a small worked example under `data/eval/` to confirm the
scoring code itself works, independent of model quality:

```bash
python -m evaluation.eval_ner --test-set data/eval/ner_example.json
python -m evaluation.eval_sentiment --test-set data/eval/sentiment_example.json
python -m evaluation.eval_kpi --test-set data/eval/kpi_example.json
python -m evaluation.eval_rag --test-set data/eval/rag_example.json
```

Add `--run-agent` to NER/sentiment/KPI to score live model output instead of
the hand-written predictions baked into those example files (see
[README.md](README.md#measured-results-this-repo) for what that actually
produces on this repo right now). `eval_rag --run-agent` additionally needs
a document already indexed via step 5/6 and `OPENAI_API_KEY` set, since it
retrieves from the same persistent ChromaDB collection.

## 10. Run it in Docker instead

```bash
docker build -t financial-fundamentals-analysis .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$(pwd)/data/vector_store:/app/data/vector_store" \
  financial-fundamentals-analysis
```

The volume mount persists the vector store across container restarts, so
you don't re-embed everything every time you `docker run`.

## Troubleshooting things you'll actually hit

**"No chunks above similarity threshold" / RAG says "Unable to answer"**
The retrieval similarity threshold and the re-query retry count are hardcoded
constants (not the `configs/*.yaml` files — those aren't wired up, see the
README's Configuration section), in `src/graph/edges.py` and
`src/agents/rag_agent.py`. If you've tuned them and answers still come back
empty, check the log line `Retrieved N chunks with avg similarity: X.XXX` —
BGE cosine similarities for genuinely relevant text usually land around
0.5-0.65, not near 1.0.

**Symlink warning from `huggingface_hub` on Windows**
Harmless. Models still download and load fine; you just don't get the
disk-space savings from symlinked cache dedup. Enable Developer Mode or run
as Administrator if you want to silence it.

**First run feels stuck for minutes with no output**
It's downloading models, not hanging. `BAAI/bge-large-en-v1.5` alone is
~1.3GB. Subsequent runs use the HuggingFace cache
(`~/.cache/huggingface/hub/`) and are fast.

**Re-running against the same document twice**
`data/vector_store/` is a persistent ChromaDB collection — chunks are
upserted by ID, not appended, so re-indexing the same document won't
duplicate entries. Delete `data/vector_store/` if you want a clean slate
(e.g. after changing the chunking or embedding config).

**Two pipeline runs at the same time hang instead of failing**
ChromaDB's persistent client isn't safe for concurrent writers against the
same `data/vector_store/` path from separate processes — one process can
sit blocked on a file lock rather than erroring out. Run one at a time
against a given vector store path, or point each run at a different
`vector_store_path`.

**`OPENAI_API_KEY not set` warning, RAG answer looks like raw chunk text**
Expected without a key — see step 4. The pipeline still runs and produces a
full report; the RAG `reasoning`/`final_answer` fields are just extractive
(concatenated chunk text) instead of LLM-generated.
