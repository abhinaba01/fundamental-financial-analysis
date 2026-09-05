"""
Measure what the parallel fan-out actually buys, in wall-clock seconds.

Builds two graphs over the *same* agent instances - one fanning NER, sentiment
and KPI out from START, one chaining them - and times both on the same parsed
document. RAG and synthesis are replaced with no-op sinks so the measurement
isolates the analysis phase and needs no embedding model, vector store, or API
key.

The honest caveat this script exists to expose: PyTorch already parallelizes a
single model's inference across cores, so running three models concurrently on
a CPU contends for the same threads and may not help. Whether fan-out is a
speedup or a wash is an empirical question about the machine it runs on, which
is why this prints a measurement instead of a claim.

Usage:
    python scripts/benchmark_parallel.py
    python scripts/benchmark_parallel.py --document data/samples/AAPL_10K.pdf --runs 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from src.agents.kpi_agent import KPIAgent  # noqa: E402
from src.agents.ner_agent import NERAgent  # noqa: E402
from src.agents.sentiment_agent import SentimentAgent  # noqa: E402
from src.graph.state import GraphState  # noqa: E402
from src.preprocessing.chunker import SemanticChunker  # noqa: E402
from src.preprocessing.cleaner import DocumentCleaner  # noqa: E402
from src.preprocessing.parser import DocumentParser  # noqa: E402

ANALYSIS_NODES = ("ner", "sentiment", "kpi")


def _sink(state: GraphState) -> dict:
    """Terminal no-op standing in for retrieve/generate/synthesis."""
    return {}


def _build(agents: dict, parallel: bool):
    """Build an analysis-only graph, fanned out or chained."""
    graph = StateGraph(GraphState)
    for name in ANALYSIS_NODES:
        graph.add_node(name, agents[name])
    graph.add_node("sink", _sink)

    if parallel:
        for name in ANALYSIS_NODES:
            graph.add_edge(START, name)
            graph.add_edge(name, "sink")
    else:
        graph.add_edge(START, ANALYSIS_NODES[0])
        for earlier, later in zip(ANALYSIS_NODES, ANALYSIS_NODES[1:]):
            graph.add_edge(earlier, later)
        graph.add_edge(ANALYSIS_NODES[-1], "sink")

    graph.add_edge("sink", END)
    return graph.compile()


def _initial_state(document) -> dict:
    return {
        "document": document,
        "query": "What is the revenue and gross margin?",
        "retrieved_chunks": [],
        "retrieval_score": 0.0,
        "retry_count": 0,
        "rag_attempts": 0,
        "ner_results": {},
        "sentiment_results": {},
        "kpi_results": {},
        "cot_reasoning": "",
        "final_answer": "",
        "report": {},
    }


def _time_runs(graph, state: dict, runs: int, device: str) -> list[float]:
    """
    Time repeated graph invocations.

    CUDA kernel launches are asynchronous, so on GPU the wall clock has to be
    fenced with synchronize() at both ends. Without it the timer measures how
    fast Python queued the work, not how long the GPU took - which typically
    reports an impossibly large speedup.
    """
    import torch

    synchronize = device == "cuda" and torch.cuda.is_available()

    timings = []
    for _ in range(runs):
        if synchronize:
            torch.cuda.synchronize()
        start = time.perf_counter()
        graph.invoke(state)
        if synchronize:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", default="data/samples/medium_filing.txt")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per topology")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help=(
            "Cap torch intra-op threads. Default lets torch take ~all cores per "
            "model, which oversubscribes badly when three branches run at once. "
            "CPU only - irrelevant when the models are on the GPU."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Run NER and sentiment on CUDA. The fan-out question is different "
            "here: the branches share one device rather than competing for CPU "
            "threads, so whether concurrency helps is an open measurement."
        ),
    )
    args = parser.parse_args()

    import torch

    if args.gpu and not torch.cuda.is_available():
        print("--gpu given but torch.cuda.is_available() is False.")
        return 1

    device = "cuda" if args.gpu else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        print(f"torch intra-op threads capped at {args.torch_threads}")

    document_path = Path(args.document)
    if not document_path.exists():
        print(f"Document not found: {document_path}")
        return 1

    print(f"Document: {document_path}")
    print("Parsing...")
    doc = SemanticChunker().chunk(DocumentCleaner().clean(DocumentParser().parse(document_path)))
    print(f"  {len(doc.chunks)} chunks, {len(doc.cleaned_text)} chars")

    print(f"Loading models on {device} (once, shared by both topologies)...")
    agents = {
        "ner": NERAgent(device=device),
        "sentiment": SentimentAgent(device=device),
        "kpi": KPIAgent(),  # regex, no device
    }

    state = _initial_state(doc)
    graphs = {"sequential": _build(agents, parallel=False), "parallel": _build(agents, parallel=True)}

    # Warm up once per topology so neither pays lazy-init costs in its timings.
    print("Warming up...")
    for graph in graphs.values():
        graph.invoke(state)

    results = {}
    for name, graph in graphs.items():
        print(f"Timing {name} ({args.runs} runs)...")
        timings = _time_runs(graph, state, args.runs, device)
        results[name] = statistics.median(timings)
        print(f"  runs: {', '.join(f'{t:.2f}s' for t in timings)}")

    seq, par = results["sequential"], results["parallel"]
    threads = args.torch_threads or ("n/a" if device == "cuda" else torch.get_num_threads())
    print("\n" + "=" * 60)
    print(f"device: {device}   torch intra-op threads: {threads}")
    print(f"sequential (median): {seq:.2f}s")
    print(f"parallel   (median): {par:.2f}s")
    if par < seq:
        print(f"speedup: {seq / par:.2f}x  ({seq - par:.2f}s saved)")
    else:
        print(f"NO SPEEDUP: parallel is {par / seq:.2f}x the sequential time")
        if device == "cpu":
            print("Expected when torch takes ~all cores per model: three branches")
            print("oversubscribe the CPU. Retry with --torch-threads <cores/3>.")
        else:
            print("The branches share one device, so concurrency has less to win")
            print("than on CPU - the GPU serializes much of the work regardless.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
