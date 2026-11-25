"""Ingest a folder, ask one question, or start the API."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from documind.config import get_settings
from documind.graph.builder import build_graph, run_question
from documind.ingestion.pipeline import ingest_directory
from documind.llm.providers import build_model
from documind.logging_utils import configure_logging
from documind.retrieval.store import CorpusStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_logging()
    parser = argparse.ArgumentParser(prog="documind")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a PDF and Markdown directory")
    ingest.add_argument("path")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")

    sub.add_parser("serve", help="run the FastAPI server")
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "serve":
        uvicorn.run(
            "documind.api.app:app",
            host=settings.host,
            port=settings.port,
            reload=False,
        )
        return

    store = CorpusStore(settings)
    if args.command == "ingest":
        summary = ingest_directory(Path(args.path), store, settings)
        print(f"indexed {len(summary.documents)} files, {summary.chunk_count} chunks")
        return

    if store.chunk_count() == 0 and settings.corpus_dir.exists():
        ingest_directory(settings.corpus_dir, store, settings)
    graph = build_graph(settings, store, build_model(settings))
    state = run_question(graph, args.question)
    print(state.get("answer") or "")
    if state.get("low_confidence"):
        print("(low confidence)")


if __name__ == "__main__":
    main()
