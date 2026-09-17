"""
Command-line entry point.

    python cli.py ingest                     # build the index from data/reports
    python cli.py ask "which role scored highest?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ingest import build_index
from app.rag import answer_question
from app.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your job-pipeline evaluation reports.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Build the embedding index from a folder of reports.")
    ingest_p.add_argument("--reports-dir", default="data/reports")
    ingest_p.add_argument("--index-dir", default="index")

    ask_p = sub.add_parser("ask", help="Ask a question against the index.")
    ask_p.add_argument("question")
    ask_p.add_argument("--index-dir", default="index")
    ask_p.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "ingest":
        n = build_index(Path(args.reports_dir), Path(args.index_dir))
        print(f"Indexed {n} chunks -> {args.index_dir}")
    elif args.command == "ask":
        retriever = Retriever(Path(args.index_dir))
        result = answer_question(retriever, args.question, top_k=args.top_k)
        print(f"\n[{result.mode}]\n{result.text}\n\nSources: {', '.join(result.citations)}")


if __name__ == "__main__":
    main()
