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
from app.structured import (
    by_risk_level,
    highest_scoring,
    load_all_summaries,
    lowest_scoring,
)


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

    stats_p = sub.add_parser("stats", help="Exact/structured lookups over parsed Machine Summary YAML.")
    stats_p.add_argument("--reports-dir", default="data/reports")
    stats_p.add_argument("--highest", action="store_true", help="Show the highest-scoring report.")
    stats_p.add_argument("--lowest", action="store_true", help="Show the lowest-scoring report.")
    stats_p.add_argument("--risk-level", help="List all reports at this risk level (e.g. Medium).")

    args = parser.parse_args()

    if args.command == "ingest":
        n = build_index(Path(args.reports_dir), Path(args.index_dir))
        print(f"Indexed {n} chunks -> {args.index_dir}")
    elif args.command == "ask":
        retriever = Retriever(Path(args.index_dir))
        result = answer_question(retriever, args.question, top_k=args.top_k)
        print(f"\n[{result.mode}]\n{result.text}\n\nSources: {', '.join(result.citations)}")
    elif args.command == "stats":
        summaries = load_all_summaries(Path(args.reports_dir))
        if args.highest:
            top = highest_scoring(summaries)
            print(f"Highest: {top['company']} — {top['score']}/5 ({top['_source']})" if top else "No scored reports found.")
        elif args.lowest:
            bottom = lowest_scoring(summaries)
            print(f"Lowest: {bottom['company']} — {bottom['score']}/5 ({bottom['_source']})" if bottom else "No scored reports found.")
        elif args.risk_level:
            matches = by_risk_level(summaries, args.risk_level)
            for m in matches:
                print(f"{m['company']} — {m.get('score', '?')}/5 ({m['_source']})")
            if not matches:
                print(f"No reports found at risk level '{args.risk_level}'.")
        else:
            print(f"{len(summaries)} reports parsed. Use --highest, --lowest, or --risk-level.")


if __name__ == "__main__":
    main()
