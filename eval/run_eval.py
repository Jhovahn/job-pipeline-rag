"""
The eval harness: runs golden.json against the live RAG pipeline and
scores each answer. This is the rerunnable, measurable piece — run it
again after changing chunking, the embedding model, or top_k, and the
score tells you if the change actually helped.

    python -m eval.run_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag import answer_question
from app.retrieve import Retriever

DEFAULT_GOLDEN_PATH = Path(__file__).parent / "golden.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def grade(answer_text: str, citations: list[str], case: dict) -> bool:
    """
    Simple, transparent grading: every expected keyword must appear
    (case-insensitive) somewhere in the answer, AND at least one citation
    must match one of the expected source report prefixes.
    """
    text_lower = answer_text.lower()
    keywords_ok = all(kw.lower() in text_lower for kw in case["expected_keywords"])
    sources_ok = any(
        any(expected in citation for expected in case["expected_sources"])
        for citation in citations
    )
    return keywords_ok and sources_ok


def run(golden_path: Path, index_dir: Path) -> None:
    retriever = Retriever(index_dir)
    golden_cases = json.loads(golden_path.read_text(encoding="utf-8"))

    results = []
    correct = 0
    for case in golden_cases:
        result = answer_question(retriever, case["question"])
        passed = grade(result.text, result.citations, case)
        correct += passed
        results.append({
            "id": case["id"],
            "question": case["question"],
            "answer": result.text,
            "citations": result.citations,
            "passed": passed,
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['id']}: {case['question']}")

    score = correct / len(golden_cases)
    print(f"\nScore: {correct}/{len(golden_cases)} ({score:.0%})")

    RESULTS_PATH.write_text(json.dumps({"score": score, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the golden-question eval suite.")
    parser.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN_PATH),
        help="Path to a golden-question JSON file. Point this at a private, "
             "gitignored file (e.g. eval/golden.local.json) to eval against "
             "your own real reports instead of the public sample set.",
    )
    parser.add_argument("--index-dir", default="index")
    args = parser.parse_args()
    run(Path(args.golden), Path(args.index_dir))
