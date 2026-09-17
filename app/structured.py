"""
Structured extraction of the `## Machine Summary` YAML block each report
carries, separate from the semantic-retrieval path.

Why this exists: "which company scored highest" is an exact-comparison
question. Embedding similarity can't reliably answer it — two report
headers that both say "**Score:** N.N/5" look nearly identical to a
sentence embedding regardless of what N.N actually is (see README's eval
history: gap-001 only passed because the mid-size sample set happened to
surface both scores in one 3-chunk window, not because the pipeline can
actually compare numbers). Numeric/exact-fact questions get a structured
answer from parsed YAML; open-ended narrative questions still go through
the RAG path in app/rag.py. Using the right tool for each question type,
rather than forcing embeddings to do arithmetic, is the point.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

MACHINE_SUMMARY_RE = re.compile(
    r"^## Machine Summary\s*\n```yaml\n(.*?)\n```", re.MULTILINE | re.DOTALL
)


def extract_machine_summary(markdown: str, source: str) -> dict[str, Any] | None:
    match = MACHINE_SUMMARY_RE.search(markdown)
    if not match:
        return None
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        return None
    data["_source"] = source
    return data


def load_all_summaries(reports_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(reports_dir.glob("*.md")):
        summary = extract_machine_summary(path.read_text(encoding="utf-8"), source=path.name)
        if summary:
            summaries.append(summary)
    return summaries


def highest_scoring(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [s for s in summaries if isinstance(s.get("score"), (int, float))]
    return max(scored, key=lambda s: s["score"]) if scored else None


def lowest_scoring(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [s for s in summaries if isinstance(s.get("score"), (int, float))]
    return min(scored, key=lambda s: s["score"]) if scored else None


def by_risk_level(summaries: list[dict[str, Any]], risk_level: str) -> list[dict[str, Any]]:
    return [s for s in summaries if str(s.get("risk_level", "")).lower() == risk_level.lower()]
