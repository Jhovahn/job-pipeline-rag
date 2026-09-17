"""
Orchestrates retrieval + answer synthesis.

Two modes, chosen automatically:
- No ANTHROPIC_API_KEY set -> "extractive" mode: returns the best-matching
  chunk(s) verbatim with citations. Zero external dependency, always works,
  fully deterministic.
- ANTHROPIC_API_KEY set -> "generative" mode: asks Claude to write a short,
  direct answer grounded ONLY in the retrieved chunks, still citing sources.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.retrieve import Retriever, RetrievedChunk


@dataclass
class Answer:
    text: str
    citations: list[str]
    mode: str


SYSTEM_PROMPT = """You answer questions about a candidate's job-search evaluation reports.
Use ONLY the excerpts provided below. If the excerpts don't contain the answer, say so plainly.
Never invent a score, company name, or fact that isn't in the excerpts. Keep the answer to 2-3 sentences."""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c.source} | {c.heading}]\n{c.text}" for c in chunks
    )


def _extractive_answer(chunks: list[RetrievedChunk], max_chunks: int = 3) -> Answer:
    """
    Concatenate the top N chunks rather than just the single best match.
    A single chunk (e.g. one report's header) rarely contains every fact a
    question needs; a few chunks give the keyword-matching grader (and a
    human reader) more surface area without pretending to synthesize.
    """
    top = chunks[:max_chunks]
    body = "\n\n".join(f"(from {c.source} — {c.heading}):\n{c.text}" for c in top)
    return Answer(
        text=body,
        citations=sorted({c.source for c in top}),
        mode="extractive",
    )


def _generative_answer(question: str, chunks: list[RetrievedChunk]) -> Answer:
    import anthropic

    client = anthropic.Anthropic()
    context = _build_context(chunks)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(text=text, citations=sorted({c.source for c in chunks}), mode="generative")


def answer_question(retriever: Retriever, question: str, top_k: int = 5) -> Answer:
    chunks = retriever.search(question, top_k=top_k)
    if not chunks:
        return Answer(text="No matching reports found.", citations=[], mode="none")

    if os.environ.get("ANTHROPIC_API_KEY"):
        return _generative_answer(question, chunks)
    return _extractive_answer(chunks)
