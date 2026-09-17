"""
Loads markdown evaluation reports, splits them into chunks, embeds each
chunk with a local sentence-transformers model, and writes a small index
to disk (a numpy array of vectors + a parallel JSON of chunk metadata).

No vector database dependency on purpose: at a few hundred reports the
whole index fits in memory, so a homemade cosine-similarity search over
a numpy array is simpler to read, debug, and explain than standing up
Chroma/pgvector for a dataset this size.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_HEADER_RE = re.compile(r"^##\s+.+$", re.MULTILINE)


@dataclass
class Chunk:
    source: str      # e.g. "reports/002-vapi-2026-09-11.md"
    heading: str      # e.g. "## B) Match with CV"
    text: str


def split_into_chunks(markdown: str, source: str) -> list[Chunk]:
    """Split a report on its `## ` headings; each chunk keeps its own heading as context."""
    matches = list(CHUNK_HEADER_RE.finditer(markdown))
    if not matches:
        return [Chunk(source=source, heading="(full document)", text=markdown.strip())]

    chunks = []
    # Anything before the first heading (title + header fields + Machine Summary).
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(source=source, heading="(header)", text=preamble))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        heading = match.group().strip()
        if body:
            chunks.append(Chunk(source=source, heading=heading, text=body))
    return chunks


def build_index(reports_dir: Path, index_dir: Path) -> int:
    """Read every *.md in reports_dir, embed all chunks, save vectors + metadata to index_dir."""
    report_paths = sorted(reports_dir.glob("*.md"))
    if not report_paths:
        raise FileNotFoundError(f"No .md reports found in {reports_dir}")

    all_chunks: list[Chunk] = []
    for path in report_paths:
        markdown = path.read_text(encoding="utf-8")
        all_chunks.extend(split_into_chunks(markdown, source=path.name))

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(
        [f"{c.heading}\n{c.text}" for c in all_chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", np.asarray(vectors, dtype=np.float32))
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in all_chunks], indent=2),
        encoding="utf-8",
    )
    return len(all_chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the report embedding index.")
    parser.add_argument("--reports-dir", default="data/reports")
    parser.add_argument("--index-dir", default="index")
    args = parser.parse_args()

    n = build_index(Path(args.reports_dir), Path(args.index_dir))
    print(f"Indexed {n} chunks from {args.reports_dir} -> {args.index_dir}")
