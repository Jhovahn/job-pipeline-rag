"""Cosine-similarity top-k retrieval over the index built by ingest.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingest import EMBEDDING_MODEL


@dataclass
class RetrievedChunk:
    source: str
    heading: str
    text: str
    score: float


class Retriever:
    def __init__(self, index_dir: Path):
        vectors_path = index_dir / "vectors.npy"
        chunks_path = index_dir / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_dir}. Run `python -m app.ingest` first."
            )
        self.vectors = np.load(vectors_path)
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.model.encode([question], normalize_embeddings=True)[0]
        # Vectors are pre-normalized, so the dot product IS the cosine similarity.
        scores = self.vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(
                source=self.chunks[i]["source"],
                heading=self.chunks[i]["heading"],
                text=self.chunks[i]["text"],
                score=float(scores[i]),
            )
            for i in top_indices
        ]
