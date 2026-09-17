"""FastAPI app: a single POST /ask endpoint over the evaluation-report index."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import answer_question
from app.retrieve import Retriever

INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", "index"))

app = FastAPI(title="Job Pipeline RAG", version="0.1.0")
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(INDEX_DIR)
    return _retriever


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    mode: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        retriever = get_retriever()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    result = answer_question(retriever, req.question, top_k=req.top_k)
    return AskResponse(answer=result.text, citations=result.citations, mode=result.mode)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
