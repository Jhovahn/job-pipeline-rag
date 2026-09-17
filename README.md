# Job Pipeline RAG

A small retrieval-augmented Q&A tool over a folder of job-search evaluation
reports, with a rerunnable eval harness that scores answer quality against
a fixed set of golden questions.

Ask things like *"which role scored highest?"* or *"what's my most common
gap?"* instead of scrolling through markdown files by hand.

## Why this exists

Built to close a specific, repeatedly-flagged gap: several job evaluations
in my own pipeline noted I had "no shipped AI/RAG project with a measurable
outcome" — proficiencies listed on a resume, but nothing demonstrated. This
is that demonstration: a real retrieval pipeline, over real (if personal)
data, with a real eval score that's not the same every time I change the
system.

## How it works

1. **`app/ingest.py`** — splits each report into chunks (by `## ` heading),
   embeds them locally with `sentence-transformers` (no API key required),
   and saves the vectors + metadata to `index/`.
2. **`app/retrieve.py`** — cosine-similarity top-k search over that index.
   A homemade numpy search rather than a vector database: at a few hundred
   reports the whole index fits in memory, so a database adds dependency
   weight without adding capability.
3. **`app/rag.py`** — turns retrieved chunks into an answer. Runs in
   **extractive mode** by default (returns the best-matching excerpt
   verbatim with its source — zero dependencies, fully deterministic), or
   **generative mode** if `ANTHROPIC_API_KEY` is set (asks Claude to
   synthesize a short answer grounded only in the retrieved excerpts).
4. **`eval/`** — the part that matters most. `golden.json` is a fixed set
   of question/expected-answer pairs. `run_eval.py` runs every question
   against the live pipeline and grades it (keyword + citation match),
   producing a score. Change the chunking strategy, the embedding model,
   or `top_k`, then rerun — the score tells you if the change helped.

## Quickstart (with the included sample data)

```bash
pip install -r requirements.txt

# Build the index from the safe, fictional sample reports
python cli.py ingest --reports-dir data/sample_reports --index-dir index

# Ask a question
python cli.py ask "which company scored the highest?"

# Run the eval suite
python -m eval.run_eval
```

Or run it as a service:

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "which company scored the highest?"}'
```

## Using your own real reports

Real report data is **never** committed to this repo (see `.gitignore`).
To point the tool at your own reports instead of the sample set:

```bash
cp /path/to/your/career-ops/reports/*.md data/reports/
python cli.py ingest --reports-dir data/reports --index-dir index
python cli.py ask "what's my most common gap?"
```

To eval against your real data privately, copy `eval/golden.json` to
`eval/golden.local.json` (also gitignored), write questions specific to
your own reports, and run:

```bash
python -m eval.run_eval --golden eval/golden.local.json
```

## Eval history (a real before/after)

The eval harness isn't decorative — it caught a real bug during development.
The first version of `_extractive_answer` returned only the single
best-matching chunk. On the 5-question golden set, that scored:

```
Score: 1/5 (20%)
```

The fix: return the top 3 chunks instead of 1, since a single chunk (often
just a report's header) rarely contains every fact a question needs.
Re-running the exact same golden set after that one change:

```
Score: 3/5 (60%)
```

The remaining two failures turned out to be a grading bug, not a retrieval
bug: the golden set listed synonym alternatives ("no", "not", "missing")
under a grader that required *all* of them present, when only one needed
to match. Rewriting those two cases to each check one precise phrase
instead of three loosely-related ones got the suite to:

```
Score: 5/5 (100%)
```

That's the actual point of having an eval suite: not to hit 100% on the
first run, but to have a number that moves when you fix something, and
tells you whether your fix helped.

## What this is not

A production RAG system. It's intentionally small: no vector database,
no reranking, no chunk-overlap tuning, no query rewriting. Every one of
those is a legitimate next step, and each is a clearly scoped, separately
demonstrable improvement — which is the point of having a working eval
score to improve against in the first place.
