from app.ingest import split_into_chunks

SAMPLE = """# Evaluation: Foo — Bar Role

**Score:** 3.0/5

## Machine Summary

```yaml
company: "Foo"
score: 3.0
```

## B) Match with CV

Some content here.

## Risk Summary

Risk content here.
"""


def test_splits_on_headings():
    chunks = split_into_chunks(SAMPLE, source="test.md")
    headings = [c.heading for c in chunks]
    assert headings == [
        "(header)",
        "## Machine Summary",
        "## B) Match with CV",
        "## Risk Summary",
    ]


def test_preamble_chunk_has_title_and_score():
    chunks = split_into_chunks(SAMPLE, source="test.md")
    preamble = chunks[0]
    assert "Evaluation: Foo" in preamble.text
    assert "3.0/5" in preamble.text


def test_no_headings_falls_back_to_whole_document():
    chunks = split_into_chunks("just plain text, no headings", source="test.md")
    assert len(chunks) == 1
    assert chunks[0].heading == "(full document)"


def test_every_chunk_keeps_its_source():
    chunks = split_into_chunks(SAMPLE, source="test.md")
    assert all(c.source == "test.md" for c in chunks)
