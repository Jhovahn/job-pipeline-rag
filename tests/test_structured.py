from app.structured import extract_machine_summary, highest_scoring, lowest_scoring, by_risk_level

REPORT_A = """# Evaluation: Foo — Role

## Machine Summary

```yaml
company: "Foo"
score: 4.0
risk_level: "Medium"
```
"""

REPORT_B = """# Evaluation: Bar — Role

## Machine Summary

```yaml
company: "Bar"
score: 2.4
risk_level: "High"
```
"""


def test_extract_parses_yaml_fields():
    summary = extract_machine_summary(REPORT_A, source="a.md")
    assert summary["company"] == "Foo"
    assert summary["score"] == 4.0
    assert summary["_source"] == "a.md"


def test_extract_returns_none_without_machine_summary():
    assert extract_machine_summary("# No summary here", source="x.md") is None


def test_highest_and_lowest_scoring():
    summaries = [
        extract_machine_summary(REPORT_A, "a.md"),
        extract_machine_summary(REPORT_B, "b.md"),
    ]
    assert highest_scoring(summaries)["company"] == "Foo"
    assert lowest_scoring(summaries)["company"] == "Bar"


def test_by_risk_level_is_case_insensitive():
    summaries = [
        extract_machine_summary(REPORT_A, "a.md"),
        extract_machine_summary(REPORT_B, "b.md"),
    ]
    matches = by_risk_level(summaries, "medium")
    assert len(matches) == 1
    assert matches[0]["company"] == "Foo"
