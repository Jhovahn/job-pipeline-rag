from eval.run_eval import grade


def test_passes_when_keyword_and_source_match():
    case = {"expected_keywords": ["no documented ai"], "expected_sources": ["002-sampletech"]}
    assert grade("This report says no documented ai project exists.", ["002-sampletech-x.md"], case)


def test_fails_when_keyword_missing():
    case = {"expected_keywords": ["no documented ai"], "expected_sources": ["002-sampletech"]}
    assert not grade("Nothing relevant here.", ["002-sampletech-x.md"], case)


def test_fails_when_source_missing():
    case = {"expected_keywords": ["no documented ai"], "expected_sources": ["002-sampletech"]}
    assert not grade("no documented ai project", ["999-wrong-file.md"], case)


def test_keyword_match_is_case_insensitive():
    case = {"expected_keywords": ["high confidence"], "expected_sources": ["001-examplecorp"]}
    assert grade("Legitimacy: HIGH CONFIDENCE", ["001-examplecorp-x.md"], case)
