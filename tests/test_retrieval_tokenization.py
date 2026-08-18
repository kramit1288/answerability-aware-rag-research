from answerability_rag.retrieval.tokenization import tokenize_bm25


def test_preserves_realistic_technical_identifiers() -> None:
    text = "Error-Code_42 v4.1.1.2 api/v1/resource server:8080/path HTTP://Host:8080/Path"
    assert tokenize_bm25(text) == [
        "error-code_42",
        "v4.1.1.2",
        "api/v1/resource",
        "server:8080/path",
        "http://host:8080/path",
    ]


def test_nfkc_and_unicode_casefold_are_deterministic() -> None:
    assert tokenize_bm25("ＦＯＯ＿１２ Straße STRASSE") == ["foo_12", "strasse", "strasse"]


def test_external_punctuation_splits_and_internal_separators_survive() -> None:
    assert tokenize_bm25("(alpha-beta), gamma.delta; /edge/ :port: trail-") == [
        "alpha-beta",
        "gamma.delta",
        "edge",
        "port",
        "trail",
    ]


def test_unapproved_identifier_punctuation_is_not_retained() -> None:
    assert tokenize_bm25("C++ C# foo@bar x=y") == ["c", "c", "foo", "bar", "x", "y"]


def test_no_stopword_removal_stemming_or_lemmatization() -> None:
    assert tokenize_bm25("The running services") == ["the", "running", "services"]


def test_whitespace_and_empty_input() -> None:
    assert tokenize_bm25("  alpha\t\n beta  ") == ["alpha", "beta"]
    assert tokenize_bm25("--- ... /// :::") == []
