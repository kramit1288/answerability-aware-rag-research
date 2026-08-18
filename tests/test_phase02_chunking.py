from pathlib import Path

from answerability_rag.hashing import sha256_text
from answerability_rag.retrieval.chunking import ChunkSpec, chunk_document


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def document(text: str, name: str = "a.txt") -> dict[str, str]:
    return {"doc_id": f"doc:{name}", "filename": name, "archive_path": f"corpus/{name}",
            "raw_sha256": sha256_text(text), "normalized_sha256": sha256_text(text)}


def spec() -> ChunkSpec:
    return ChunkSpec("fixture", "revision", 4, 1, 3, "test-v1")


def test_deterministic_boundaries_overlap_offsets_and_hashes() -> None:
    text = "abcdefgh"
    rows = chunk_document(text, document=document(text), tokenizer=CharacterTokenizer(),
                          spec=spec(), corpus_manifest_sha256="corpus")
    assert [(r["token_start"], r["token_end"], r["char_start"], r["char_end"]) for r in rows] == [
        (0, 4, 0, 4), (3, 7, 3, 7), (6, 8, 6, 8)
    ]
    assert [r["text"] for r in rows] == ["abcd", "defg", "gh"]
    assert all(text[r["char_start"]:r["char_end"]] == r["text"] for r in rows)
    assert all(sha256_text(r["text"]) == r["text_sha256"] for r in rows)
    again = chunk_document(text, document=document(text), tokenizer=CharacterTokenizer(),
                           spec=spec(), corpus_manifest_sha256="corpus")
    assert rows == again


def test_chunk_ids_do_not_depend_on_document_processing_order() -> None:
    left = chunk_document("abcde", document=document("abcde", "a.txt"),
                          tokenizer=CharacterTokenizer(), spec=spec(), corpus_manifest_sha256="x")
    right = chunk_document("vwxyz", document=document("vwxyz", "b.txt"),
                           tokenizer=CharacterTokenizer(), spec=spec(), corpus_manifest_sha256="x")
    assert {r["chunk_id"] for r in left + right} == {r["chunk_id"] for r in right + left}


def test_empty_document_behavior() -> None:
    assert chunk_document("", document=document(""), tokenizer=CharacterTokenizer(),
                          spec=spec(), corpus_manifest_sha256="x") == []
