import pytest
from branchlab.data import split_documents, sha256


def test_document_disjoint_deterministic_order_invariant():
    texts = [f"story {i}" for i in range(30)] + ["story 1", " story 1 "]
    a = split_documents(texts, 10, 5, 7)
    b = split_documents(reversed(texts), 10, 5, 7)
    assert a == b
    ids = [r["id"] for rows in a.values() for r in rows]
    assert len(ids) == len(set(ids)) == 20
    assert all(r["id"] == sha256(r["text"].encode()) for rows in a.values() for r in rows)
    assert a != split_documents(texts, 10, 5, 8)


def test_insufficient_unique_documents_fails():
    with pytest.raises(ValueError):
        split_documents(["same"] * 20, 3, 1)


@pytest.mark.parametrize("train_count,eval_count", [(-1, 2), (2, -1), (0, 2), (2, 0),
                                                   (True, 2), (2, False), (2.5, 2), (2, 1.5)])
def test_invalid_counts_rejected_before_negative_slicing_can_overlap_splits(train_count, eval_count):
    with pytest.raises(ValueError, match="positive integer"):
        split_documents([f"story {i}" for i in range(30)], train_count, eval_count)
