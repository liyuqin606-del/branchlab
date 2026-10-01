"""Pinned official-source subset for the portfolio training experiment.

One official training shard supplies training documents. Development/test use
separate documents from the official validation shard. This is a subset study,
not a full-data TinyStories benchmark. Tokenizer fitting sees only train text.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq

from .data import REVISION, SOURCE_BYTES, SOURCE_SHA256, SOURCE_URL, sha256
from .tokenizer import ByteBPETokenizer
from .training import json_write, pack_texts

TRAIN_FILE = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"
TRAIN_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/{TRAIN_FILE}"
TRAIN_SHA256 = "77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b"
TRAIN_BYTES = 248731111


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(
    url: str, path: Path, expected_size: int, expected_hash: str
) -> None:
    """Download to a temporary file; never accept a partial cached source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        request = Request(url, headers={"User-Agent": "BranchLab/0.2"})
        try:
            with (
                urlopen(request, timeout=60) as response,
                temporary.open("wb") as target,
            ):
                while block := response.read(1024 * 1024):
                    target.write(block)
            if (
                temporary.stat().st_size != expected_size
                or file_sha256(temporary) != expected_hash
            ):
                raise ValueError("Downloaded dataset size/SHA256 mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != expected_size or file_sha256(path) != expected_hash:
        raise ValueError("Cached dataset size/SHA256 mismatch")


def document_id(text: str) -> str:
    canonical = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip())
    return sha256(canonical.encode("utf-8"))


def select_partitions(
    train_texts, validation_texts, train_docs=50000, eval_docs=512, seed=20261001
):
    """Discard training overlap with *all* validation docs before subset selection."""
    if train_docs < 1 or eval_docs < 1:
        raise ValueError("Document counts must be positive")
    validation = {document_id(t): t.strip() for t in validation_texts if t.strip()}
    training = {}
    overlap = 0
    for text in train_texts:
        if not text.strip():
            continue
        identity = document_id(text)
        if identity in validation:
            overlap += 1
        else:
            training.setdefault(identity, text.strip())
    rank = lambda identity: sha256(f"{seed}:{identity}".encode())
    train_ids = sorted(training, key=rank)[:train_docs]
    eval_ids = sorted(validation, key=rank)[: 2 * eval_docs]
    if len(train_ids) != train_docs or len(eval_ids) != 2 * eval_docs:
        raise ValueError("Insufficient distinct documents for requested partitions")
    partitions = {
        "train": [{"id": i, "text": training[i]} for i in train_ids],
        "dev": [{"id": i, "text": validation[i]} for i in eval_ids[:eval_docs]],
        "test": [{"id": i, "text": validation[i]} for i in eval_ids[eval_docs:]],
    }
    return partitions, {
        "unique_train_available": len(training),
        "unique_validation_available": len(validation),
        "train_rows_removed_for_validation_overlap": overlap,
    }


def prepare_portfolio(
    output_dir,
    train_docs=50000,
    eval_docs=512,
    vocab_size=2048,
    tokenizer_docs=10000,
    seed=20261001,
):
    out = Path(output_dir)
    if (out / "manifest.json").exists():
        raise FileExistsError(
            "Prepared data already exists; use a new output directory"
        )
    download_verified(
        TRAIN_URL, out / "train-source.parquet", TRAIN_BYTES, TRAIN_SHA256
    )
    download_verified(
        SOURCE_URL, out / "validation-source.parquet", SOURCE_BYTES, SOURCE_SHA256
    )
    train = pq.read_table(out / "train-source.parquet", columns=["text"])[
        "text"
    ].to_pylist()
    valid = pq.read_table(out / "validation-source.parquet", columns=["text"])[
        "text"
    ].to_pylist()
    partitions, audit = select_partitions(train, valid, train_docs, eval_docs, seed)
    del train, valid
    tokenizer = ByteBPETokenizer.train(
        [r["text"] for r in partitions["train"][:tokenizer_docs]], vocab_size
    )
    tokenizer.save(out / "tokenizer.json")
    manifest = {
        "format": "branchlab-official-subset-v1",
        "revision": REVISION,
        "seed": seed,
        "sources": [
            {"url": TRAIN_URL, "sha256": TRAIN_SHA256, "bytes": TRAIN_BYTES},
            {"url": SOURCE_URL, "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES},
        ],
        "dataset": "roneneldan/TinyStories",
        "license": "CDLA-Sharing-1.0",
        "scope": "One official train shard subset; document-disjoint dev/test from official validation",
        "deduplication": "NFC + whitespace normalized exact document hash; no near-duplicate audit",
        "tokenizer_training_documents": min(tokenizer_docs, train_docs),
        "tokenizer_sha256": file_sha256(out / "tokenizer.json"),
        "vocab_size": tokenizer.vocab_size,
        "audit": audit,
        "splits": {},
    }
    for split, rows in partitions.items():
        text_path = out / f"{split}.jsonl"
        text_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        tokens = pack_texts((r["text"] for r in rows), tokenizer)
        token_path = out / f"{split}_tokens.npy"
        np.save(token_path, tokens)
        manifest["splits"][split] = {
            "documents": len(rows),
            "tokens": len(tokens),
            "documents_sha256": file_sha256(text_path),
            "tokens_sha256": file_sha256(token_path),
            "document_ids": [r["id"] for r in rows],
        }
    json_write(out / "manifest.json", manifest)
    return manifest
