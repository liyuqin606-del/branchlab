"""Pinned, small TinyStories source and document-disjoint local partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCE_FILE = "data/validation-00000-of-00001-869c898b519ad725.parquet"
SOURCE_SHA256 = "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab"
SOURCE_BYTES = 9989127
SOURCE_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/{SOURCE_FILE}"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def split_documents(texts, train_count=6000, eval_count=600, seed=1729):
    """Exact normalized-document deduplication; no cross-split document reuse."""
    for name, count in (("train_count", train_count), ("eval_count", eval_count)):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    unique = {}
    for text in texts:
        text = text.strip()
        if text:
            unique.setdefault(sha256(text.encode("utf-8")), text)
    ordered = sorted(unique, key=lambda h: sha256(f"{seed}:{h}".encode()))
    need = train_count + 2 * eval_count
    if len(ordered) < need:
        raise ValueError(f"Need {need} distinct documents, found {len(ordered)}")
    indices = {"train": ordered[:train_count], "dev": ordered[train_count:train_count+eval_count],
               "test": ordered[train_count+eval_count:need]}
    return {split: [{"id": h, "text": unique[h]} for h in ids] for split, ids in indices.items()}


def prepare_data(output_dir, max_train_docs=6000, max_eval_docs=600, seed=1729):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "tinystories-validation.parquet"
    if not source.exists():
        request = Request(SOURCE_URL, headers={"User-Agent": "BranchLab/0.1"})
        with urlopen(request, timeout=60) as response:
            raw = response.read(SOURCE_BYTES + 1)
        if len(raw) != SOURCE_BYTES or sha256(raw) != SOURCE_SHA256:
            raise ValueError("Pinned TinyStories source failed size/hash verification")
        source.write_bytes(raw)
    raw = source.read_bytes()
    if len(raw) != SOURCE_BYTES or sha256(raw) != SOURCE_SHA256:
        raise ValueError("Cached TinyStories source failed size/hash verification")
    texts = pq.read_table(source, columns=["text"])["text"].to_pylist()
    splits = split_documents(texts, max_train_docs, max_eval_docs, seed)
    manifest = {"dataset": "roneneldan/TinyStories", "revision": REVISION,
                "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                "license": "CDLA-Sharing-1.0", "seed": seed,
                "scope": "Local repartition of official validation subset; not official TinyStories evaluation",
                "deduplication": "Exact normalized-document hash only; near-duplicate leakage not audited",
                "splits": {}}
    for split, rows in splits.items():
        raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
        (out / f"{split}.jsonl").write_bytes(raw)
        manifest["splits"][split] = {"count": len(rows), "sha256": sha256(raw), "document_ids": [r["id"] for r in rows]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_texts(path):
    with Path(path).open() as source:
        return [json.loads(line)["text"] for line in source if line.strip()]
