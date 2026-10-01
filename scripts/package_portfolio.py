#!/usr/bin/env python3
"""Build portable release bundles and SHA256SUMS from completed verified runs."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from verify_portfolio import verify

from branchlab.portfolio_data import file_sha256


def add_tree(archive, root, prefix):
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            archive.add(path, arcname=str(Path(prefix) / path.relative_to(root)))


def package(results, prepared, reports, output):
    results, prepared, reports, output = map(Path, (results, prepared, reports, output))
    verify(results, prepared)
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    inference = output / "branchlab-v0.2.0-inference.tar.gz"
    with tarfile.open(inference, "w:gz") as archive:
        for name in ("mha-s101", "gqa-s101", "showcase-gqa"):
            for filename in (
                "model.pt",
                "tokenizer.json",
                "run.json",
                "config.json",
                "examples.json",
            ):
                archive.add(
                    results / name / filename, arcname=f"portfolio/{name}/{filename}"
                )
    assets.append(inference)
    evidence = output / "branchlab-v0.2.0-evidence.tar.gz"
    with tarfile.open(evidence, "w:gz") as archive:
        add_tree(archive, results, "portfolio")
        for name in (
            "manifest.json",
            "tokenizer.json",
            "train_tokens.npy",
            "dev_tokens.npy",
            "test_tokens.npy",
        ):
            archive.add(prepared / name, arcname=f"official-data/{name}")
        add_tree(archive, reports, "reports/portfolio")
        archive.add("THIRD_PARTY.md", arcname="THIRD_PARTY.md")
        archive.add("LICENSE", arcname="LICENSE")
    assets.append(evidence)
    sums = "".join(f"{file_sha256(path)}  {path.name}\n" for path in assets)
    (output / "SHA256SUMS").write_text(sums)
    print(sums, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/portfolio")
    parser.add_argument("--prepared", default="artifacts/official-data")
    parser.add_argument("--reports", default="reports/portfolio")
    parser.add_argument("--output", default="dist/portfolio")
    args = parser.parse_args()
    package(args.results, args.prepared, args.reports, args.output)
