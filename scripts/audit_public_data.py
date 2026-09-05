#!/usr/bin/env python3
"""Audit a pinned public Delphi snapshot; never downloads model weights."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/project1-delphi-data"))
    parser.add_argument("--offline", action="store_true", help="Only use hash-verified cached files")
    args = parser.parse_args()
    lock_path = ROOT / "sources" / "delphi.lock.json"
    lock = json.loads(lock_path.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    files = []
    unique_urls = set()
    for item in lock["files"]:
        config = item["config"]
        path = args.cache / f"{config}.parquet"
        url = (f"https://huggingface.co/datasets/{lock['dataset']}/resolve/"
               f"{lock['revision']}/{config}/train.parquet")
        if path.exists():
            raw = path.read_bytes()
        elif args.offline:
            raise SystemExit(f"Missing cached source: {path}")
        else:
            request = Request(url, headers={"User-Agent": "Project1-source-audit/0.1"})
            with urlopen(request, timeout=30) as response:
                raw = response.read(item["bytes"] + 1)
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != item["bytes"] or digest != item["sha256"]:
            raise SystemExit(f"Size/hash mismatch: {config}; source lock was not updated")
        if not path.exists():
            path.write_bytes(raw)
        table = pq.read_table(pa.BufferReader(raw))
        rows = table.to_pylist()
        urls = {row["wandb_url"] for row in rows if row.get("wandb_url")}
        unique_urls.update(urls)
        # Field-name screening is a diagnostic, not a proof about the wider corpus.
        temporal_fields = [field for field in table.column_names if field.lower() in {
            "step", "global_step", "_step", "timestamp", "_timestamp", "action",
            "decision", "reward", "checkpoint", "policy", "optimizer_state", "rng_state",
        }]
        files.append({
            "config": config, "source_url": url, "bytes": len(raw), "sha256": digest,
            "rows": table.num_rows, "columns": table.column_names,
            "kind_counts": dict(Counter(str(row.get("kind")) for row in rows)),
            "state_counts": dict(Counter(str(row.get("state")) for row in rows)),
            "unique_wandb_urls": len(urls),
            "wandb_nulls": sum(not row.get("wandb_url") for row in rows),
            "explicit_temporal_or_action_fields": temporal_fields,
        })
    report = {
        "status": "SOURCE_AUDIT_ONLY_NO_TRAINING",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": lock["dataset"], "revision": lock["revision"],
        "source_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "pyarrow_version": pa.__version__, "files": files,
        "total_rows": sum(item["rows"] for item in files),
        "total_bytes": sum(item["bytes"] for item in files),
        "unique_nonempty_wandb_urls_across_tables": len(unique_urls),
        "limitations": [
            "Rows include derived fits and repeated or external evaluations; rows are not independent trajectories.",
            "W&B URLs were counted, not downloaded or verified as complete histories.",
            "No full model/optimizer/RNG/data-cursor checkpoints were downloaded.",
            "Field-name screening applies only to these six pinned parquet files.",
            "This audit does not measure training quality, intervention effects, or RSI.",
        ],
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Delphi 数据可用性实测", "", f"状态：`{report['status']}`", "",
        f"固定版本：`{lock['revision']}`；全部文件逐一通过大小及 SHA-256 核对。", "",
        "| 子集 | 表格行数 | 字节数 | 非空 W&B 链接数（子集内去重） |",
        "|---|---:|---:|---:|",
    ]
    for item in files:
        lines.append(f"| {item['config']} | {item['rows']} | {item['bytes']} | {item['unique_wandb_urls']} |")
    lines += [
        "", f"合计 {report['total_rows']} 行，{report['total_bytes']} 字节。行数不能解释为训练轨迹数。", "",
        "逐字段检查的详细结果见同目录 source_audit.json。6 个表均未出现本脚本检查的显式逐步、动作或完整恢复状态字段；这不是对整个 Marin/W&B 语料可用性的结论。", "",
        "这些文件可以用于配方比较、规模拟合与可用性筛查。在线干预、同状态分叉实验及递归改进收益需另行产生实测数据。", "",
        "本次没有下载权重、获取 W&B 完整历史、调用模型 API 或进行模型训练。", "",
    ]
    (out / "source_audit.md").write_text("\n".join(lines))
    print(json.dumps({key: report[key] for key in ["status", "revision", "total_rows", "total_bytes"]}))


if __name__ == "__main__":
    main()
