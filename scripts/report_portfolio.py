#!/usr/bin/env python3
"""Generate a readable report and CV entries only from verified raw records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from verify_portfolio import verify

from branchlab.portfolio_data import file_sha256


def apply_style():
    """Optional installed Figura helpers; portable defaults require only matplotlib."""
    skill_root = os.environ.get("FIGURA_SKILL_ROOT")
    if skill_root:
        sys.path.insert(0, str(Path(skill_root) / "scripts"))
        import colors
        import pubstyle

        pubstyle.apply()
        colors.apply_cycle()
    else:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
                "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73"]),
            }
        )


def export_figure(figure, out, name):
    for extension in ("png", "svg", "pdf"):
        figure.savefig(out / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def report(results, output, prepared=None):
    root, out = Path(results), Path(output)
    verification = verify(root, prepared)
    out.mkdir(parents=True, exist_ok=True)
    if (root / "validation").exists():
        shutil.copytree(
            root / "validation",
            out / "validation",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.tar.gz"),
        )
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    for path in root.rglob("*.json"):
        if "source" in path.parts:
            continue
        if path.name == "data_manifest.json":
            target = raw / "data_manifest.json"
            if not target.exists():
                shutil.copy2(path, target)
            continue
        target = raw / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    runs = {
        path.parent.name: json.loads(path.read_text())
        for path in root.glob("*/run.json")
    }
    showcase = runs["showcase-gqa"]
    paired = {
        v: [runs[f"{v}-s{s}"]["test"]["cross_entropy"] for s in (101, 202, 303)]
        for v in ("mha", "gqa")
    }
    differences = [b - a for a, b in zip(paired["mha"], paired["gqa"])]
    summary = {
        "verification": verification,
        "paired_test_loss": paired,
        "mean_gqa_minus_mha": statistics.mean(differences),
        "paired_differences": differences,
        "showcase": showcase,
        "scope": "Descriptive 3-seed subset comparison; no statistical superiority claim",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    apply_style()
    palette = {"mha": "#0072B2", "gqa": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for variant in ("mha", "gqa"):
        histories = [
            json.loads((root / f"{variant}-s{s}" / "history.json").read_text())
            for s in (101, 202, 303)
        ]
        x = np.array([h["trained_tokens"] for h in histories[0]]) / 1e6
        y = np.array(
            [[h["dev"]["cross_entropy"] for h in history] for history in histories]
        )
        axes[0].plot(
            x,
            y.mean(0),
            color=palette[variant],
            ls="-" if variant == "mha" else "--",
            label=variant.upper(),
        )
        axes[0].fill_between(
            x,
            y.mean(0) - y.std(0, ddof=1),
            y.mean(0) + y.std(0, ddof=1),
            color=palette[variant],
            alpha=0.15,
        )
    axes[0].set(
        xlabel="Training tokens (millions)",
        ylabel="Development cross-entropy",
        title="Matched token budgets · mean ± seed SD",
    )
    axes[0].legend(frameon=False)
    for seed_index, seed in enumerate((101, 202, 303)):
        axes[1].plot(
            [0, 1],
            [paired["mha"][seed_index], paired["gqa"][seed_index]],
            color="#999999",
            lw=1,
        )
        axes[1].scatter(
            [0, 1],
            [paired["mha"][seed_index], paired["gqa"][seed_index]],
            c=[palette["mha"], palette["gqa"]],
            s=32,
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["MHA", "GQA"],
        ylabel="Final test cross-entropy",
        title="Three paired training seeds",
    )
    export_figure(fig, out, "quality")
    fixed = {
        name: json.loads((root / name / "benchmark-b1-p256.json").read_text())
        for name in ("mha-s101", "gqa-s101", "showcase-gqa")
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    positions = np.arange(3)
    for i, (mode, color) in enumerate(
        zip(("uncached", "dynamic", "static"), ("#666666", "#0072B2", "#009E73"))
    ):
        values = [
            fixed[name]["paths"][mode]["decode_tokens_per_second"] for name in fixed
        ]
        axes[0].bar(positions + (i - 1) * 0.23, values, 0.23, color=color, label=mode)
    axes[0].set(
        xticks=positions,
        xticklabels=["MHA s101", "GQA s101", "GQA showcase"],
        ylabel="Decode tokens / second",
        title="L4 · BF16 · batch 1 · prompt 256 / decode 64",
    )
    axes[0].legend(frameon=False, loc="upper left", ncol=3)
    axes[0].set_ylim(0, 180)
    memory = [
        fixed[name]["paths"]["static"]["cache_allocated_bytes"] / 1024**2
        for name in fixed
    ]
    axes[1].bar(positions, memory, color=["#0072B2", "#D55E00", "#D55E00"])
    axes[1].set(
        xticks=positions,
        xticklabels=["MHA s101", "GQA s101", "GQA showcase"],
        ylabel="Persistent K/V buffer (MiB)",
        title="Actual allocated storage · same capacity",
    )
    export_figure(fig, out, "systems")
    rows = []
    for variant in ("mha", "gqa"):
        row = runs[f"{variant}-s101"]
        rows.append(
            f"| {variant.upper()} | {row['parameters']:,} | {row['trained_tokens']:,} | {statistics.mean(paired[variant]):.4f} ± {statistics.stdev(paired[variant]):.4f} |"
        )
    measurements = []
    for name, benchmark in fixed.items():
        for mode, row in benchmark["paths"].items():
            measurements.append(
                f"| {name} | {mode} | {row['decode_tokens_per_second']:.1f} | {row['speedup_vs_uncached']:.2f}× | {row['cache_allocated_bytes'] / 1024**2:.3f} |"
            )
    cache_reduction = (
        1
        - fixed["gqa-s101"]["paths"]["static"]["cache_allocated_bytes"]
        / fixed["mha-s101"]["paths"]["static"]["cache_allocated_bytes"]
    )
    graph_section = ""
    graph_cv = ""
    graph_cv_en = ""
    graph_root = root / "graph-inference"
    if graph_root.exists():
        graph_records = [
            json.loads(path.read_text())
            for path in sorted(graph_root.glob("graph-b*.json"))
        ]
        graph_records.sort(
            key=lambda r: (r["metadata"]["batch_size"], r["metadata"]["prompt_tokens"])
        )
        labels = [
            f"B{r['metadata']['batch_size']} / P{r['metadata']['prompt_tokens']}"
            for r in graph_records
        ]
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        x = np.arange(len(graph_records))
        for offset, mode, color in [
            (-0.18, "dynamic", "#0072B2"),
            (0.18, "cuda_graph", "#D55E00"),
        ]:
            ax.bar(
                x + offset,
                [r["decode_tokens_per_second"][mode] for r in graph_records],
                0.36,
                label=mode,
                color=color,
            )
        ax.set(
            xticks=x,
            xticklabels=labels,
            ylabel="Decode tokens / second",
            title="Reusable CUDA graph · L4 / BF16 · decode 64",
        )
        ax.legend(frameon=False)
        export_figure(fig, out, "cuda_graph")
        graph_rows = []
        for row in graph_records:
            m = row["metadata"]
            graph_rows.append(
                f"| {m['batch_size']} | {m['prompt_tokens']} | {row['decode_tokens_per_second']['dynamic']:.1f} | {row['decode_tokens_per_second']['cuda_graph']:.1f} | {row['speedup_vs_dynamic']:.2f}× | {row['capture_and_warmup_seconds']:.3f} | {row['parity']['mean_distribution_kl']:.6f} | {row['parity']['argmax_disagreement_fraction']:.2%} |"
            )
        selected = next(
            r
            for r in graph_records
            if r["metadata"]["batch_size"] == 1
            and r["metadata"]["prompt_tokens"] == 256
        )
        graph_section = (
            """## Reusable CUDA graph extension

This inference-only extension was added after observing the small ordinary-cache
speedups. The training protocol and weights were unchanged. A fixed-shape decoder
reuses a captured graph with device-side position and preallocated K/V buffers.
Measurements use the showcase checkpoint, L4/BF16, 64 forced decode tokens and
seven alternating-order repetitions. Capture/warmup cost is excluded from steady
state and reported separately. Prefill times remain in the raw records.

| Batch | Prompt | Dynamic tokens/s | Graph tokens/s | Speedup | Setup seconds | Mean KL | Argmax disagreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
            + "\n".join(graph_rows)
            + """

BF16 paths have small numerical differences; argmax disagreements mean identical
free-running text is not guaranteed. Timing uses identical forced tokens to avoid
changing the workload. Fixed weights, dtype, batch and capacity are required;
one decoder serves one serial request at a time. This is not an end-to-end
serving benchmark, custom attention kernel, or universally applicable speedup.

![CUDA graph measurements](cuda_graph.png)

"""
        )
        graph_cv = f"- 实现固定形状 CUDA Graph 解码；在 L4/BF16、batch 1、prompt 256/decode 64 的七次重复中，稳态吞吐达 {selected['decode_tokens_per_second']['cuda_graph']:.1f} tokens/s，相对动态 KV-cache 提速 {selected['speedup_vs_dynamic']:.2f}×；单独报告捕获开销与数值误差。\n"
        graph_cv_en = f"Implemented reusable CUDA graph decoding, reaching {selected['decode_tokens_per_second']['cuda_graph']:.1f} tokens/s ({selected['speedup_vs_dynamic']:.2f}× over dynamic KV caching) on L4/BF16 at batch 1, prompt 256/decode 64 across seven repetitions; capture and prefill excluded and separately reported.\n"
    report_text = f"""# BranchLab v0.2 — verified language-model engineering

Seven fixed-budget GPU training runs completed in {json.loads((root / "completion.json").read_text())["wall_seconds"] / 60:.2f} minutes, including the original inference grid.
All stored budgets, model weights,
artifact hashes and benchmark summary statistics passed independent record checks.
These checks do not substitute for rerunning training.

## Model and data

The showcase has **{showcase["parameters"]:,} parameters**, trained on
**{showcase["trained_tokens"]:,} tokens**, with final held-out cross-entropy
**{showcase["test"]["cross_entropy"]:.4f}** and perplexity **{showcase["test"]["perplexity"]:.2f}**.
Each final evaluation uses {showcase["test"]["target_tokens"]:,} nonrepeated target tokens.

Training uses a deterministic 50,000-document subset of one pinned official
TinyStories training shard. Development and test each have 512 distinct documents
from official validation, selected before training. All training documents that
exactly match any normalized validation document are excluded. BPE fits only on
10,000 training documents. Near-duplicate leakage is not audited. This is a subset
study, not a full-data or official full-split benchmark.

## MHA/GQA comparison

| Architecture | Parameters | Tokens per run | Test cross-entropy, mean ± seed SD |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

Seeds: 101, 202, 303. Paired GQA-minus-MHA loss differences:
{", ".join(f"{d:+.5f}" for d in differences)}; mean **{statistics.mean(differences):+.5f}**.
Negative values favor GQA. The comparison matches tokens, model width/depth and
seeds; GQA has fewer parameters and different FLOPs. Three seeds are descriptive
and do not establish statistical superiority or general language-model quality.
The independently declared showcase has a larger budget and is excluded from this comparison.

![Training and held-out quality](quality.png)

## Same-weight inference paths

The table uses **NVIDIA L4, BF16, batch 1, prompt 256, decode 64, seven repeats**.
Full grid records also include batch 4 and prompt 64. Prefill and decode are timed
separately; these speedups describe decode only. All paths consume identical
forced tokens and use the same checkpoint within each model.

| Model | Path | Decode tokens/s | Speedup vs uncached | K/V buffer MiB |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(measurements)}

GQA uses **{cache_reduction:.0%} less persistent K/V buffer storage** at identical
capacity and dtype. This does not imply the same reduction in total GPU memory.
Static caching's actual speed change versus dynamic concatenation is reported in
the raw grid; preallocation is not assumed to be faster for every workload.

![Inference throughput and cache storage](systems.png)

{graph_section}## Correctness and recovery

The published GPU test log records **159 passed / 1 skipped** (MPS unavailable);
the final local log records **156 passed / 5 skipped** (CUDA unavailable). The local
count includes one additional graph-record tampering test added after GPU validation.
All seven final losses were replayed from saved weights with a separate single-window
evaluation loop; see `raw/evaluation_replay.json` for numerical differences.

Tests cover causal invariance, manual/SDPA outputs and gradients, an independent
per-head GQA reference, dynamic/static multi-token cache equivalence, overflow
checks, optimizer equivalence, accumulation, immutable output protection and
complete disk recovery. CPU and CUDA recovery tests require exactly equal final
weights after an interrupted run. Verification logs state the actual tested
software/devices; cross-platform bitwise equality is not promised.

Raw numerical cache errors and distribution KL, CUDA allocation samples,
per-repeat timings, learning curves, full configurations, source hashes and
sampling examples are in `raw/`. See `verification.json` for the verification
scope and `artifact_checksums.json` for report-file hashes.

## Scope and honest claims

This is an AI-assisted personal engineering project. Manual attention and the
educational optimizer are project implementations. Optimized runs use PyTorch
SDPA and fused AdamW; no custom FlashAttention kernel is claimed. GQA is an
existing method. The small story model is not an instruction-tuned assistant.
Historical training-diagnostic experiments returned NOGO and remain documented
in `docs/RESEARCH_HISTORY.md`; no RSI or algorithm-leading claim is made.

Reproduction: `docs/PORTFOLIO_REPRODUCIBILITY.md`.
Code explanation: `docs/ARCHITECTURE.md`.
Interview preparation: `docs/INTERVIEW_GUIDE.md`.
"""
    (out / "REPORT.md").write_text(report_text)
    cv = f"""# BranchLab — 从零训练语言模型与高效推理系统

- 实现 byte-level BPE、decoder-only Transformer、RoPE、RMSNorm、SwiGLU、GQA 与手写 attention 参考路径；通过输出/梯度对照验证 PyTorch SDPA 路径，提供可阅读的张量形状与算法导读。
- 在 Colab NVIDIA L4 上训练 {showcase["parameters"] / 1e6:.2f}M 参数模型，处理 {showcase["trained_tokens"] / 1e6:.2f}M 训练 tokens；在独立文档的 65,536 个测试 target tokens 上达到交叉熵 {showcase["test"]["cross_entropy"]:.4f}、困惑度 {showcase["test"]["perplexity"]:.2f}，发布权重与复现配置。
- 实现动态与预分配 KV-cache，8→2 KV heads 将同容量持久 K/V 缓冲区减少 {cache_reduction:.0%}；公开普通缓存的有限加速收益及三组种子的等 token 质量对照。
{graph_cv}- 完成三组训练种子的等 token 预算 MHA/GQA 对照，实现梯度累积、BF16 训练与原子完整状态检查点，并以连续/中断恢复轨迹一致性测试和独立哈希/指标重算验证结果。

英文压缩版：

Built an explainable from-scratch language-model stack with BPE, RoPE, RMSNorm,
SwiGLU, GQA and verifiable attention/cache implementations; trained a
{showcase["parameters"] / 1e6:.2f}M-parameter TinyStories subset model on {showcase["trained_tokens"] / 1e6:.2f}M tokens.
Implemented resumable BF16 training, dynamic/static KV caching and three-seed
MHA/GQA comparisons, with 75% less persistent KV storage for 8→2 KV heads at equal capacity.

{graph_cv_en}
使用边界：以上指标来自本次特定配置。交叉熵仅代表该数据子集，解码提速不含 prefill。
按你实际掌握和承担的工作描述贡献；项目使用 AI 辅助，不宣称无辅助独立完成或原创 GQA。
"""
    (out / "CV_ZH_EN.md").write_text(cv)
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    hashes = {
        str(p.relative_to(out)): file_sha256(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "artifact_checksums.json"
    }
    (out / "artifact_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/portfolio")
    parser.add_argument("--output", default="reports/portfolio")
    parser.add_argument("--prepared")
    args = parser.parse_args()
    report(args.results, args.output, args.prepared)
