"""Render a release report from measured JSON artifacts without inferring gains.

``build_report(args)`` is the CLI adapter; args supplies pilot, showcase, and
output directories. ``render_report`` and ``render_resume`` are pure functions.
Plots use per-training-seed aggregates, never treat multiple episodes or search
seeds as independent training repeats, and distinguish descriptive SD from CI.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Mapping, Sequence


MISSING = "未提供"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric measurement, got {value!r}")
    return float(value)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _number(value)
    if number is None:
        return MISSING
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _cell(value: Any) -> str:
    if value is None:
        return MISSING
    return str(value).replace("|", "\\|").replace("\n", " ")


def _methods(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    methods = summary.get("methods", [])
    if not isinstance(methods, list) or any(not isinstance(m, dict) for m in methods):
        raise ValueError("summary.methods must be a list of method objects")
    return methods


def _method_label(method: Mapping[str, Any]) -> str:
    name = str(method.get("name", "unnamed method"))
    if method.get("search_seed") is not None:
        name += f" / search seed {method['search_seed']}"
    return name


def audit_series(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One plotted entry per method/search seed; mean over unique audit seeds."""
    result = []
    for method in _methods(summary):
        rows = method.get("per_seed", [])
        if not rows:
            continue
        seeds: list[Any] = []
        losses: list[float] = []
        for row in rows:
            if "seed" not in row:
                raise ValueError("Each per_seed aggregate needs a seed identifier")
            seed = row["seed"]
            if seed in seeds:
                raise ValueError("Duplicate per_seed records would inflate the plotted repeat count")
            value = _number(row.get("test_loss"))
            if value is None:
                raise ValueError("per_seed.test_loss is required when plotting an audit seed")
            seeds.append(seed)
            losses.append(value)
        result.append({"label": _method_label(method), "seeds": seeds, "losses": losses,
                       "mean": statistics.mean(losses),
                       "sd": statistics.stdev(losses) if len(losses) > 1 else None})
    return result


def render_report(summary: Mapping[str, Any], showcase: Mapping[str, Any] | None = None,
                  benchmark: Mapping[str, Any] | None = None, *,
                  charts: Sequence[str] = ()) -> str:
    """Render exactly the supplied metrics; absent measurements remain absent."""
    methods = _methods(summary)
    gate = summary.get("gate") or {}
    collection = summary.get("collection") or {}
    config = summary.get("config") or {}
    scope = summary.get("scope", MISSING)
    if isinstance(scope, (dict, list)):
        scope = json.dumps(scope, ensure_ascii=False, sort_keys=True)
    lines = ["# BranchLab 实测发布报告", "",
             f"运行状态：**{_cell(summary.get('status'))}**。预设 gate：**{_cell(gate.get('status'))}**。", "",
             f"判定原因：{_cell(gate.get('reason'))}", "", f"实验范围：{scope}", "",
             "本报告从随附 JSON 生成；未提供的指标明确留空。软件测试通过、一次小模型训练成功和研究假设成立是不同结论。", "",
             "## 从零模型训练", ""]
    if showcase is None:
        lines += ["未提供 showcase/run.json，不能填写训练参数量、token 数或 loss。", ""]
    else:
        lines += ["| 项目 | 实测记录 |", "| --- | --- |",
                  f"| 参数量 | {_fmt(showcase.get('parameters'), 0)} |",
                  f"| 训练 token | {_fmt(showcase.get('trained_tokens'), 0)} |",
                  f"| 初始开发集交叉熵 | {_fmt(showcase.get('initial_dev_loss'))} |",
                  f"| 最终开发集交叉熵 | {_fmt(showcase.get('final_dev_loss'))} |",
                  f"| 训练调用总耗时 / s | {_fmt(showcase.get('elapsed_seconds'), 3)} |",
                  f"| 设备 | {_cell(showcase.get('device'))} |",
                  f"| PyTorch | {_cell(showcase.get('torch_version'))} |", "",
                  "这里报告开发集指标，不能改写为最终测试集成绩。训练样本规模和完整配置见 [run.json](run.json)。", ""]
    if "training_curve.png" in charts:
        lines += ["![Observed training history](training_curve.png)", "",
                  "曲线只连接 history.json 中实际记录的开发集评估点，不插入额外测量。", ""]

    lines += ["## KV cache 测量", ""]
    if benchmark is None:
        lines += ["未提供 benchmark.json，不能填写缓存提速或 logits 误差。", ""]
    else:
        meta = benchmark.get("metadata") or {}
        lines += ["| 项目 | 实测记录 |", "| --- | --- |",
                  f"| Cached decode / tokens·s⁻¹ | {_fmt(benchmark.get('cached_decode_tokens_per_second'), 3)} |",
                  f"| Uncached decode / tokens·s⁻¹ | {_fmt(benchmark.get('uncached_decode_tokens_per_second'), 3)} |",
                  f"| Decode 倍率 | {_fmt(benchmark.get('decode_speedup'), 3)} |",
                  f"| Prefill / s | {_fmt(benchmark.get('prefill_seconds'))} |",
                  f"| Logits 最大绝对差 | {_fmt(benchmark.get('logits_max_diff'), 9)} |",
                  f"| 设备 / 精度 | {_cell(meta.get('device'))} / {_cell(meta.get('dtype'))} |",
                  f"| Batch / prompt / decode长度 | {_fmt(meta.get('batch_size'), 0)} / {_fmt(meta.get('prompt_tokens'), 0)} / {_fmt(meta.get('decode_tokens_per_sequence'), 0)} |",
                  f"| 重复次数 | {_fmt(meta.get('repeats'), 0)} |", "",
                  "测速使用相同固定 token 续写、预热和设备同步，prefill 与 decode 分开计时。倍率只适用于该模型和配置，不计作训练或 RSI 收益。", ""]
        peak = benchmark.get("peak_memory") or {}
        if peak.get("available") is True:
            lines += [f"峰值内存指标：{_cell(peak.get('metric'))}；cached {_fmt(peak.get('cached_bytes'), 0)} bytes，uncached {_fmt(peak.get('uncached_bytes'), 0)} bytes。", ""]
        else:
            lines += [f"可比的逐次 allocator 峰值内存：未提供。原因：{_cell(peak.get('unavailable_reason'))}。", ""]

    episode_count = summary.get("episodes")
    if isinstance(episode_count, list):
        episode_count = len(episode_count)
    lines += ["## 训练诊断 pilot", "",
              f"Episode 数：{_fmt(episode_count, 0)}；audit seeds：{_cell(summary.get('audit_seeds'))}。", "",
              "| 方法 / 搜索种子 | 已选探针 | 固定预算测试 loss | 固定期限开发 regret | 平均诊断成本 | 平均修复步数 | 搜索揭示 cells | 搜索重放成本 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        probes = ", ".join(method.get("selected_probes") or []) or "无"
        lines.append(f"| {_cell(_method_label(method))} | {_cell(probes)} | "
                     f"{_fmt(method.get('mean_test_loss'))} | {_fmt(method.get('mean_regret'))} | "
                     f"{_fmt(method.get('mean_probe_cost'), 2)} | {_fmt(method.get('mean_repair_steps'), 2)} | "
                     f"{_fmt(method.get('search_revealed_cells'), 0)} | {_fmt(method.get('search_replay_probe_cost'), 2)} |")
    if not methods:
        lines.append(f"| {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} |")
    lines += ["", "表格使用 summary.json 的原始汇总，两个损失指标对应不同测量：", "",
              f"- regret 使用固定 {_fmt(config.get('repair_horizon'), 0)} 次更新后的开发文本 loss，以受限动作菜单中的最低 loss 为参照；这些状态来自 holdout 训练 seeds。它不是最终测试文本 regret，也未扣除诊断预算。",
              f"- 测试 loss 使用每个状态 {_fmt(config.get('audit_budget_forward_batches'), 0)} 个 forward-equivalent 代理单位的预算，先扣除诊断成本，再分配修复更新步数。它才是这次 gate 比较的结果指标。", "",
              "同名方法的不同 search seed 单独列出。counterexample/enumeration 当前为确定性排序，更换 search seed 不产生新算法重复。多个 episode 属于同一训练 seed 时也不是独立重复。完整动作、剩余预算与逐 seed 指标见 [summary.json](summary.json)。", ""]
    if any(m.get("name") == "full_enumeration" for m in methods):
        lines += [f"预算说明：counterexample、random 和 enumeration 的搜索上限为 {_fmt(config.get('search_budget_cells'), 0)} 个唯一 episode/probe cells；full_enumeration 使用 100,000-cell 上限来提供完整候选参考，**不是同搜索预算对照**。实际揭示量按各行原样报告，不把上限当消耗。", ""]
    batch_size, seq_len = _number(config.get("batch_size")), _number(config.get("seq_len"))
    if batch_size is not None and seq_len is not None:
        lines += [f"每次分支评估只用一个固定开发批次和一个固定测试批次，各 {_fmt(batch_size * seq_len, 0)} tokens。这个很小的评估窗口限制结果稳定性与外推范围。", ""]
    if "audit_loss.png" in charts:
        lines += ["![Audit losses aggregated by training seed](audit_loss.png)", "",
                  "图中每项为该方法的 audit-seed 宏平均；点代表一个训练 seed，误差棒为 seed 间样本标准差，并非置信区间。只有一个 seed 时不画误差棒。不同搜索种子不合并为额外训练重复。", ""]
    comparisons = gate.get("comparisons")
    if comparisons:
        lines += ["Gate 比较明细（按输入保留）：", "", "```json",
                  json.dumps(comparisons, ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
    lines += ["## 成本与失败记录", "",
              "| 实际全表采集记录 | 数值 |", "| --- | ---: |",
              f"| 总耗时 / s | {_fmt(collection.get('elapsed_seconds'), 3)} |",
              f"| Training updates | {_fmt(collection.get('training_updates'), 0)} |",
              f"| Evaluation batches | {_fmt(collection.get('evaluation_batches'), 0)} |",
              f"| 训练 token | {_fmt(collection.get('trained_tokens'), 0)} |", "",
              "全表采集是已实际发生的计算。搜索重放成本是按查询账本模拟的可见信息开销；测试期的诊断开销另计。失败/丢弃分支也属于真实采集成本。读出共享分支不代表已实现对应的端到端速度节省。", "",
              "如使用 optimizer-update 或 forward-equivalent 预算，它是报告中明确限定的计算代理量，不能替代完整生命周期或硬件耗时比较。", ""]
    failures = summary.get("failures")
    if failures is None:
        lines += ["失败清单：未提供。", ""]
    elif not failures:
        lines += ["输入失败清单为空；这只描述该清单，不保证覆盖尚未记录的失败。", ""]
    else:
        lines += ["记录的失败：", "", "```json", json.dumps(failures, ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
    if summary.get("limitations"):
        lines += ["运行保存的限制：", ""]
        lines += [f"- {_cell(item)}" for item in summary["limitations"]]
        lines.append("")
    lines += ["## 可支持的结论", "",
              f"本次 gate 状态为 **{_cell(gate.get('status'))}**，原因是：{_cell(gate.get('reason'))}", "",
              "PASS_EXPLORATORY 仅表示预设 pilot gate 的探索性通过；NOGO 表示未通过该 gate，均不自动构成显著性、SOTA、跨规模复现或新颖性证明。", "",
              "此实现继承已选诊断程序，合成器规则本身保持固定。没有完成旧/新改进器产生后代的受控比较，因此不宣称验证递归自我改进或 full RSI。Marin 公开资料提供问题动机，本地分叉提供本实验的实际干预证据。", "",
              "输入 JSON 和生成文件的 SHA-256 见 [report_manifest.json](report_manifest.json)。"]
    return "\n".join(lines) + "\n"


def render_resume(summary: Mapping[str, Any], showcase: Mapping[str, Any] | None = None,
                  benchmark: Mapping[str, Any] | None = None) -> str:
    """Measured, deliberately bounded Chinese project bullets for review."""
    lines = ["# BranchLab：从零实现 Transformer 与训练诊断程序搜索实验", "",
             "以下条目只引用随附产物；使用时保留测量条件与结果边界。", "",
             "- 实现 byte-level BPE、causal attention、RoPE、RMSNorm、SwiGLU、AdamW、完整训练恢复和 KV cache，并建立自动化正确性检查。"]
    if showcase is not None:
        lines.append(f"- 在报告记录的数据子集上训练 {_fmt(showcase.get('parameters'), 0)} 参数模型，共 {_fmt(showcase.get('trained_tokens'), 0)} tokens；开发集交叉熵 {_fmt(showcase.get('initial_dev_loss'))} → {_fmt(showcase.get('final_dev_loss'))}，设备 {_cell(showcase.get('device'))}。")
    else:
        lines.append("- 训练参数量、token 数与 loss 尚未随报告提供；暂不填写训练效果数字。")
    if benchmark is not None:
        meta = benchmark.get("metadata") or {}
        lines.append(f"- 在 {_cell(meta.get('device'))} / {_cell(meta.get('dtype'))}、batch {_fmt(meta.get('batch_size'), 0)}、prompt {_fmt(meta.get('prompt_tokens'), 0)} / decode {_fmt(meta.get('decode_tokens_per_sequence'), 0)} 的固定测量中，uncached/cached 吞吐分别为 {_fmt(benchmark.get('uncached_decode_tokens_per_second'), 2)} / {_fmt(benchmark.get('cached_decode_tokens_per_second'), 2)} tokens/s，decode 倍率 {_fmt(benchmark.get('decode_speedup'), 3)}；logits 最大绝对差 {_fmt(benchmark.get('logits_max_diff'), 9)}。")
    lines.append("- 实现 18 项短干预探针的有限诊断语言、反例优先搜索及最多两个探针的继承，隔离发现/开发/测试状态，与免费日志、固定专家、随机搜索和枚举比较；查询成本与真实全分支采集成本分别记录。")
    gate = summary.get("gate") or {}
    lines.append(f"- 该次 pilot 的预设 gate 为 {_cell(gate.get('status'))}：{_cell(gate.get('reason'))}。保留原始结果、动作记录和失败清单，结论限定于本次训练配置。")
    lines += ["", "不将工程完成写成算法领先；不将开发集成绩写成测试集成绩；不宣称复现 Marin 大模型或验证 full RSI。", ""]
    return "\n".join(lines)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def _training_chart(history: Any, path: Path) -> bool:
    if not isinstance(history, list):
        raise ValueError("history.json must contain a list of measured training records")
    points = []
    for item in history:
        if item.get("dev_loss") is not None:
            step, loss = _number(item.get("step")), _number(item["dev_loss"])
            if step is None:
                raise ValueError("Each training evaluation requires a recorded step")
            points.append((step, loss))
    if not points:
        return False
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    points.sort()
    ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color="#236e9a", linewidth=1.8, markersize=4)
    ax.set(xlabel="Optimizer update", ylabel="Development cross-entropy",
           title="From-scratch LM: recorded development evaluations")
    ax.grid(alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _audit_chart(summary: Mapping[str, Any], path: Path) -> bool:
    series = audit_series(summary)
    if not series:
        return False
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.2, max(4.2, 0.47 * len(series) + 1.7)), layout="constrained")
    for i, entry in enumerate(series):
        offsets = [(j - (len(entry["losses"]) - 1) / 2) * 0.055 for j in range(len(entry["losses"]))]
        ax.scatter(entry["losses"], [i + o for o in offsets], s=24, color="#6d8491", alpha=0.65,
                   zorder=2, label="One audit training seed" if i == 0 else None)
        ax.errorbar(entry["mean"], i, xerr=entry["sd"], fmt="D", color="#c3532d",
                    capsize=4, markersize=5, zorder=3,
                    label="Seed mean +/- sample SD (not CI)" if i == 0 else None)
    ax.set_yticks(range(len(series)), [e["label"] for e in series])
    ax.invert_yaxis()
    ax.set(xlabel="Held-out cross-entropy (lower is better)",
           title="Diagnostic pilot: macro-average over audit training seeds")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _read_optional(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def build_report(args: Any) -> dict[str, str]:
    """CLI adapter: copy measured evidence, plot it, and write a hashed report."""
    pilot, showcase_dir, output = Path(args.pilot), Path(args.showcase), Path(args.output)
    sources = {"summary.json": pilot / "summary.json", "run.json": showcase_dir / "run.json",
               "history.json": showcase_dir / "history.json", "benchmark.json": showcase_dir / "benchmark.json"}
    if not sources["summary.json"].is_file():
        raise FileNotFoundError(f"A completed pilot summary is required: {sources['summary.json']}")
    data = {name: _read_optional(path) for name, path in sources.items()}
    summary, run, benchmark = data["summary.json"], data["run.json"], data["benchmark.json"]
    # Validate scalar reporting and seed aggregation before copying any artifact.
    render_report(summary, run, benchmark)
    audit_series(summary)
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, source in sources.items():
        if data[name] is not None:
            target = output / name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            copied.append(name)
    charts = []
    if data["history.json"] is not None and _training_chart(data["history.json"], output / "training_curve.png"):
        charts.append("training_curve.png")
    if _audit_chart(summary, output / "audit_loss.png"):
        charts.append("audit_loss.png")
    (output / "REPORT.md").write_text(render_report(summary, run, benchmark, charts=charts), encoding="utf-8")
    (output / "resume_zh.md").write_text(render_resume(summary, run, benchmark), encoding="utf-8")
    generated = charts + ["REPORT.md", "resume_zh.md"]
    manifest = {"format_version": 1, "inputs": {name: _digest(output / name) for name in copied},
                "missing_optional_inputs": [name for name, value in data.items() if value is None],
                "generated": {name: _digest(output / name) for name in generated},
                "note": "Hashes bind measured inputs and generated outputs; they do not establish scientific validity."}
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {"report": str(output / "REPORT.md"), "resume": str(output / "resume_zh.md"),
              "manifest": str(output / "report_manifest.json")}
    print(json.dumps({"event": "report_written", **result}, ensure_ascii=False))
    return result
