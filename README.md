# BranchLab

**From-scratch language modeling and counterexample-guided training diagnostics.**

Development branch: [the active goal](GOAL.md) continues after the v0.1.0 release. [Iteration 2](reports/development_v2/REPORT.md) also returned NOGO: conditional diagnostics retained the logs-only policy on 56 development states. Failed attempts are preserved in the [iteration ledger](protocols/ITERATION_LEDGER.json); no independent confirmation has been attempted.

BranchLab combines the engineering agenda of [Stanford CS336](https://cs336.stanford.edu/) with an audit of [Marin/Delphi's public experiments](https://huggingface.co/datasets/marin-community/delphi-blog-data). It implements a small language model, complete-state training branches, and a finite diagnostic-program search. The question is concrete: **does spending compute to diagnose a training state leave a better model than spending that compute on training?**

This is an independent, AI-assisted research engineering project. It is not an official Stanford assignment solution or a reproduction of Marin's large-model training.

## What is implemented

- Byte-level BPE; causal decoder-only Transformer; RoPE, RMSNorm and SwiGLU; an explicit AdamW implementation; clipping, learning-rate scheduling, generation and KV-cache decoding.
- Complete checkpoint restoration: model, optimizer moments/step, schedule, Python/NumPy/Torch/device RNG, and training-data identity/cursor.
- Three controlled training states: clean, 4× learning rate and reversed first moment. Each repair starts from the same saved state and receives the same training batches.
- An 18-production diagnostic grammar: `{halve LR, clear first moment}` × `{2, 4, 8 updates}` × `{loss contrast, recovery contrast, update alignment}`. A program accumulates at most two probes and predicts which repair to apply.
- Counterexample-ordered, random-ordered and enumerated program search under the same **probe-cell** cap; passive, logs-only, fixed-expert and direct short-trial baselines. Every deployed probe is deducted from a separate 160-unit training budget.
- Pinned public-source audit, real training logs, branch tables, search ledgers, reproducible reports and release artifacts.

The innovation explored here is a small, inspectable search over **diagnostic experiments**, with inherited probe programs and explicit opportunity cost. The grammar, ridge estimator and search algorithm stay fixed. Program accumulation alone is not recursive self-improvement. Closely related work and the limits of the novelty claim are documented in [RELATED_WORK](docs/RELATED_WORK.md).

## Measured release

**v0.1.0 result: the diagnostic-advantage gate is NOGO.** Logs-only repair selection reached **3.416762** mean final test cross-entropy; counterexample, random and enumerated search each reached **3.451266**. Their fixed-horizon repair regret was identical (**0.002045**). Here the diagnostic adds cost without improving held-out repair selection; full RSI was not tested.

The 19M training run processed **1,228,800 tokens**, with fixed-window development cross-entropy **6.381 → 2.045**. CPU KV-cache decoding measured **155.55 vs 58.72 tokens/s (2.65×)** for batch 1, a 64-token prompt and 32 forced continuation tokens, using five repeats. Local implementation checks: **69 passed, 1 CUDA test skipped**.

Actual metrics and the mechanism verdict are generated from the released records in [the experiment report](reports/release/REPORT.md). The run uses a 19,140,096-parameter showcase model and a separate ~1M-parameter diagnostic model. No benchmark score or improvement is borrowed from another project.

![Showcase training curve](reports/release/training_curve.png)

![Fixed-budget audit outcomes](reports/release/audit_loss.png)

The source audit read all six pinned Delphi tables: **4,117 rows**, mixing run results, fitted values and external evaluations. They do not supply complete restorable trajectories. Accordingly, Marin supports the public-data audit and engineering motivation; all intervention evidence comes from BranchLab's own small-model runs. No empirical benefit from a Marin-derived policy prior is claimed. See [source audit](reports/source_audit.md).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
branchlab prepare
branchlab train --device auto
branchlab pilot --device cpu
branchlab benchmark --device cpu --output artifacts/showcase/benchmark.json
branchlab report
branchlab generate --device cpu --tokens 64
python -m pytest -q
```

The data preparation downloads a pinned ~10 MB TinyStories parquet. CPU and Apple MPS are supported; the published run required no rented GPU or external model API. Use a new `--output` directory to repeat the pilot without replacing evidence. Full [reproduction instructions](docs/REPRODUCIBILITY.md) cover checkpoints, data partitioning, dependencies, costs and analysis replay.

Downloads: [v0.1.0 release](https://github.com/liyuqin606-del/branchlab/releases/tag/v0.1.0) includes inference weights/tokenizer, full trainer checkpoints, experimental records and SHA256 checksums. Source and lightweight reports are in this repository.

```bash
# After installing BranchLab, try the released model without retraining:
gh release download v0.1.0 --repo liyuqin606-del/branchlab -p branchlab-v0.1.0-model.tar.gz
tar -xzf branchlab-v0.1.0-model.tar.gz
branchlab generate --model branchlab-19m --device cpu --tokens 64
```

## Evidence boundaries

The pilot uses eight training initializations, three controlled states per seed, and only three held-out audit seeds. Per-branch loss evaluation uses one fixed 256-token dev batch and one test batch. The data is a local repartition of the official TinyStories validation file, deduplicated only by exact normalized document hash. This is an exploratory engineering experiment, not a powered benchmark or official TinyStories result.

Program selection uses discovery/development seeds and dev-text loss; final text-test loss never enters the predictor or probe search. Selection regret at 24 updates and final test loss after paying for probes answer different questions. The budget's forward-batch equivalents are an analytic proxy, not measured FLOPs. Equal probe-cell caps do not imply equal search FLOPs; actual query cost and physical table-collection work are reported separately.

The [frozen protocol](protocols/pilot_v1.md) admits a diagnostic advantage only when it beats every prespecified comparator in every audit seed. A failed gate is published as a negative/inconclusive result. Even a passing pilot would not demonstrate full RSI: improving the synthesizer itself and measuring its descendants' ability to improve would require a separate experiment.

## 中文说明

这是一个可复现的“从零训练 + 训练状态分支实验 + 诊断程序合成”项目。创新点放在：从错误修复案例中选择短探针，保存可继承的诊断程序，并检验探针节省的试错是否抵得上它消耗的训练预算。完整实现、实际结果和未通过的假设均公开，RSI 只作为后续可检验方向。

简历表述见 [实际指标版中文项目条目](reports/release/resume_zh.md)。请结合自己实际掌握和承担的工作使用，勿将 AI 辅助实现写成未经辅助的独立完成。

Code: [MIT](LICENSE). Dataset and dependency attribution: [THIRD_PARTY](THIRD_PARTY.md). Checkpoint scope and limitations: [model card](docs/MODEL_CARD.md).
