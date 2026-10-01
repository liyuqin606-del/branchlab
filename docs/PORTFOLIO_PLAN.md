# Portfolio delivery contract

Requested 2026-10-01: turn BranchLab into a complete, readable, explainable,
verifiable personal language-model engineering project. Colab Pro+ CLI use is
part of the requested execution. Historical diagnostic experiments and their
negative findings remain unchanged in their reports and research goal.

## Deliverables

- Educational manual Transformer attention plus a numerically checked PyTorch
  SDPA backend. Explicit grouped-query projections and per-request static cache.
- Fixed-token-budget training with BF16, gradient accumulation, deterministic
  data cursor, periodic atomic full-state checkpoints and exact resume checks.
- Pinned TinyStories training-shard subset; train-only tokenizer; exact
  normalized document exclusion across train/development/test partitions.
- Three prespecified paired seeds for MHA/GQA, identical token budgets. Report
  parameter differences and compute differences; do not call this equal-FLOP.
- Seven-repeat benchmark grid, same weights/tokens, explicit prefill/decode
  boundaries, raw timings, distribution parity and actual buffer allocation.
- Published inference weights, full showcase trainer state, raw experiment
  records, checksums, runnable Colab notebook, CLI/demo and generated CV copy.
- Architecture walkthrough and interview guide explaining tensor shapes,
  causality, GQA, caching, accumulation and restoration.

## Evidence and budget

Hardware preflight: local Apple M1 / 16 GiB; allocated NVIDIA L4 / ~22 GiB,
PyTorch 2.11.0+cu128, Python 3.13.15, BF16 supported. Account read before
allocation: 588.53 compute units, no active assignments. A single project-owned
session is used and must be stopped after artifacts have been recovered.

A synthetic-data preflight determines feasibility. Main study is 3 seeds × 2
variants × 800 updates, plus an independently declared GQA showcase at 2400
updates. Effective batch: 16 × 4 accumulation × 256 = 16,384 tokens/update.
Total planned training: 117,964,800 tokens. The job has a 7200-second limit;
no unattended background run is left at handoff. These are planned counts until
completion records confirm them.

Tests verify implementation; training logs establish observed training; final
held-out loss measures this data subset; benchmark speedups apply only to their
reported hardware and workload. No SOTA, significance, novel GQA, custom
FlashAttention or recursive self-improvement claim is part of this delivery.
