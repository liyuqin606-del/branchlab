# Reproduce and inspect

Use Python 3.10+ (the released run used Python 3.12.14). Install a platform-appropriate PyTorch wheel, then `pip install -e '.[dev]'`. No external language-model API or paid compute is needed.

```bash
branchlab prepare
branchlab train --config configs/showcase.json --device auto
branchlab pilot --config configs/pilot.json --device cpu
branchlab benchmark --model artifacts/showcase --device cpu --output artifacts/showcase/benchmark.json
branchlab report
python -m pytest -q
python scripts/audit_public_data.py
```

Run from the repository root because default configuration/data paths are relative to it. `prepare` downloads a pinned ~10 MB TinyStories file, verifies its size/SHA256, deduplicates exact normalized documents, and creates 6,000/600/600 train/dev/test document partitions. It trains BPE only on the local training partition. The vocabulary is 512 tokens. Near-duplicate leakage is not audited. These partitions come from the official validation file and are **not an official TinyStories benchmark split**.

`train` saves an inference artifact, tokenizer, a complete trainer checkpoint, configuration and training history. `pilot` trains eight independent small baselines, saves their complete checkpoints, executes all three repair branches for each controlled state, and then replays the diagnostic searches. A completed pilot output directory is protected from overwrite; use `--output artifacts/pilot-repeat` for another run. All branch labels have a real collection cost in `collection.json`.

The complete trainer snapshot includes model weights, optimizer moments and step, scheduler metadata, data hash/cursor and Python/NumPy/Torch/device RNG. The baseline schedule explicitly finishes by setting a fixed learning rate for diagnostic continuation. The snapshot is at that transition, not a promise to continue the prior cosine schedule. `capture_state`/`restore_state` support an actual scheduler object too; disk round-trip tests verify identical subsequent updates on the tested backend.

Trainer snapshots contain Python/NumPy RNG state and therefore require `torch.load(..., weights_only=False)`; load them only from a trusted source and verify the release hash. The inference `model.pt` uses `weights_only=True`. The eight pilot checkpoints additionally save the exact document order and last clipped gradient used by alignment probes. Recreate token arrays using the released tokenizer/preparation before restoring a stream; a mismatching stream hash raises an error.

CPU results should reproduce closely with the tested software and thread count. Different BLAS, PyTorch releases, devices or floating-point kernels can change exact numbers and near-tie decisions. MPS speed measurements are hardware-specific. The benchmark uses the same forced continuation in both paths, five timed repeats after warmup, explicit synchronization, and separate prefill/decode measurements. CPU/MPS allocator peak memory is not available in this implementation.

The pilot protocol and source hashes are frozen before outcome collection in `freeze.json`; this is a local protocol record, not an independent preregistration. Later packaging/report fixes are recorded separately from the source that generated the results. `episodes.json` contains probe features and fixed-horizon dev labels; final text-test scores stay in `curves.json` and cannot enter program fitting. Search ledgers record every revealed probe cell and promotion. Replaying analysis cannot be reported as avoiding the physical table construction.

The GitHub release provides model weights, full trainer checkpoints, experiment records and hashes. The source repository includes compact reports. Published test counts describe implementation checks, not scientific evidence of diagnostic or RSI success.
