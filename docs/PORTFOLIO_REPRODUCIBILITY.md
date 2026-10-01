# Reproduce the portfolio project

Python 3.10+; CUDA portfolio run observed Python 3.13.15 / PyTorch 2.11.0+cu128.
CPU correctness also runs on Python 3.12 / PyTorch 2.8.0. Use a PyTorch wheel
appropriate to your device. BF16 training requires a supported CUDA GPU; CPU
BF16 is provided for small tests, and MPS uses FP32. Hardware/library changes
may affect speed and numerical results.

## Load released weights first

```bash
git clone --branch v0.2.0 https://github.com/liyuqin606-del/branchlab.git
cd branchlab
python -m pip install -e '.[dev]'
gh release download v0.2.0 -R liyuqin606-del/branchlab \
  -p branchlab-v0.2.0-inference.tar.gz -p SHA256SUMS
shasum -a 256 branchlab-v0.2.0-inference.tar.gz
# Compare the displayed hash with the matching SHA256SUMS entry before extraction.
mkdir -p artifacts
tar -xzf branchlab-v0.2.0-inference.tar.gz -C artifacts
branchlab generate --model artifacts/portfolio/showcase-gqa \
  --prompt 'Tom found a small red box in the garden.' \
  --tokens 128 --temperature 0.8 --top-k 40 --seed 42
```

For an interactive local interface:

```bash
python -m pip install -e '.[demo]'
python scripts/demo.py --model artifacts/portfolio/showcase-gqa
```

The demo binds localhost and does not create a public sharing endpoint.
The Colab notebook offers the same inference and benchmark flow.

## Train from scratch

```bash
branchlab prepare-official --output artifacts/official-data
python scripts/run_portfolio.py --preflight --output artifacts/portfolio-repeat
python scripts/run_portfolio.py --prepared artifacts/official-data \
  --output artifacts/portfolio-repeat --max-seconds 7200
python scripts/verify_portfolio.py --results artifacts/portfolio-repeat \
  --prepared artifacts/official-data
python scripts/report_portfolio.py --results artifacts/portfolio-repeat \
  --prepared artifacts/official-data --output reports/portfolio-repeat
```

The synthetic preflight measures resource use without evaluating held-out text.
Data sources are revision-pinned with size/SHA256 verification. All splits and
vocabulary are frozen before training. Three paired seeds receive the same
training-token budgets; the GQA showcase is predetermined and uses a separate
larger budget. No early stopping or test-based tuning is performed. The fixed
protocol records the exact source hashes and runner hash.

## Interrupt and resume one run

```bash
branchlab fit --config configs/portfolio/gqa.json \
  --prepared artifacts/official-data --output artifacts/recovery-demo \
  --device cuda --stop-after 100
branchlab fit --config configs/portfolio/gqa.json \
  --prepared artifacts/official-data --output artifacts/recovery-demo \
  --device cuda --resume artifacts/recovery-demo/checkpoint.pt
```

Only trusted full checkpoints may be loaded with `weights_only=False`; inference
weights use `weights_only=True`. Checkpoint resume requires identical source,
configuration, data/tokenizer hashes, schedule and stream. Changed precision,
optimizer or global budget is rejected. Saves occur at optimizer boundaries.
Completed runs cannot be overwritten. A second `run_portfolio.py` invocation
reuses completed runs and resumes partial ones from the last checkpoint.

The two-hour runner limit is checked between runs. For a hard process deadline
on Linux/Colab use `timeout 7200 python scripts/run_portfolio.py ...`; the last
atomic checkpoint remains recoverable even if the process is terminated.
Colab storage is ephemeral: download/back up complete checkpoints and prepared
arrays before releasing or losing the runtime.

## Recompute measurements independently

```bash
branchlab systems --model artifacts/portfolio/showcase-gqa --device cuda \
  --precision bf16 --batch-size 1 --prompt-tokens 256 --tokens 64 \
  --repeats 7 --output artifacts/benchmark-repeat.json
python -m pytest -q
```

Raw benchmark samples include warmup scope, precision, device, model hash,
fixed-token continuation, distribution parity, prefill/decode timing and memory
semantics. `verify_portfolio.py` recomputes all medians, throughput/speedups,
parameter counts and actual cache-size formulas. It verifies stored evidence;
it does not claim to re-run training. The separate `scripts/replay_evaluation.py` script replays test loss with an
independent single-window evaluation loop and the saved precision; its command
is shown below. It does not call the training evaluator.

## Historical research

`docs/RESEARCH_HISTORY.md`, `reports/release` and the development protocols retain
the previous training-diagnostic experiments and NOGO results. `GOAL.md` is the
preserved research contract; `docs/PORTFOLIO_PLAN.md` is this delivery's scope.
The new project does not depend on claiming a diagnostic or RSI advantage.

## Optional CUDA graph inference extension

After inspecting the ordinary-cache measurements, a separate inference-only
extension evaluates reusable fixed-shape CUDA graphs on the predetermined
showcase model. This leaves the training study and its outcomes unchanged.

```bash
python scripts/cuda_graph_decode.py --model artifacts/portfolio/showcase-gqa \
  --output artifacts/graph-repeat
python scripts/replay_evaluation.py --results artifacts/portfolio \
  --prepared artifacts/official-data --device cuda \
  --output artifacts/evaluation-replay.json
```

The graph script writes a separate protocol before measurements. Its fixed grid
uses batches 1/4, prompt lengths 64/256 and 64 forced continuation tokens, with
seven alternating-order trials. Capture/warmup time is reported separately from
steady-state decode. Reuse requires fixed model weights, device, dtype, batch
and capacity. A graph is not shared concurrently across independent requests.
