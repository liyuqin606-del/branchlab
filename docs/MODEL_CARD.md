# BranchLab TinyStories 19M — experimental model card

## Model and intended use

An independently implemented, AI-assisted decoder-only Transformer for studying training, checkpoint recovery and KV-cache inference. This is an undertrained educational base model, not an instruction model, assistant, Stanford model, or Marin checkpoint.

- Parameters: 19,140,096 trainable; six layers, width 512, eight attention heads.
- Components: RoPE, RMSNorm, SwiGLU, tied token/output embeddings, causal attention.
- Tokenizer: original byte BPE, 512 vocabulary entries, including EOS; trained only on the local training partition.
- Configured context: 256 tokens; actual training sequence length: 128 tokens.
- Training: AdamW, clipping, warmup/cosine schedule, 1,200 updates, batch size 8, 1,228,800 tokens, seed 42.
- Hardware/software: Apple Silicon M1, macOS arm64, PyTorch 2.8.0, Python 3.12.14, MPS float32. Runtime metadata and exact configuration are in the release.

## Data and evaluation

A pinned official TinyStories validation parquet was repartitioned locally into 6,000 training, 600 development and 600 test documents. The dataset card declares CDLA-Sharing-1.0. Corpus source attribution and SHA256 are included in the data preparation module and manifest. The training stream contains ~3.03M tokens; this run consumes only its first 1.23M tokens. Exact normalized document hashes are disjoint; near-duplicate overlap was not audited.

Recorded development cross-entropy fell from 6.381173 to 2.045100 nats/token. This is measured on the same fixed 2,048-token development window throughout training, not all 600 development documents. It is neither a standard benchmark result nor a reliable estimate of broad language quality. The separate diagnostic pilot uses its own ~1M models and reports different losses.

The sample file is a single greedy continuation of a fixed prompt, without selection among alternative outputs. Short plausible text does not establish general reasoning or instruction-following capability. Limited data, training and context can produce repetition, incoherence or inaccurate content.

## Artifacts and licensing

The inference archive contains `model.pt` (config plus state tensors), `tokenizer.json`, `run.json`, `history.json`, `benchmark.json`, `sample.txt` and this model card. `model.pt` is loaded with `weights_only=True`. Full trainer checkpoints are in a separate archive with optimizer moments, RNG and data cursor; consult REPRODUCIBILITY.md before loading them.

The project grants its own code and model-weight rights under MIT. This does not relicense the TinyStories corpus or third-party software. Dataset attribution and its original license remain documented in THIRD_PARTY.md. The raw corpus is not redistributed.

No safety alignment or deployment suitability evaluation was performed. Intended use is local experimentation and teaching about the implementation, with the source limitations above.
