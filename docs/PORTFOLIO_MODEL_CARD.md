# BranchLab v0.2 model card

## Intended use

Educational exploration of small language-model training, grouped-query
attention, inference caching and reproducible performance measurements. The
model generates short English stories. It is not trained as an assistant and
is not evaluated for factual answers, code generation or instruction following.

## Architecture

Decoder-only Transformer: six layers, width 512, eight query heads, two KV heads
for the showcase; RoPE, pre-normalization RMSNorm, SwiGLU, tied embeddings and a
2048-token byte-level BPE vocabulary. Maximum context is 512 tokens; training
windows are 256 tokens. GQA comparison and MHA configurations are distributed.

## Training and evaluation

See `reports/portfolio/REPORT.md` and its raw configurations for actual parameter
count, token budgets and observed loss. Sources are pinned to a specific
TinyStories dataset revision. Training uses 50,000 selected documents from one
training shard; BPE fitting uses 10,000 of those documents. Development and test
use 512 documents each from official validation. Exact normalized document
leakage is removed; semantic/near-duplicate leakage is not audited.

Three paired seeds compare MHA/GQA. The showcase is chosen before comparison
outcomes and trained separately with a larger budget. Test evaluations use
65,536 nonrepeated target tokens and fixed 256-token contexts. The records
support this subset experiment only, not the full official validation split.

## Limitations

Small vocabulary, synthetic English story data, small model, fixed context and
limited training. Samples may repeat, contradict themselves or emit malformed
text. No factuality, bias, safety, multilingual, reasoning, or out-of-domain
benchmark is claimed. Fixed prompt examples are all retained; they are not a
representative human-evaluation study. Data is packed across EOS boundaries.

## Release and provenance

Inference weights load with `weights_only=True`. Full checkpoints additionally
contain Python/NumPy RNG state and must be loaded only from a trusted verified
source. SHA256SUMS covers release bundles; model and data hashes are recorded
per run. Both training and evaluation code are available. Training results are
observed on NVIDIA L4; cross-device equality and performance are not guaranteed.

Project code is MIT. TinyStories source attribution and its CDLA-Sharing-1.0
license are retained in THIRD_PARTY.md; prepared token arrays derive from that
data. Model creation and project implementation used AI assistance.
