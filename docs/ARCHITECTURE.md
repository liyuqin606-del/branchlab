# Read the model from tokens to logits

Start with `src/branchlab/model.py`, then `fit.py`, then `systems.py`. These are
three different questions: what is computed, how it is trained, and how it is
measured. The older diagnostic research code is an optional advanced appendix.

## 1. Tokens and targets

Byte-level BPE starts with every UTF-8 byte and an end-of-document token. Merges
replace frequent adjacent pairs inside whitespace/non-whitespace pieces. The
vocabulary therefore represents unseen Unicode without an unknown-token rule.
This pretokenizer is deliberately simpler than GPT-2's: we do not claim exact
GPT-2 tokenization. The tokenizer learns only from a declared training subset.

Packed documents are separated by EOS. For tokens `[a,b,c,d]`, inputs are
`[a,b,c]` and targets `[b,c,d]`. Packing allows attention across EOS; EOS is a
learned boundary rather than an enforced document attention mask. Evaluation
uses a fixed count of nonrepeated targets and resets context at window borders.

## 2. Decoder block

For input `x` of shape `(B,T,D)`, each block computes:

```
x = x + Attention(RMSNorm(x))
x = x + Down(SiLU(Gate(RMSNorm(x))) * Up(RMSNorm(x)))
```

RMSNorm scales a vector by the reciprocal root mean square, followed by a
learned weight. SwiGLU uses a gated feedforward expansion. RoPE rotates pairs
of query/key dimensions using their absolute token positions; cached tokens
keep their original rotated keys. Final norm and a vocabulary projection turn
hidden states into `(B,T,V)` logits. Input/output embeddings share one weight.

## 3. MHA versus GQA

Queries have shape `(B,Hq,T,Dh)`. Keys and values have shape `(B,Hkv,T,Dh)`.
MHA uses `Hkv=Hq`; our GQA comparison uses eight query heads and two KV heads.
Every group of four query heads shares one KV head. Packed Q/K/V projections
make the parameter saving visible: `D * (Hq+2*Hkv) * Dh`, rather than `3D²`.

The manual backend explicitly expands the shared KV heads, computes scaled
scores, applies the causal mask, performs softmax and combines values. SDPA
uses the same projected tensors with PyTorch's optimized attention function.
The implementation also explicitly expands GQA for portability; temporary
attention tensors are not reduced by the same factor as persistent cache.
Neither backend is a custom CUDA/FlashAttention kernel written by the author.

## 4. Causal masking with a cache

With `P` cached positions and `T` new positions, a query at local position `i`
may see key positions `j <= P+i`. A plain upper-left triangular mask is wrong
for multi-token cached chunks. A one-token append may see the entire prefix.
Tests compare arbitrary chunkings with one full causal forward pass and compare
both parameter gradients and logits.

Dynamic caches concatenate every new key/value onto their prefix, allocating
new storage. Static caches reserve `(B,Hkv,capacity,Dh)` once, append in place,
and expose only the filled prefix. They are inference-only and private to a
request. Context/capacity checks occur before a block mutates its cache.

Persistent cache bytes are `2 * layers * B * Hkv * capacity * Dh * dtype_bytes`.
Reducing KV heads from eight to two therefore reduces that buffer by 75% for
identical capacity/dtype. This identity does not imply a 75% reduction in total
GPU memory or a fourfold speedup. Those are separate measured quantities.

## 5. Training and recovery

Each optimizer update averages four equal-size microbatch losses. Backward
runs after each divided loss; clipping and AdamW occur once after accumulation.
Parameters and optimizer moments stay FP32; CUDA autocast performs supported
operations in BF16. BF16 does not require the FP16 loss-scaler path, which this
runner does not implement. The educational AdamW is checked against PyTorch;
large portfolio runs use PyTorch's fused AdamW, explicitly recorded in each run.

Learning rate is a function of the original update index and original budget:
linear warmup followed by cosine decay to a declared minimum ratio. An
interruption does not restart warmup or shorten the cosine schedule.

A complete checkpoint records weights, moments, update count, schedule, input
cursor/data hash, Python/NumPy/Torch/CUDA RNG and accumulated history. Saves use
a temporary path and atomic rename at update boundaries. Resume rejects
configuration, data/tokenizer and source mismatches. Exact equality is tested
on CPU; CUDA recovery is tested on the allocated backend too. Cross-platform
bitwise equality is not promised.

## 6. Measure without changing the task

All three decode paths consume the same forced continuation. No sampled output
changes their workload. Correctness checks precede timing. Seven repeats rotate
path order; synchronized wall time prevents timing only asynchronous launches.
Prefill includes static allocation and is reported separately. Cache allocation
counts the actual K/V buffers; CUDA peak allocation counts all live tensors.
Raw samples remain available, so medians and speedups can be recomputed.

A performance improvement and a language-quality improvement are different
claims. MHA/GQA use matched tokens, width, depth and seeds, but different
parameter/FLOP counts. Three seeds show variation; they are insufficient for a
broad statistical superiority claim.

## 7. Fixed-shape CUDA graph decoding

The standalone `scripts/cuda_graph_decode.py` extension keeps the original
training code and its source fingerprints unchanged. It expresses one decoding
step using a fixed token buffer, full-capacity K/V tensors and a device-side
position scalar. That scalar selects the cache write slot, RoPE rotation and
visible-key mask. Tensor shapes and memory addresses stay fixed across steps.
A CUDA graph captures the operation sequence once and replays it on new token
values. There is no `.item()` or tensor-dependent Python branch inside capture.

The decoder owns its buffers, serves one request at a time and must be reset
between requests. Keep its model weights, dtype and device fixed for its entire
lifetime. Its output buffer is reused; clone outputs that must survive the next
step. Bounds are checked before replay. CPU eager tests and actual CUDA replay
tests compare multiple complete requests with the original full model.

Graph capture and warmup are one-time costs, reported separately. The steady
measurement includes token copies and graph replay, uses the same forced tokens
as ordinary dynamic caching, rotates trial order and reports every timing.
Graph-private intermediate storage is additional memory; the 75% GQA cache
reduction refers only to persistent K/V buffers, not the total graph pool.
