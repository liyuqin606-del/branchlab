# Explain this project in an interview

## A 60-second introduction

“I built a small decoder-only language-model training and inference stack. The
reference attention implementation exposes the Q/K/V tensors, causal mask and
RoPE positions; an SDPA backend is checked against it. I compared MHA and GQA
under fixed token budgets, then measured dynamic and preallocated caches on the
same trained weights. I also implemented complete-state checkpoint recovery and
published the configurations, raw results and checksums.”

Use the generated portfolio report for exact numbers. Do not memorize historical
CPU speedups as if they were the new GPU result. Describe AI-assisted work
accurately and be prepared to implement/explain each core component yourself.

## Questions you should be able to answer

1. Why does autoregressive training predict the next token rather than the current
   token? Derive the shifted cross-entropy inputs/targets.
2. What are the shapes of Q/K/V and the attention score matrix? Count the work
   and explain why long-context uncached decoding repeats expensive computation.
3. How does GQA change parameter count and cache storage? Why might it not yield
   a proportional speedup? Explain temporary KV expansion in this implementation.
4. Why are RoPE offsets essential when decoding? Draw a cached multi-token causal
   mask and explain why `is_causal=True` alone is insufficient there.
5. Why can an in-place static cache break autograd? Explain the inference-only
   guard, cache ownership, capacity and reset strategy.
6. When is gradient accumulation equivalent to a large batch? Explain equal-size
   loss weighting, one optimizer update, one clipping operation and no dropout.
7. What must be restored beyond weights? Give an example where resetting Adam
   moments or restarting warmup changes the resumed trajectory.
8. What does BF16 autocast change? Which state remains FP32? Distinguish the
   educational handwritten optimizer from the fused optimizer used in large runs.
9. How did you prevent exact document leakage? Which forms of near-duplicate
   leakage were not checked? Explain tokenizer fitting and test isolation.
10. Why time fixed tokens instead of independently sampled generations? Explain
    GPU synchronization, warmup, repeated samples and prefill/decode boundaries.
11. Why does equal-token training not imply equal compute for MHA and GQA? State
    the parameter difference and avoid claiming an equal-FLOP comparison.
12. What failed? The older diagnostic experiments did not beat strong simple
    controls; their negative results are preserved. Explain why engineering
    correctness does not establish a novel training method or RSI.

## Demonstration sequence

- Open `model.py` and explain one decoder block and one cached attention step.
- Run the small CPU tests for attention parity and disk recovery.
- Load published weights and generate from a prompt outside the fixed examples.
- Open one benchmark JSON and recompute a median throughput/speedup.
- Point to the fixed protocol, document hashes, test-token count and limitations.
