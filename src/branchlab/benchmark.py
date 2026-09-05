"""Synchronized, fixed-continuation KV-cache measurements.

The benchmark measures teacher-forced decoding: both paths consume exactly the
same tokens, irrespective of their argmax predictions.  Prefill is reported
separately and excluded from decode throughput.  These timings describe the
provided model, device, precision, prompt length, and batch size only.
"""

from __future__ import annotations

import math
import statistics
import time

import torch

from .training import synchronize


def benchmark_kv(model, prompt_ids, generated_tokens=32, repeats=5):
    """Compare cached and full-prefix decoding, returning JSON-safe measurements.

    ``generated_tokens`` is the number of forced continuation tokens per batch
    item.  Tokens per second counts *all* batch items.  CUDA peak allocated
    memory is available; CPU and MPS do not expose an equivalent allocator peak
    through this function, so their peaks are explicitly ``None``.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] == 0 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape (nonempty batch, nonempty sequence)")
    for name, value in (("generated_tokens", generated_tokens), ("repeats", repeats)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    parameter = next(model.parameters())
    device = parameter.device
    if prompt_ids.device != device:
        raise ValueError("prompt_ids must already be on the model's device")
    if prompt_ids.shape[1] + generated_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus continuation exceeds max_seq_len")

    was_training = model.training
    model.eval()
    prefill_samples, cached_samples, uncached_samples = [], [], []
    peaks = {"cached_bytes": None, "uncached_bytes": None}
    cuda_peak = device.type == "cuda"

    try:
        with torch.inference_mode():
            # No random draws and no sample-dependent token sequence changes.
            offsets = torch.arange(1, generated_tokens + 1, device=device)
            continuation = (prompt_ids[:, -1:] + offsets[None, :]) % model.config.vocab_size
            full_ids = torch.cat((prompt_ids, continuation), dim=1)
            prompt_length = prompt_ids.shape[1]

            def prefill():
                _, cache = model(prompt_ids, use_cache=True)
                return cache

            def decode_cached(cache, record=False):
                recorded = [] if record else None
                for index in range(generated_tokens):
                    logits, cache = model(continuation[:, index:index + 1], cache=cache, use_cache=True)
                    if recorded is not None:
                        recorded.append(logits[:, -1].clone())
                return torch.stack(recorded, dim=1) if record else None

            def decode_uncached(record=False):
                recorded = [] if record else None
                for index in range(generated_tokens):
                    logits, _ = model(full_ids[:, :prompt_length + index + 1])
                    if recorded is not None:
                        recorded.append(logits[:, -1].clone())
                return torch.stack(recorded, dim=1) if record else None

            # Correctness and warmup are outside all measured regions.
            reference = decode_uncached(record=True)
            cached = decode_cached(prefill(), record=True)
            max_diff = float((reference.float() - cached.float()).abs().max())
            if not math.isfinite(max_diff):
                raise FloatingPointError("Nonfinite logits in KV benchmark")
            del reference, cached
            synchronize(device)

            def measured(call):
                synchronize(device)
                started = time.perf_counter()
                result = call()
                synchronize(device)
                elapsed = time.perf_counter() - started
                return elapsed, result

            for repeat in range(repeats):
                elapsed, cache = measured(prefill)
                prefill_samples.append(elapsed)
                del cache

                def cached_trial():
                    # Transfer ownership into the decode loop so an old prefix
                    # cache is not artificially kept alive during every step.
                    cache_holder = [prefill()]
                    if cuda_peak:
                        torch.cuda.reset_peak_memory_stats(device)
                    elapsed, _ = measured(lambda: decode_cached(cache_holder.pop()))
                    cached_samples.append(elapsed)
                    if cuda_peak:
                        peak = torch.cuda.max_memory_allocated(device)
                        peaks["cached_bytes"] = max(peaks["cached_bytes"] or 0, peak)

                def uncached_trial():
                    if cuda_peak:
                        torch.cuda.reset_peak_memory_stats(device)
                    elapsed, _ = measured(decode_uncached)
                    uncached_samples.append(elapsed)
                    if cuda_peak:
                        peak = torch.cuda.max_memory_allocated(device)
                        peaks["uncached_bytes"] = max(peaks["uncached_bytes"] or 0, peak)

                # Alternate the order to reduce consistent thermal/order bias.
                if repeat % 2:
                    uncached_trial()
                    cached_trial()
                else:
                    cached_trial()
                    uncached_trial()

            token_count = prompt_ids.shape[0] * generated_tokens
            cached_time = statistics.median(cached_samples)
            uncached_time = statistics.median(uncached_samples)
            return {
                "prefill_seconds": statistics.median(prefill_samples),
                "cached_decode_tokens_per_second": token_count / cached_time,
                "uncached_decode_tokens_per_second": token_count / uncached_time,
                "decode_speedup": uncached_time / cached_time,
                "logits_max_diff": max_diff,
                "prefill_seconds_samples": prefill_samples,
                "cached_decode_seconds_samples": cached_samples,
                "uncached_decode_seconds_samples": uncached_samples,
                "peak_memory": {
                    "available": cuda_peak,
                    "metric": "torch.cuda.max_memory_allocated" if cuda_peak else None,
                    **peaks,
                    "scope": "Absolute live tensor allocation including model and inputs; no allocator-reserved memory",
                    "unavailable_reason": None if cuda_peak else "No comparable per-trial peak allocator counter exposed for CPU/MPS",
                },
                "metadata": {
                    "device": str(device), "dtype": str(parameter.dtype),
                    "batch_size": prompt_ids.shape[0], "prompt_tokens": prompt_length,
                    "decode_tokens_per_sequence": generated_tokens, "timed_tokens_per_trial": token_count,
                    "repeats": repeats, "warmup_runs_per_path": 1,
                    "aggregation": "median seconds, throughput = batch tokens / median seconds",
                    "prefill_in_decode_timing": False,
                    "continuation": "Fixed arithmetic token IDs shared by both paths; no sampling",
                    "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                    "attention_implementation": "manual causal attention",
                },
            }
    finally:
        model.train(was_training)
