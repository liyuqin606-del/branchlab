#!/usr/bin/env python3
"""Fixed-shape single-token decoding with an optional reusable CUDA graph.

This is an inference extension; the frozen training implementation is unchanged.
Unlike a sliced cache, buffers and attention shapes stay fixed. A device-side
position tensor controls RoPE, cache writes and the mask. Capture cost is reported
separately from steady-state requests; one decoder serves one request at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from branchlab.cli import load_inference
from branchlab.portfolio_data import file_sha256
from branchlab.training import json_write, synchronize


class FixedShapeDecoder:
    """Own a fixed-capacity cache; reset before each independent request."""

    @torch.inference_mode()
    def __init__(self, model, batch_size: int, capacity: int, capture: bool = True):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        if batch_size < 1 or not 1 <= capacity <= model.config.max_seq_len:
            raise ValueError("Invalid batch/capacity")
        if capture and self.device.type != "cuda":
            raise ValueError("CUDA graph capture requires CUDA")
        self.batch_size, self.capacity = batch_size, capacity
        self.input = torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)
        self.position = torch.zeros(1, device=self.device, dtype=torch.long)
        self.slots = torch.arange(capacity, device=self.device)
        shape = (
            batch_size,
            model.config.kv_heads,
            capacity,
            model.config.d_model // model.config.n_heads,
        )
        self.cache = [
            (
                torch.zeros(shape, device=self.device, dtype=self.dtype),
                torch.zeros(shape, device=self.device, dtype=self.dtype),
            )
            for _ in model.blocks
        ]
        self.cursor = 0
        self.graph = None
        started = time.perf_counter()
        if capture:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.position.zero_()
                    self._forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            synchronize(self.device)
            self.position.zero_()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self._forward()
            synchronize(self.device)
        self.setup_seconds = time.perf_counter() - started

    def _forward(self):
        """Graph-safe computation: no .item(), host branching on tensors or resizing."""
        hidden = self.model.embedding(self.input)
        visible = self.slots[None, :] <= self.position[:, None]
        for block, (keys, values) in zip(self.model.blocks, self.cache):
            attention = block.attn
            q_size = attention.n_heads * attention.head_dim
            kv_size = attention.n_kv_heads * attention.head_dim
            q, k, v = attention.qkv(block.attn_norm(hidden)).split(
                (q_size, kv_size, kv_size), dim=-1
            )
            q = q.reshape(
                self.batch_size, 1, attention.n_heads, attention.head_dim
            ).transpose(1, 2)
            k = k.reshape(
                self.batch_size, 1, attention.n_kv_heads, attention.head_dim
            ).transpose(1, 2)
            v = v.reshape(
                self.batch_size, 1, attention.n_kv_heads, attention.head_dim
            ).transpose(1, 2)
            angle = (
                self.position.float()[:, None]
                * attention.rope.inv_freq.float()[None, :]
            )
            cosine, sine = angle.cos().to(q.dtype), angle.sin().to(q.dtype)

            def rotate(x):
                even, odd = x[..., 0::2], x[..., 1::2]
                return torch.stack(
                    (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
                ).flatten(-2)

            q, k = rotate(q), rotate(k)
            keys.index_copy_(2, self.position, k)
            values.index_copy_(2, self.position, v)
            groups = attention.n_heads // attention.n_kv_heads
            expanded_k = keys.repeat_interleave(groups, dim=1) if groups > 1 else keys
            expanded_v = (
                values.repeat_interleave(groups, dim=1) if groups > 1 else values
            )
            attended = F.scaled_dot_product_attention(
                q, expanded_k, expanded_v, attn_mask=visible
            )
            attended = attended.transpose(1, 2).reshape(self.batch_size, 1, -1)
            hidden = hidden + attention.out_proj(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        logits = self.model.lm_head(self.model.final_norm(hidden))
        self.position.add_(1)
        return logits

    @torch.inference_mode()
    def reset(self, prompt):
        if (
            prompt.ndim != 2
            or prompt.shape[0] != self.batch_size
            or not 0 < prompt.shape[1] < self.capacity
        ):
            raise ValueError("Prompt shape must fit the fixed request capacity")
        if prompt.device != self.device:
            raise ValueError("Prompt/device mismatch")
        _, prefix = self.model(prompt, use_cache=True)
        for (keys, values), (prefix_k, prefix_v) in zip(self.cache, prefix):
            keys.zero_()
            values.zero_()
            keys[:, :, : prompt.shape[1]].copy_(prefix_k)
            values[:, :, : prompt.shape[1]].copy_(prefix_v)
        self.position.fill_(prompt.shape[1])
        self.cursor = prompt.shape[1]

    @torch.inference_mode()
    def step(self, token):
        if (
            token.shape != self.input.shape
            or token.device != self.device
            or token.dtype != torch.long
        ):
            raise ValueError(
                "Expected one long token per batch item on the model device"
            )
        if self.cursor >= self.capacity:
            raise ValueError("Fixed cache capacity exceeded")
        self.input.copy_(token)
        if self.graph is None:
            output = self._forward()
        else:
            self.graph.replay()
            output = self.output
        self.cursor += 1
        # Graph output is reused on the next replay: callers retaining it must clone.
        return output


@torch.inference_mode()
def benchmark_graph(model, prompt, decode_tokens=64, repeats=7):
    decoder = FixedShapeDecoder(model, prompt.shape[0], prompt.shape[1] + decode_tokens)
    device = prompt.device
    continuation = (
        prompt[:, -1:] + torch.arange(1, decode_tokens + 1, device=device)
    ) % model.config.vocab_size
    _, cache = model(prompt, use_cache=True)
    decoder.reset(prompt)
    expected, actual = [], []
    for index in range(decode_tokens):
        token = continuation[:, index : index + 1]
        logits, cache = model(token, cache, use_cache=True)
        expected.append(logits[:, -1].clone())
        actual.append(decoder.step(token)[:, -1].clone())
    reference, candidate = (
        torch.stack(expected, 1).float(),
        torch.stack(actual, 1).float(),
    )
    difference = float((reference - candidate).abs().max())
    kl = float(
        F.kl_div(candidate.log_softmax(-1), reference.softmax(-1), reduction="none")
        .sum(-1)
        .mean()
        .clamp_min(0)
    )
    tolerance = 2e-3 if next(model.parameters()).dtype == torch.bfloat16 else 1e-6
    if not math.isfinite(kl) or kl > tolerance:
        raise AssertionError(
            f"Graph parity failed: KL {kl}, max difference {difference}"
        )
    mismatch = float((reference.argmax(-1) != candidate.argmax(-1)).float().mean())
    del expected, actual, reference, candidate, cache
    samples = {"dynamic": [], "cuda_graph": []}
    prefills = {"dynamic": [], "cuda_graph": []}
    for repeat in range(repeats):
        for mode in (
            ("dynamic", "cuda_graph") if repeat % 2 == 0 else ("cuda_graph", "dynamic")
        ):
            synchronize(device)
            started = time.perf_counter()
            if mode == "dynamic":
                _, cache = model(prompt, use_cache=True)
            else:
                decoder.reset(prompt)
            synchronize(device)
            prefills[mode].append(time.perf_counter() - started)
            started = time.perf_counter()
            for index in range(decode_tokens):
                token = continuation[:, index : index + 1]
                if mode == "dynamic":
                    _, cache = model(token, cache, use_cache=True)
                else:
                    decoder.step(token)
            synchronize(device)
            samples[mode].append(time.perf_counter() - started)
            if mode == "dynamic":
                del cache
    count = prompt.shape[0] * decode_tokens
    times = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "decode_seconds_samples": samples,
        "prefill_seconds_samples": prefills,
        "decode_tokens_per_second": {
            name: count / value for name, value in times.items()
        },
        "speedup_vs_dynamic": times["dynamic"] / times["cuda_graph"],
        "capture_and_warmup_seconds": decoder.setup_seconds,
        "parity": {
            "logits_max_abs_difference": difference,
            "mean_distribution_kl": kl,
            "argmax_disagreement_fraction": mismatch,
            "kl_tolerance": tolerance,
        },
        "metadata": {
            "batch_size": prompt.shape[0],
            "prompt_tokens": prompt.shape[1],
            "decode_tokens": decode_tokens,
            "repeats": repeats,
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "dtype": str(next(model.parameters()).dtype),
            "scope": "Warm reusable graph; capture cost excluded from decode and separately reported; same forced tokens; alternating order",
            "script_sha256": file_sha256(__file__),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = load_inference(args.model, torch.device("cuda"))
    model.to(dtype=torch.bfloat16)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "grid": [
            {"batch_size": b, "prompt_tokens": p, "decode_tokens": 64}
            for b in (1, 4)
            for p in (64, 256)
        ],
        "repeats": 7,
        "model_sha256": file_sha256(Path(args.model) / "model.pt"),
        "scope": "Inference extension after weak short-context cache speedups; training protocol unchanged",
        "script_sha256": file_sha256(__file__),
    }
    json_write(out / "graph_protocol.json", protocol)
    encoded = tokenizer.encode("Once upon a time, there was a little girl who")
    for row in protocol["grid"]:
        length = row["prompt_tokens"]
        ids = (encoded * (length // len(encoded) + 1))[:length]
        prompt = torch.tensor([ids] * row["batch_size"], device="cuda")
        result = benchmark_graph(model, prompt)
        result["metadata"]["model_sha256"] = protocol["model_sha256"]
        path = out / f"graph-b{row['batch_size']}-p{length}.json"
        json_write(path, result)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "speedup": result["speedup_vs_dynamic"],
                    "parity": result["parity"],
                }
            ),
            flush=True,
        )
