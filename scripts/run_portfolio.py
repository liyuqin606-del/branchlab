#!/usr/bin/env python3
"""Execute the frozen MHA/GQA subset study and same-weight inference benchmarks.

Run locally or on Colab. Every completed run is immutable. A partial run is
resumed only from its full checkpoint. Source/config/data identity is checked.
A two-hour wall limit bounds this job; subsequent restart resumes completed work.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from branchlab.cli import load_inference
from branchlab.fit import FitConfig, fit, train_update
from branchlab.model import ModelConfig, Transformer
from branchlab.portfolio_data import file_sha256
from branchlab.systems import benchmark_paths
from branchlab.training import TokenStream, json_write, seed_all, synchronize


def preflight(output, device):
    """Use synthetic tokens: preflight does not inspect held-out text outcomes."""
    config = FitConfig(
        model={
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "n_kv_heads": 2,
            "max_seq_len": 512,
            "attention_backend": "sdpa",
        }
    )
    seed_all(7)
    model = Transformer(ModelConfig(vocab_size=2048, **config.model)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.95), fused=device.type == "cuda"
    )
    stream = TokenStream(
        torch.arange(100000) % 2048, config.micro_batch, config.seq_len
    )
    for _ in range(3):
        train_update(model, optimizer, stream, config, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(10):
        synchronize(device)
        started = time.perf_counter()
        train_update(model, optimizer, stream, config, device)
        synchronize(device)
        times.append(time.perf_counter() - started)
    result = {
        "seconds_per_update_samples": times,
        "mean_seconds_per_update": sum(times) / len(times),
        "tokens_per_update": config.tokens_per_update,
        "synthetic_tokens": True,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "planned_updates": 7200,
        "estimated_training_only_seconds": sum(times) / len(times) * 7200,
        "estimate_excludes": "Data preparation, evaluation, benchmarking, checkpoints and upload/download",
    }
    json_write(output, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(prepared, output, config_dir, device, max_seconds):
    started = time.monotonic()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    plans = []
    for seed in (101, 202, 303):
        for variant in ("mha", "gqa"):
            config = FitConfig(
                **json.loads((Path(config_dir) / f"{variant}.json").read_text())
            )
            plans.append((f"{variant}-s{seed}", replace(config, seed=seed)))
    showcase = FitConfig(**json.loads((Path(config_dir) / "gqa.json").read_text()))
    plans.append(
        (
            "showcase-gqa",
            replace(
                showcase, seed=42, updates=2400, warmup_updates=100, eval_interval=200
            ),
        )
    )
    protocol = {
        "purpose": "Engineering quality/cache tradeoff; no novel method or statistical superiority claim",
        "training_units": "Three paired training seeds; shared text/tokenizer and equal tokens per run",
        "comparison": "MHA 8 KV heads vs GQA 2 KV heads; same width/depth, different parameter count and FLOPs",
        "primary_quality_metric": "Final test-token mean cross-entropy over 65,536 nonrepeated target tokens",
        "test_selection": "No early stopping; final declared update only; no tuning on test scores",
        "showcase": "GQA chosen before outcomes, independent seed42, 2400 updates; not a fair-budget comparison",
        "benchmark": "Same saved weights and forced tokens across uncached/dynamic/static; BF16 CUDA; 7 repeats",
        "benchmark_grid": [
            {"batch_size": b, "prompt_tokens": p, "decode_tokens": 64}
            for b in (1, 4)
            for p in (64, 256)
        ],
        "limits": "One GPU, one training subset, three seeds, no near-duplicate audit, no significance claim",
        "wall_limit_seconds": max_seconds,
        "runs": {name: asdict(cfg) for name, cfg in plans},
        "data_manifest_sha256": file_sha256(Path(prepared) / "manifest.json"),
        "source_hashes": {
            str(path): file_sha256(path)
            for path in sorted(Path("src/branchlab").glob("*.py"))
        },
        "runner_sha256": file_sha256(__file__),
    }
    freeze = out / "protocol.json"
    if freeze.exists() and json.loads(freeze.read_text()) != protocol:
        raise ValueError("Existing experiment protocol does not match this job")
    json_write(freeze, protocol)
    progress = []
    for name, cfg in plans:
        if time.monotonic() - started > max_seconds:
            raise TimeoutError(
                "Portfolio wall-time budget exhausted; completed runs are preserved"
            )
        destination = out / name
        if not (destination / "run.json").exists():
            checkpoint = destination / "checkpoint.pt"
            result = fit(
                cfg,
                prepared,
                destination,
                device,
                resume=checkpoint if checkpoint.exists() else None,
            )
            assert result["status"] == "completed"
        # Main systems grid uses the first *prespecified* pair and showcase.
        if name in ("mha-s101", "gqa-s101", "showcase-gqa"):
            model, tokenizer = load_inference(destination, device)
            model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
            encoded = tokenizer.encode("Once upon a time, there was a little girl who")
            for batch in (1, 4):
                for prompt_length in (64, 256):
                    path = destination / f"benchmark-b{batch}-p{prompt_length}.json"
                    if path.exists():
                        continue
                    prompt = (encoded * (prompt_length // len(encoded) + 1))[
                        :prompt_length
                    ]
                    tensor = torch.tensor([prompt] * batch, device=device)
                    benchmark = benchmark_paths(
                        model, tensor, decode_tokens=64, repeats=7
                    )
                    benchmark["metadata"]["model_sha256"] = file_sha256(
                        destination / "model.pt"
                    )
                    json_write(path, benchmark)
            # Generation examples are a fixed prompt/seed suite, without cherry picking.
            model.float()
            examples = []
            for prompt in (
                "Once upon a time, there was a little girl",
                "Tom found a small red box in the garden.",
                "The dog was afraid of the rain.",
            ):
                ids = torch.tensor([tokenizer.encode(prompt)], device=device)
                generated = model.generate(
                    ids,
                    128,
                    eos_id=tokenizer.eos_id,
                    cache_mode="static",
                    temperature=0.8,
                    top_k=40,
                    generator=torch.Generator(device=device).manual_seed(42),
                )
                examples.append(
                    {
                        "prompt": prompt,
                        "temperature": 0.8,
                        "top_k": 40,
                        "sampling_seed": 42,
                        "text": tokenizer.decode(generated[0].tolist()),
                    }
                )
            json_write(destination / "examples.json", examples)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        progress.append(name)
        json_write(
            out / "progress.json",
            {
                "completed": progress,
                "total_runs": len(plans),
                "wall_seconds": time.monotonic() - started,
            },
        )
        print(
            json.dumps(
                {
                    "event": "run_complete",
                    "name": name,
                    "completed": len(progress),
                    "total": len(plans),
                }
            ),
            flush=True,
        )
    json_write(
        out / "completion.json",
        {
            "status": "completed",
            "runs": progress,
            "wall_seconds": time.monotonic() - started,
            "protocol_sha256": file_sha256(freeze),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default="/content/branchlab-data")
    parser.add_argument("--output", default="/content/branchlab-results")
    parser.add_argument("--configs", default="configs/portfolio")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    target = torch.device(args.device)
    if args.preflight:
        preflight(Path(args.output) / "preflight.json", target)
    else:
        run(args.prepared, args.output, args.configs, target, args.max_seconds)
