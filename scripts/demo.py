#!/usr/bin/env python3
"""Local interactive story-generation demo; uses the project's trained weights."""

from __future__ import annotations

import argparse
import os

import torch

from branchlab.cli import load_inference
from branchlab.training import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/portfolio/showcase-gqa")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    # Local startup checks must not travel through a system HTTP proxy.
    for name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(name, "")
        os.environ[name] = ",".join(
            filter(None, (existing, "localhost", "127.0.0.1", "::1"))
        )
    try:
        import gradio as gr
    except ImportError as error:
        raise SystemExit(
            'Install the optional demo: python -m pip install -e ".[demo]"'
        ) from error
    device = resolve_device(args.device)
    model, tokenizer = load_inference(args.model, device)

    def generate(prompt, tokens, temperature, top_k, seed, cache):
        encoded = tokenizer.encode(prompt)
        if not encoded:
            raise gr.Error("Enter a nonempty prompt.")
        if len(encoded) + int(tokens) > model.config.max_seq_len:
            raise gr.Error(
                f"Prompt + continuation must fit {model.config.max_seq_len} tokens."
            )
        generator_device = "cpu" if device.type == "mps" else device
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        ids = torch.tensor([encoded], dtype=torch.long, device=device)
        result = model.generate(
            ids,
            int(tokens),
            eos_id=tokenizer.eos_id,
            cache_mode=cache,
            temperature=float(temperature),
            top_k=int(top_k),
            generator=generator,
        )
        return tokenizer.decode(result[0].tolist())

    description = (
        f"From-scratch TinyStories subset model · {model.num_parameters():,} parameters · "
        f"{model.config.n_heads} query / {model.config.kv_heads} KV heads · {device}. "
        "A small story model; factual accuracy and instruction following are not evaluated."
    )
    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(
                value="Once upon a time, there was a little girl", label="Story opening"
            ),
            gr.Slider(1, 256, value=128, step=1, label="Maximum new tokens"),
            gr.Slider(0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)"),
            gr.Slider(1, 100, value=40, step=1, label="Top-k"),
            gr.Number(value=42, precision=0, label="Sampling seed"),
            gr.Radio(
                ["static", "dynamic"], value="static", label="Cache implementation"
            ),
        ],
        outputs=gr.Textbox(label="Generated story", lines=12),
        flagging_mode="never",
        title="BranchLab — A small language model, explained",
        description=description,
        examples=[
            ["Tom found a small red box in the garden.", 128, 0.8, 40, 42, "static"],
            ["The dog was afraid of the rain.", 128, 0.8, 40, 42, "static"],
        ],
    )
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
