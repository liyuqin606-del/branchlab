"""Command-line entry point for the complete BranchLab workflow."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import prepare_data, read_texts
from .model import ModelConfig, TransformerLM
from .tokenizer import ByteBPETokenizer
from .training import pack_texts, train_baseline, resolve_device, json_write


def prepare(args):
    manifest = prepare_data(args.data_dir, args.train_docs, args.eval_docs)
    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteBPETokenizer.train(read_texts(Path(args.data_dir)/"train.jsonl"), vocab_size=args.vocab_size)
    tokenizer.save(out/"tokenizer.json")
    counts = {}
    for split in ("train", "dev", "test"):
        tokens = pack_texts(read_texts(Path(args.data_dir)/f"{split}.jsonl"), tokenizer)
        np.save(out/f"{split}_tokens.npy",tokens)
        counts[split] = len(tokens)
    json_write(out/"preparation.json", {"source_revision":manifest["revision"], "tokens":counts,
               "tokenizer_vocab_size":tokenizer.vocab_size,"tokenizer_training_split":"train"})
    print(json.dumps({"event":"prepared","tokens":counts}))


def train(args):
    cfg = json.loads(Path(args.config).read_text())
    tokenizer = ByteBPETokenizer.load(Path(args.artifacts)/"tokenizer.json")
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **cfg["model"])
    tokens = {s:np.load(Path(args.artifacts)/f"{s}_tokens.npy") for s in ("train","dev")}
    start = time.perf_counter()
    model,optimizer,stream,state,history = train_baseline(model_config,tokens["train"],tokens["dev"],
        device=args.device,seed=cfg["seed"],steps=cfg["steps"],batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],lr=cfg["lr"],output_dir=args.output,eval_interval=cfg.get("eval_interval",50))
    out=Path(args.output)
    torch.save({"config":asdict(model_config),"model":{k:v.cpu() for k,v in model.state_dict().items()}},out/"model.pt")
    tokenizer.save(out/"tokenizer.json")
    json_write(out/"run.json", {"config":cfg,"parameters":model.num_parameters(),"device":str(next(model.parameters()).device),
          "torch_version":torch.__version__,"python":platform.python_version(),"platform":platform.platform(),
          "initial_dev_loss":history[0]["dev_loss"],"final_dev_loss":history[-1]["dev_loss"],
          "elapsed_seconds":time.perf_counter()-start,"trained_tokens":cfg["steps"]*cfg["batch_size"]*cfg["seq_len"],
          "scope":"From-scratch small-sample engineering run, not Marin replication or an RSI result"})


def load_inference(path,device):
    artifact=torch.load(Path(path)/"model.pt",map_location="cpu",weights_only=True)
    model=TransformerLM(ModelConfig(**artifact["config"])).to(device)
    model.load_state_dict(artifact["model"])
    model.eval()
    return model,ByteBPETokenizer.load(Path(path)/"tokenizer.json")


def benchmark(args):
    """Measure a saved model on a declared prompt shape with fixed continuation."""
    import hashlib
    from .benchmark import benchmark_kv

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.prompt_tokens is not None and args.prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    device = resolve_device(args.device)
    model, tokenizer = load_inference(args.model, device)
    encoded = tokenizer.encode(args.prompt)
    if not encoded:
        raise ValueError("Benchmark prompt must encode to at least one token")
    if args.prompt_tokens is not None:
        encoded = (encoded * ((args.prompt_tokens + len(encoded) - 1) // len(encoded)))[:args.prompt_tokens]
    prompt = torch.tensor([encoded] * args.batch_size, dtype=torch.long, device=device)
    result = benchmark_kv(model, prompt, generated_tokens=args.tokens, repeats=args.repeats)
    result["metadata"].update({
        "model_artifact": str(Path(args.model) / "model.pt"),
        "model_sha256": hashlib.sha256((Path(args.model) / "model.pt").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((Path(args.model) / "tokenizer.json").read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.cpu().numpy().tobytes()).hexdigest(),
        "parameters": model.num_parameters(),
    })
    json_write(args.output, result)
    print(json.dumps({"event": "benchmark", "output": args.output,
                      **{key: result[key] for key in ("prefill_seconds", "cached_decode_tokens_per_second",
                          "uncached_decode_tokens_per_second", "decode_speedup", "logits_max_diff")}}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",type=int,default=4)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--data-dir",default="data")
    p.add_argument("--artifacts",default="artifacts")
    p.add_argument("--train-docs",type=int,default=6000)
    p.add_argument("--eval-docs",type=int,default=600)
    p.add_argument("--vocab-size",type=int,default=512)
    p.set_defaults(func=prepare)
    p=sub.add_parser("train")
    p.add_argument("--config",default="configs/showcase.json")
    p.add_argument("--artifacts",default="artifacts")
    p.add_argument("--output",default="artifacts/showcase")
    p.add_argument("--device",default="auto")
    p.set_defaults(func=train)
    p=sub.add_parser("pilot")
    p.add_argument("--config",default="configs/pilot.json")
    p.add_argument("--artifacts",default="artifacts")
    p.add_argument("--output",default="artifacts/pilot")
    p.add_argument("--device",default="cpu")
    p.set_defaults(func=lambda a: __import__("branchlab.experiments",fromlist=["run_pilot"]).run_pilot(a))
    p=sub.add_parser("report")
    p.add_argument("--pilot",default="artifacts/pilot")
    p.add_argument("--showcase",default="artifacts/showcase")
    p.add_argument("--output",default="reports/release")
    p.set_defaults(func=lambda a: __import__("branchlab.reporting",fromlist=["build_report"]).build_report(a))
    p=sub.add_parser("generate")
    p.add_argument("--model",default="artifacts/showcase")
    p.add_argument("--prompt",default="Once upon a time, there was a little")
    p.add_argument("--tokens",type=int,default=64)
    p.add_argument("--device",default="auto")
    p.set_defaults(func=lambda a: generate(a))
    p=sub.add_parser("benchmark",help="Compare KV-cached and full-prefix decoding on identical tokens")
    p.add_argument("--model",default="artifacts/showcase")
    p.add_argument("--prompt",default="Once upon a time, there was a little")
    p.add_argument("--prompt-tokens",type=int,default=64,help="Repeat/truncate the encoded prompt to this token length")
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--tokens",type=int,default=32,help="Forced continuation tokens per batch item")
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--device",default="auto")
    p.add_argument("--output",default="reports/release/kv_benchmark.json")
    p.set_defaults(func=benchmark)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    args.func(args)


def generate(args):
    device=resolve_device(args.device)
    model,tokenizer=load_inference(args.model,device)
    prompt=torch.tensor([tokenizer.encode(args.prompt)],device=device)
    output=model.generate(prompt,max_new_tokens=args.tokens)
    print(tokenizer.decode(output[0].tolist()))


if __name__=="__main__":
    main()
