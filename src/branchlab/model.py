"""A small decoder-only Transformer implemented without attention shortcuts.

Keys in the cache are already rotary-position encoded.  Cache tensors have shape
``(batch, heads, past_tokens, head_dim)`` and remain attached to autograd; callers
doing inference should use ``torch.no_grad()`` (as :meth:`generate` does).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


KVCache = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    max_seq_len: int = 256
    ffn_mult: float = 8 / 3
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not math.isfinite(self.ffn_mult) or self.ffn_mult <= 0:
            raise ValueError("ffn_mult must be finite and positive")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Keep reductions stable under half-precision training.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq", 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim),
            persistent=False,
        )

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor, cache: KVCache | None = None,
                use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        past_len = 0 if cache is None else cache[0].shape[-2]
        q, k = self.rope(q, past_len), self.rope(k, past_len)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=-2), torch.cat((cache[1], v), dim=-2)
        # With a nonempty cache and several new tokens, is_causal=True would
        # require the correct offset.  Build that mask explicitly here.
        query_positions = past_len + torch.arange(seq_len, device=x.device)
        key_positions = torch.arange(k.shape[-2], device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        softmax_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
        weights = F.softmax(scores, dim=-1, dtype=softmax_dtype).to(q.dtype)
        output = (weights @ v).transpose(1, 2).contiguous().reshape(batch, seq_len, dim)
        return self.out_proj(output), (k, v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = max(1, int(config.d_model * config.ffn_mult))
        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: Tensor, cache: KVCache | None = None,
                use_cache: bool = False) -> tuple[Tensor, KVCache | None]:
        attention, new_cache = self.attn(self.attn_norm(x), cache, use_cache)
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), new_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(self.config) for _ in range(self.config.n_layers))
        self.final_norm = RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _validate_cache(self, input_ids: Tensor, cache: Sequence[KVCache] | None) -> int:
        if cache is None:
            return 0
        if len(cache) != self.config.n_layers:
            raise ValueError("cache must contain exactly one (key, value) pair per layer")
        lengths: set[int] = set()
        for layer in cache:
            if len(layer) != 2:
                raise ValueError("each layer cache must contain key and value tensors")
            k, v = layer
            if k.ndim != 4 or k.shape != v.shape:
                raise ValueError("cached keys and values must have matching (B,H,T,D) shapes")
            if (k.shape[0], k.shape[1], k.shape[3]) != (
                input_ids.shape[0], self.config.n_heads, self.config.d_model // self.config.n_heads,
            ):
                raise ValueError("cache shape does not match input batch or model dimensions")
            if k.device != input_ids.device or v.device != input_ids.device:
                raise ValueError("cache and input must be on the same device")
            lengths.add(k.shape[-2])
        if len(lengths) != 1:
            raise ValueError("every layer cache must have the same sequence length")
        return next(iter(lengths))

    def forward(self, input_ids: Tensor, cache: Sequence[KVCache] | None = None,
                use_cache: bool = False) -> tuple[Tensor, list[KVCache] | None]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty sequence)")
        past_len = self._validate_cache(input_ids, cache)
        if past_len + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input plus cached tokens exceeds max_seq_len")
        x = self.embedding(input_ids)
        new_cache: list[KVCache] | None = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            x, layer_cache = block(x, None if cache is None else cache[index], use_cache)
            if new_cache is not None:
                assert layer_cache is not None
                new_cache.append(layer_cache)
        return self.lm_head(self.final_norm(x)), new_cache

    def num_parameters(self) -> int:
        """Count unique parameters, so tied embeddings are counted once."""
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 32,
                 eos_id: int | None = None, use_cache: bool = True) -> Tensor:
        """Greedy batched generation, returning the prompt followed by new tokens."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty sequence)")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("requested generation exceeds max_seq_len")
        if eos_id is not None and not 0 <= eos_id < self.config.vocab_size:
            raise ValueError("eos_id is outside the vocabulary")
        was_training = self.training
        self.eval()
        try:
            result = input_ids.clone()
            cache = None
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for _ in range(max_new_tokens):
                current = result if cache is None else result[:, -1:]
                logits, cache = self(current, cache=cache, use_cache=use_cache)
                next_ids = logits[:, -1].argmax(dim=-1)
                if eos_id is not None:
                    next_ids = torch.where(finished, eos_id, next_ids)
                    finished |= next_ids == eos_id
                result = torch.cat((result, next_ids[:, None]), dim=1)
                if eos_id is not None and bool(finished.all()):
                    break
            return result
        finally:
            self.train(was_training)


# Both descriptive names are useful at the command line and in notebooks.
DecoderOnlyTransformer = Transformer
TransformerLM = Transformer
