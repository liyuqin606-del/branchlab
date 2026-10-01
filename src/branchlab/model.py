"""A small decoder-only Transformer implemented without attention shortcuts.

Keys in the cache are already rotary-position encoded.  Cache tensors have shape
``(batch, heads, past_tokens, head_dim)`` and remain attached to autograd; callers
doing inference should use ``torch.no_grad()`` (as :meth:`generate` does).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
    n_kv_heads: int | None = None
    attention_backend: str = "manual"

    @property
    def kv_heads(self) -> int:
        """MHA is the special case where every query has its own K/V head."""
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.kv_heads, int)
            or isinstance(self.kv_heads, bool)
            or self.kv_heads <= 0
        ):
            raise ValueError("n_kv_heads must be a positive integer")
        if self.n_heads % self.kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.attention_backend not in ("manual", "sdpa"):
            raise ValueError("attention_backend must be manual or sdpa")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not math.isfinite(self.ffn_mult) or self.ffn_mult <= 0:
            raise ValueError("ffn_mult must be finite and positive")


@dataclass
class StaticKVCache:
    """Inference-only storage: append into a fixed buffer rather than reallocating.

    Stored tensors have KV heads, not query heads. The filled prefix is the only
    region attention may read. Allocate a fresh cache for each request.
    """

    key: Tensor
    value: Tensor
    length: int = 0

    @property
    def capacity(self) -> int:
        return self.key.shape[-2]

    @property
    def allocated_bytes(self) -> int:
        return (self.key.numel() + self.value.numel()) * self.key.element_size()

    def append(self, key: Tensor, value: Tensor) -> KVCache:
        if torch.is_grad_enabled():
            raise ValueError(
                "StaticKVCache is inference-only; use inference_mode/no_grad"
            )
        end = self.length + key.shape[-2]
        if end > self.capacity:
            raise ValueError("StaticKVCache capacity exceeded")
        if key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            raise ValueError("StaticKVCache dtype must match projected keys/values")
        self.key[:, :, self.length : end].copy_(key)
        self.value[:, :, self.length : end].copy_(value)
        self.length = end
        return self.key[:, :, :end], self.value[:, :, :end]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Keep reductions stable under half-precision training.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim),
            persistent=False,
        )

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        positions = torch.arange(
            offset, offset + x.shape[-2], device=x.device, dtype=torch.float32
        )
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_kv_heads = config.kv_heads
        self.backend = config.attention_backend
        packed_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(config.d_model, packed_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        batch, seq_len, dim = x.shape
        # Packed Q/K/V keeps old MHA checkpoints compatible. GQA reduces K/V
        # projection width and persistent cache storage, not query head count.
        q_width, kv_width = (
            self.n_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        )
        q, k, v = self.qkv(x).split((q_width, kv_width, kv_width), dim=-1)
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        past_len = (
            0
            if cache is None
            else (
                cache.length if isinstance(cache, StaticKVCache) else cache[0].shape[-2]
            )
        )
        q, k = self.rope(q, past_len), self.rope(k, past_len)
        if isinstance(cache, StaticKVCache):
            k, v = cache.append(k, v)
            next_cache = cache
        else:
            if cache is not None:
                k, v = (
                    torch.cat((cache[0], k), dim=-2),
                    torch.cat((cache[1], v), dim=-2),
                )
            next_cache = (k, v)
        # Explicit expansion is portable to CPU/MPS and easy to inspect. We do
        # not claim a custom fused GQA kernel: SDPA dispatch belongs to PyTorch.
        groups = self.n_heads // self.n_kv_heads
        attention_k = k.repeat_interleave(groups, dim=1) if groups > 1 else k
        attention_v = v.repeat_interleave(groups, dim=1) if groups > 1 else v
        if self.backend == "sdpa" and past_len == 0:
            output = F.scaled_dot_product_attention(
                q, attention_k, attention_v, is_causal=True
            )
        else:
            # Cached multi-token chunks need an offset causal mask. For a single
            # new token all stored keys are visible, so no mask is necessary.
            allowed = None
            if seq_len > 1:
                queries = past_len + torch.arange(seq_len, device=x.device)
                keys = torch.arange(k.shape[-2], device=x.device)
                allowed = keys[None, :] <= queries[:, None]
            if self.backend == "sdpa":
                output = F.scaled_dot_product_attention(
                    q, attention_k, attention_v, attn_mask=allowed
                )
            else:
                scores = (q @ attention_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if allowed is not None:
                    scores = scores.masked_fill(~allowed, float("-inf"))
                softmax_dtype = (
                    torch.float32
                    if scores.dtype in (torch.float16, torch.bfloat16)
                    else scores.dtype
                )
                weights = F.softmax(scores, dim=-1, dtype=softmax_dtype).to(q.dtype)
                output = weights @ attention_v
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, dim)
        return self.out_proj(output), next_cache if use_cache else None


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

    def forward(
        self,
        x: Tensor,
        cache: KVCache | StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        attention, new_cache = self.attn(self.attn_norm(x), cache, use_cache)
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), new_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        )
        self.final_norm = RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.apply(self._init_weights)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _validate_cache(
        self, input_ids: Tensor, cache: Sequence[KVCache | StaticKVCache] | None
    ) -> int:
        if cache is None:
            return 0
        if len(cache) != self.config.n_layers:
            raise ValueError(
                "cache must contain exactly one (key, value) pair per layer"
            )
        lengths: set[int] = set()
        for layer in cache:
            if isinstance(layer, StaticKVCache):
                k, v = layer.key, layer.value
                if (
                    not isinstance(layer.length, int)
                    or not 0 <= layer.length <= layer.capacity
                ):
                    raise ValueError("Invalid static cache length")
                if layer.length + input_ids.shape[1] > layer.capacity:
                    raise ValueError("StaticKVCache capacity exceeded")
                if torch.is_grad_enabled():
                    raise ValueError(
                        "StaticKVCache is inference-only; use inference_mode/no_grad"
                    )
                length = layer.length
            elif len(layer) != 2:
                raise ValueError("each layer cache must contain key and value tensors")
            else:
                k, v = layer
                length = k.shape[-2]
            if k.ndim != 4 or k.shape != v.shape:
                raise ValueError(
                    "cached keys and values must have matching (B,H,T,D) shapes"
                )
            if (k.shape[0], k.shape[1], k.shape[3]) != (
                input_ids.shape[0],
                self.config.kv_heads,
                self.config.d_model // self.config.n_heads,
            ):
                raise ValueError(
                    "cache shape does not match input batch or model dimensions"
                )
            if k.device != input_ids.device or v.device != input_ids.device:
                raise ValueError("cache and input must be on the same device")
            lengths.add(length)
        if len(lengths) != 1:
            raise ValueError("every layer cache must have the same sequence length")
        return next(iter(lengths))

    def forward(
        self,
        input_ids: Tensor,
        cache: Sequence[KVCache | StaticKVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[KVCache | StaticKVCache] | None]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty sequence)")
        past_len = self._validate_cache(input_ids, cache)
        if past_len + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input plus cached tokens exceeds max_seq_len")
        x = self.embedding(input_ids)
        new_cache: list[KVCache | StaticKVCache] | None = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            x, layer_cache = block(
                x, None if cache is None else cache[index], use_cache
            )
            if new_cache is not None:
                assert layer_cache is not None
                new_cache.append(layer_cache)
        return self.lm_head(self.final_norm(x)), new_cache

    def allocate_cache(
        self,
        batch_size: int,
        capacity: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[StaticKVCache]:
        """Reserve buffers for one inference request; unused capacity is counted."""
        capacity = self.config.max_seq_len if capacity is None else capacity
        if batch_size < 1 or capacity < 1 or capacity > self.config.max_seq_len:
            raise ValueError("Invalid cache batch_size/capacity")
        parameter = next(self.parameters())
        shape = (
            batch_size,
            self.config.kv_heads,
            capacity,
            self.config.d_model // self.config.n_heads,
        )
        return [
            StaticKVCache(
                torch.empty(
                    shape, device=parameter.device, dtype=dtype or parameter.dtype
                ),
                torch.empty(
                    shape, device=parameter.device, dtype=dtype or parameter.dtype
                ),
            )
            for _ in self.blocks
        ]

    def num_parameters(self) -> int:
        """Count unique parameters, so tied embeddings are counted once."""
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        eos_id: int | None = None,
        use_cache: bool = True,
        cache_mode: str = "dynamic",
        temperature: float = 0.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Greedy batched generation, returning the prompt followed by new tokens."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty sequence)")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("requested generation exceeds max_seq_len")
        if eos_id is not None and not 0 <= eos_id < self.config.vocab_size:
            raise ValueError("eos_id is outside the vocabulary")
        if cache_mode not in ("dynamic", "static"):
            raise ValueError("cache_mode must be dynamic or static")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        was_training = self.training
        self.eval()
        try:
            result = input_ids.clone()
            cache = (
                self.allocate_cache(
                    input_ids.shape[0], input_ids.shape[1] + max_new_tokens
                )
                if use_cache and cache_mode == "static" and max_new_tokens
                else None
            )
            finished = torch.zeros(
                input_ids.shape[0], dtype=torch.bool, device=input_ids.device
            )
            for index in range(max_new_tokens):
                current = result if index == 0 or not use_cache else result[:, -1:]
                logits, cache = self(current, cache=cache, use_cache=use_cache)
                scores = logits[:, -1].float()
                if temperature == 0:
                    next_ids = scores.argmax(dim=-1)
                else:
                    scores = scores / temperature
                    if top_k is not None:
                        threshold = scores.topk(
                            min(top_k, scores.shape[-1]), dim=-1
                        ).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, float("-inf"))
                    probabilities = scores.softmax(dim=-1)
                    if (
                        generator is not None
                        and probabilities.device != generator.device
                    ):
                        probabilities = probabilities.to(generator.device)
                    next_ids = (
                        torch.multinomial(probabilities, 1, generator=generator)
                        .squeeze(-1)
                        .to(scores.device)
                    )
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
