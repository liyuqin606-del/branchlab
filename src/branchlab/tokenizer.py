"""Deterministic byte-level BPE with no tokenizer-library dependency.

Whitespace and non-whitespace runs form pre-tokenization boundaries.  Training
counts repeated pieces once with a frequency weight; it does not repeatedly
scan entire documents.  UTF-8 bytes ensure complete coverage of unseen text.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


_PIECES = re.compile(r"\s+|[^\s]+")


def _merge_piece(piece: tuple[int, ...], pair: tuple[int, int], token: int) -> tuple[int, ...]:
    result: list[int] = []
    index = 0
    while index < len(piece):
        if index + 1 < len(piece) and (piece[index], piece[index + 1]) == pair:
            result.append(token)
            index += 2
        else:
            result.append(piece[index])
            index += 1
    return tuple(result)


class ByteBPETokenizer:
    eos_id = 256

    def __init__(self, merges: Sequence[Sequence[int]] = ()) -> None:
        self.merges: list[tuple[int, int]] = []
        self._token_bytes = {index: bytes([index]) for index in range(256)}
        self._token_bytes[self.eos_id] = b""
        self._ranks: dict[tuple[int, int], int] = {}
        for raw_pair in merges:
            if len(raw_pair) != 2:
                raise ValueError("each BPE merge must contain exactly two token IDs")
            pair = tuple(raw_pair)
            token_id = 257 + len(self.merges)
            if any(not isinstance(item, int) or isinstance(item, bool) or item < 0
                   or item >= token_id or item == self.eos_id for item in pair):
                raise ValueError("merge refers to an invalid, special, or not-yet-created token")
            if pair in self._ranks:
                raise ValueError("duplicate BPE merge")
            self._ranks[pair] = len(self.merges)
            self.merges.append(pair)
            self._token_bytes[token_id] = self._token_bytes[pair[0]] + self._token_bytes[pair[1]]

    @property
    def vocab_size(self) -> int:
        return 257 + len(self.merges)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 512,
              min_frequency: int = 2) -> "ByteBPETokenizer":
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 257:
            raise ValueError("vocab_size must be an integer >= 257 (all bytes plus EOS)")
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool) or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")
        if isinstance(texts, str):
            texts = [texts]
        pieces: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("training texts must be strings")
            pieces.update(tuple(piece.encode("utf-8")) for piece in _PIECES.findall(text))
        merges: list[tuple[int, int]] = []
        while len(merges) + 257 < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for piece, frequency in pieces.items():
                for pair in zip(piece, piece[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            # Break ties on token IDs, independent of corpus order and hash seed.
            pair, frequency = min(counts.items(), key=lambda entry: (-entry[1], entry[0]))
            if frequency < min_frequency:
                break
            token_id = 257 + len(merges)
            next_pieces: Counter[tuple[int, ...]] = Counter()
            for piece, count in pieces.items():
                next_pieces[_merge_piece(piece, pair, token_id)] += count
            pieces = next_pieces
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoded: list[int] = []
        for raw_piece in _PIECES.findall(text):
            piece = tuple(raw_piece.encode("utf-8"))
            while len(piece) > 1:
                candidates = [(self._ranks[pair], pair) for pair in zip(piece, piece[1:])
                              if pair in self._ranks]
                if not candidates:
                    break
                rank, pair = min(candidates)
                piece = _merge_piece(piece, pair, rank + 257)
            encoded.extend(piece)
        if add_eos:
            encoded.append(self.eos_id)
        return encoded

    def decode(self, ids: Iterable[int]) -> str:
        raw: list[bytes] = []
        for token in ids:
            if token not in self._token_bytes:
                raise ValueError(f"token ID outside vocabulary: {token}")
            raw.append(self._token_bytes[token])
        # An arbitrary token prefix can end mid-codepoint; full encode/decode
        # round trips remain exact for valid Unicode strings.
        return b"".join(raw).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "branchlab-byte-bpe", "version": 1,
                   "pretokenizer": "whitespace-runs-v1", "merges": self.merges}
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload.get("format"), payload.get("version"), payload.get("pretokenizer")) != (
            "branchlab-byte-bpe", 1, "whitespace-runs-v1",
        ):
            raise ValueError("unsupported tokenizer format or version")
        return cls(payload["merges"])
