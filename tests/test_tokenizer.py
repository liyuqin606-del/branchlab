import json

import pytest

from branchlab.tokenizer import ByteBPETokenizer


@pytest.mark.parametrize("text", ["", "a\x00b", "hello\t world\r\n", "中文训练轨迹 🙂 café e\u0301", "🧑🏽‍💻" * 4])
def test_roundtrip_including_unseen_unicode(text):
    tokenizer = ByteBPETokenizer.train(["hello hello world\n", "world hello"], vocab_size=280)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode(text, add_eos=True)) == text
    assert tokenizer.eos_id == 256


def test_deterministic_order_independent_training_and_serialization(tmp_path):
    texts = ["alpha beta beta", "你好 你好 alpha", "gamma\n", "alpha gamma"]
    first = ByteBPETokenizer.train(texts, vocab_size=290)
    second = ByteBPETokenizer.train(reversed(texts), vocab_size=290)
    assert first.merges == second.merges
    assert 257 < first.vocab_size <= 290
    path = tmp_path / "tokenizer.json"
    first.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.merges == first.merges
    assert loaded.encode("你好 gamma alpha") == first.encode("你好 gamma alpha")
    second_path = tmp_path / "second.json"
    loaded.save(second_path)
    assert path.read_bytes() == second_path.read_bytes()


def test_merge_tie_breaking_and_nonoverlapping_pairs():
    tokenizer = ByteBPETokenizer.train(["aaaa abab"], vocab_size=258, min_frequency=1)
    assert tokenizer.merges == [(97, 97)]
    assert tokenizer.encode("aaaa") == [257, 257]
    assert tokenizer.decode([257, 257]) == "aaaa"


def test_vocabulary_limit_and_insufficient_frequency():
    tokenizer = ByteBPETokenizer.train(["a b c"], vocab_size=512)
    assert tokenizer.vocab_size == 257
    assert tokenizer.encode("abc") == [97, 98, 99]
    assert ByteBPETokenizer.train(["abc abc"], vocab_size=257).vocab_size == 257
    with pytest.raises(ValueError, match=">= 257"):
        ByteBPETokenizer.train(["abc"], vocab_size=256)
    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([257])
    with pytest.raises(ValueError, match="not-yet-created"):
        ByteBPETokenizer([(257, 97)])


def test_load_rejects_incompatible_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "unrelated", "version": 1, "merges": []}))
    with pytest.raises(ValueError, match="unsupported"):
        ByteBPETokenizer.load(path)
