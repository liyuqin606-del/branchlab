"""End-to-end recovery, data isolation, accumulation and evidence checks."""

import json

import numpy as np
import pytest
import torch

from branchlab.fit import FitConfig, evaluate_tokens, fit, learning_rate, train_update
from branchlab.model import ModelConfig, Transformer
from branchlab.portfolio_data import file_sha256, select_partitions
from branchlab.systems import benchmark_paths
from branchlab.tokenizer import ByteBPETokenizer
from branchlab.training import TokenStream


def prepared_fixture(path):
    path.mkdir()
    ByteBPETokenizer().save(path / "tokenizer.json")
    manifest = {"splits": {}, "tokenizer_sha256": file_sha256(path / "tokenizer.json")}
    for i, name in enumerate(("train", "dev", "test")):
        np.save(path / f"{name}_tokens.npy", (np.arange(1025) + i) % 257)
        manifest["splits"][name] = {
            "tokens_sha256": file_sha256(path / f"{name}_tokens.npy")
        }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def tiny_config(**kwargs):
    config = dict(
        model={
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "max_seq_len": 32,
        },
        updates=6,
        micro_batch=2,
        accumulation=2,
        seq_len=8,
        warmup_updates=1,
        eval_interval=2,
        checkpoint_interval=2,
        eval_tokens=31,
        precision="fp32",
    )
    return FitConfig(**(config | kwargs))


def test_official_source_overlap_removed_and_order_independent():
    train = [f"story {i}" for i in range(30)] + [" story   33 "]
    val = [f"story {i}" for i in range(30, 50)]
    splits, audit = select_partitions(train, val, 10, 4)
    reversed_splits, _ = select_partitions(reversed(train), reversed(val), 10, 4)
    assert splits == reversed_splits
    assert audit["train_rows_removed_for_validation_overlap"] == 1
    identities = [r["id"] for rows in splits.values() for r in rows]
    assert len(identities) == len(set(identities)) == 18


def test_accumulation_matches_equivalent_large_batch_update():
    torch.manual_seed(2)
    small = Transformer(ModelConfig(vocab_size=257, d_model=16, n_heads=2, n_layers=1))
    large = Transformer(small.config)
    large.load_state_dict(small.state_dict())
    a, b = tiny_config(), tiny_config(micro_batch=4, accumulation=1)
    tokens = np.arange(500) % 257
    train_update(
        small,
        torch.optim.AdamW(small.parameters(), lr=0.001),
        TokenStream(tokens, 2, 8),
        a,
        torch.device("cpu"),
    )
    train_update(
        large,
        torch.optim.AdamW(large.parameters(), lr=0.001),
        TokenStream(tokens, 4, 8),
        b,
        torch.device("cpu"),
    )
    for p, q in zip(small.parameters(), large.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit_disk_resume_is_identical_and_completed_outputs_are_protected(
    tmp_path, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = prepared_fixture(tmp_path / "data")
    config = tiny_config(precision="bf16" if device == "cuda" else "fp32")
    direct = fit(config, data, tmp_path / "direct", device)
    partial = fit(config, data, tmp_path / "resumed", device, stop_after=3)
    assert partial["status"] == "interrupted"
    assert not (tmp_path / "resumed" / "run.json").exists()
    resumed = fit(
        config, data, tmp_path / "resumed", device, resume=partial["checkpoint"]
    )
    a = torch.load(tmp_path / "direct" / "model.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "model.pt", weights_only=True)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert direct["test"] == resumed["test"]
    assert direct["trained_tokens"] == config.updates * config.tokens_per_update
    assert direct["test"]["target_tokens"] == 31
    with pytest.raises(FileExistsError):
        fit(config, data, tmp_path / "direct", device)


def test_resume_rejects_changed_data_or_config(tmp_path):
    data = prepared_fixture(tmp_path / "data")
    config = tiny_config()
    partial = fit(config, data, tmp_path / "run", "cpu", stop_after=2)
    with pytest.raises(ValueError, match="mismatch"):
        fit(
            tiny_config(lr=0.002),
            data,
            tmp_path / "run",
            "cpu",
            resume=partial["checkpoint"],
        )
    np.save(data / "test_tokens.npy", np.arange(1025) % 256)
    with pytest.raises(ValueError, match="manifest"):
        fit(config, data, tmp_path / "run", "cpu", resume=partial["checkpoint"])


def test_systems_benchmark_reports_actual_static_bytes_and_parity():
    model = Transformer(
        ModelConfig(
            vocab_size=43,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            max_seq_len=32,
        )
    )
    result = benchmark_paths(model, torch.tensor([[1, 3, 7], [2, 4, 8]]), 4, 2)
    expected = 2 * 2 * 1 * 7 * 8 * 4
    assert result["paths"]["static"]["cache_allocated_bytes"] == expected
    assert result["paths"]["dynamic"]["cache_allocated_bytes"] == expected
    for mode in ("dynamic", "static"):
        assert result["paths"][mode]["parity"]["mean_distribution_kl"] < 1e-6
        assert len(result["paths"][mode]["decode_seconds_samples"]) == 2
    json.dumps(result, allow_nan=False)
    assert model.training


def test_cosine_endpoints_and_nonwrapping_evaluation():
    config = tiny_config()
    assert learning_rate(config, 0) == config.lr
    assert learning_rate(config, config.updates - 1) == pytest.approx(
        config.lr * config.min_lr_ratio
    )
    model = Transformer(ModelConfig(vocab_size=257, d_model=16, n_heads=2, n_layers=1))
    with pytest.raises(ValueError, match="never wrap"):
        evaluate_tokens(model, np.arange(31), config, torch.device("cpu"))
