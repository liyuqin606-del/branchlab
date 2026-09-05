"""Synthetic implementation fixtures; these are not v2 experimental evidence."""
import copy
import json

import pytest

from branchlab.value_aware import ValueAwareDiagnosticLearner


P = "lr_half:2:loss_delta"
Q = "momentum_zero:4:grad_alignment"


def episodes(*, useful=False):
    rows = []
    for split, seeds in (("discovery", [0, 1, 2]), ("development", [3, 4]), ("test", [5, 6])):
        for seed in seeds:
            for i in range(8):
                bit = float(i % 2 * 2 - 1)
                # No free information about the action bit; probe discloses it.
                # A useful probe's correct action beats the free mean, whereas
                # a useless probe's cost-aligned target makes diagnosis harmful.
                rows.append({"id": f"{split}-{seed}-{i}", "seed": seed, "split": split,
                             "log_features": [1.0], "probe_features": {P: [bit], Q: [0.0]},
                             "probe_costs": {P: 16, Q: 28},
                             "budget_losses": {"0": [2 + bit, 2 - bit],
                                               "16": [2.1 + bit, 2.1 - bit] if useful else [4 + bit, 4 - bit],
                                               "28": [5 + bit, 5 - bit]}})
    return rows


class Forbidden(dict):
    def __getitem__(self, key):
        raise AssertionError("Skipped/forbidden data was read")


def test_net_objective_rejects_harmful_probe_and_skip_does_not_read():
    rows = episodes()
    model = ValueAwareDiagnosticLearner(search_budget_cells=200).fit(rows)
    assert model.selected_probe is None
    fixed = ValueAwareDiagnosticLearner(search_budget_cells=200).fit_fixed(rows, P)
    for row in rows:
        if row["split"] == "test":
            row["probe_features"] = Forbidden(row["probe_features"])
            row["probe_costs"] = Forbidden(row["probe_costs"])
    result = fixed.predict(rows)
    assert result["evaluation_revealed_cells"] == 0
    assert all(p is None for p in result["probe_ids"])
    assert result["evaluation_probe_cost"] == 0
    assert fixed.evaluate(rows)["mean_dev_loss"] == 2.0


def test_useful_probe_promotes_and_state_roundtrip_is_frozen():
    rows = episodes(useful=True)
    model = ValueAwareDiagnosticLearner(strategy="enumeration", search_budget_cells=200).fit(rows)
    assert model.selected_probe == P
    result = model.evaluate(rows)
    assert result["mean_dev_loss"] == pytest.approx(1.1)
    assert result["evaluation_revealed_cells"] == 16
    state = model.state_dict()
    loaded = ValueAwareDiagnosticLearner.from_state_dict(json.loads(json.dumps(state)))
    assert loaded.evaluate(rows) == result
    assert model.state_dict() == state


def test_test_poison_cannot_change_selection_or_predictions():
    original = episodes(useful=True)
    poisoned = copy.deepcopy(original)
    for row in poisoned:
        if row["split"] == "test":
            row["budget_losses"] = "not labels"
            row["log_features"] = [float("nan")]
            row["probe_features"] = Forbidden(row["probe_features"])
            row["probe_costs"] = Forbidden(row["probe_costs"])
    for strategy in ("counterexample", "random", "enumeration"):
        a = ValueAwareDiagnosticLearner(strategy=strategy, search_budget_cells=100).fit(original)
        b = ValueAwareDiagnosticLearner(strategy=strategy, search_budget_cells=100).fit(poisoned)
        assert a.state_dict() == b.state_dict()
    for row in original:
        if row["split"] == "test":
            row["budget_losses"] = Forbidden(row["budget_losses"])
    a.predict(original)  # prediction may read paid features, never future labels


def test_crossfit_excludes_whole_seeds_and_requires_three():
    model = ValueAwareDiagnosticLearner(search_budget_cells=200).fit(episodes(useful=True))
    folds = model.search_report["crossfit_folds"]
    assert len(folds) == 3
    for fold in folds:
        assert fold["holdout_seed"] not in fold["training_seeds"]
        assert not (set(fold["holdout_ids"]) & set(fold["training_ids"]))
        assert len(fold["holdout_ids"]) == 8
    fewer = [e for e in episodes() if e["seed"] != 2]
    with pytest.raises(ValueError, match="at least three"):
        ValueAwareDiagnosticLearner().fit(fewer)
    # Logs-only and an always-paid expert do not train a conditional gate.
    ValueAwareDiagnosticLearner().fit_fixed(fewer)
    ValueAwareDiagnosticLearner().fit_fixed(fewer, P, always_probe=True)


def test_budget_precedes_query_and_counts_unique_vectors():
    rows = episodes(useful=True)
    for row in rows:
        row["probe_features"] = Forbidden(row["probe_features"])
    for strategy in ("counterexample", "random", "enumeration"):
        model = ValueAwareDiagnosticLearner(strategy=strategy, search_budget_cells=0).fit(rows, [P, Q])
        assert model.search_report["revealed_cells"] == 0
    rows = episodes(useful=True)
    for budget in (1, 10, 24, 30, 50, 100):
        model = ValueAwareDiagnosticLearner(search_budget_cells=budget).fit(rows)
        queries = model.search_report["queries"]
        assert len(queries) <= budget
        assert len(queries) == len({(q["episode_id"], q["probe_id"]) for q in queries})
        assert all(q["split"] != "test" for q in queries)
        assert model.search_report["replay_probe_cost"] == sum(q["cost"] for q in queries)


def test_always_probe_expert_and_free_baseline_have_honest_costs():
    rows = episodes()
    free = ValueAwareDiagnosticLearner().fit_fixed(rows).evaluate(rows)
    expert = ValueAwareDiagnosticLearner().fit_fixed(rows, P, always_probe=True).evaluate(rows)
    assert free["mean_dev_loss"] == 2.0
    assert expert["mean_dev_loss"] == 3.0
    assert free["evaluation_probe_cost"] == 0
    assert expert["evaluation_probe_cost"] == 16 * 16
    assert expert["search_revealed_cells"] == 24


def test_custom_catalog_ids_are_allowed():
    rows = episodes(useful=True)
    custom = "lr_half:1:next_batch_loss"
    for row in rows:
        row["probe_features"][custom] = row["probe_features"].pop(P)
        row["probe_costs"][custom] = row["probe_costs"].pop(P)
    model = ValueAwareDiagnosticLearner(search_budget_cells=200).fit(rows)
    assert model.selected_probe == custom


def test_shorter_logs_baseline_routes_labels_without_buying_probe():
    rows = episodes()
    for row in rows:
        row["budget_losses"]["2"] = [10 + float(int(row["id"][-1]) % 2 * 2 - 1),
                                      10 - float(int(row["id"][-1]) % 2 * 2 - 1)]
        row["probe_features"] = Forbidden(row["probe_features"])
        row["probe_costs"] = Forbidden(row["probe_costs"])
    model = ValueAwareDiagnosticLearner(search_budget_cells=0).fit_fixed(rows, budget_cost=2)
    result = model.evaluate(rows)
    assert result["mean_dev_loss"] == 10.0
    assert result["probe_costs"] == [0] * 16
    assert result["budget_costs"] == [2] * 16
    assert result["evaluation_revealed_cells"] == 0
    restored = ValueAwareDiagnosticLearner.from_state_dict(json.loads(json.dumps(model.state_dict())))
    assert restored.evaluate(rows) == result


def test_conditional_gate_reads_only_beneficial_episode_subset():
    rows = episodes(useful=True)
    for row in rows:
        regime = float((int(row["id"][-1]) // 2) % 2)
        bit = row["probe_features"][P][0]
        row["log_features"] = [regime]
        row["budget_losses"]["16"] = [2.1 + bit, 2.1 - bit] if regime == 0 else [4 + bit, 4 - bit]
    fixed = ValueAwareDiagnosticLearner(search_budget_cells=200).fit_fixed(rows, P)
    for row in rows:
        if row["split"] == "test" and row["log_features"][0] == 1:
            row["probe_features"] = Forbidden(row["probe_features"])
            row["probe_costs"] = Forbidden(row["probe_costs"])
    result = fixed.evaluate(rows)
    assert result["evaluation_revealed_cells"] == 8
    assert result["evaluation_probe_cost"] == 8 * 16
    assert result["mean_dev_loss"] == pytest.approx((1.1 + 2.0) / 2)
