"""Synthetic fixtures check bookkeeping/selection isolation, not scientific gains."""

import copy
import json
import unittest

import numpy as np

from branchlab.synthesis import (GreedyProgramSynthesizer, ProbeSpec,
                                candidate_probe_specs, evaluate_fixed_probes,
                                evaluate_no_probe, evaluate_oracle)


P1 = "lr_half:2:loss_delta"
P2 = "momentum_zero:4:recovery_slope"
P3 = "lr_half:8:grad_alignment"


def episodes():
    rows = []
    for split_index, split in enumerate(("discovery", "development", "test")):
        for i in range(24):
            x = (-1.0, 1.0)[i % 2]
            z = (-1.0, 1.0)[(i // 2) % 2]
            rows.append({"id": f"{split}-{i:03d}", "seed": split_index,
                         "split": split, "log_features": [1.0],
                         "probe_features": {P1: [x], P2: [z], P3: [0.0]},
                         "repair_losses": [2 + x + z, 2 + x - z, 2 - x + z, 2 - x - z],
                         "probe_costs": {P1: 4.0, P2: 8.0, P3: 16.0}})
    return rows


class ForbiddenMapping(dict):
    def __getitem__(self, key):
        raise AssertionError("An unselected probe was read")


class SelectiveMapping(dict):
    def __init__(self, data, allowed):
        super().__init__(data)
        self.allowed = set(allowed)

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError(f"Forbidden probe read: {key}")
        return super().__getitem__(key)


class SynthesisTests(unittest.TestCase):
    def test_probe_spec_validation_and_catalog(self):
        self.assertEqual(len(candidate_probe_specs()), 18)
        self.assertEqual(ProbeSpec.from_id(P1), ProbeSpec("lr_half", 2, "loss_delta"))
        self.assertEqual(ProbeSpec("keep", 8, "grad_alignment").id, "keep:8:grad_alignment")
        with self.assertRaises(ValueError):
            ProbeSpec("erase_weights", 2, "loss_delta")
        with self.assertRaises(ValueError):
            ProbeSpec("keep", 7, "loss_delta")

    def test_test_labels_and_features_never_influence_search(self):
        original = episodes()
        modified = copy.deepcopy(original)
        for row in modified:
            if row["split"] == "test":
                row["repair_losses"] = "not even a valid target"
                row["probe_features"] = ForbiddenMapping(row["probe_features"])
                row["log_features"] = [float("nan")]
        for strategy in ("counterexample", "random", "enumeration"):
            first = GreedyProgramSynthesizer(strategy=strategy, search_budget_cells=150).fit(original)
            second = GreedyProgramSynthesizer(strategy=strategy, search_budget_cells=150).fit(modified)
            self.assertEqual(first.state_dict(), second.state_dict())

    def test_insufficient_budget_never_reads_candidate_values(self):
        rows = episodes()
        for row in rows:
            row["probe_features"] = ForbiddenMapping(row["probe_features"])
            row["probe_costs"] = ForbiddenMapping(row["probe_costs"])
        # A complete candidate requires 24 discovery + 24 development vectors.
        # With 47 cells the CE strategy cannot scout while preserving that reserve.
        for strategy in ("counterexample", "random", "enumeration"):
            for budget in (0, 47):
                model = GreedyProgramSynthesizer(strategy=strategy, search_budget_cells=budget).fit(
                    rows, candidate_ids=[P1, P2, P3])
                self.assertEqual(model.selected_probes, [])
                self.assertEqual(model.search_report["revealed_cells"], 0)

    def test_budget_is_unique_vectors_and_repeat_queries_free(self):
        model = GreedyProgramSynthesizer(strategy="enumeration", search_budget_cells=144).fit(episodes())
        report = model.search_report
        self.assertEqual(report["revealed_cells"], 144)
        self.assertEqual(report["revealed_scalars"], 144)
        self.assertEqual(report["replay_probe_cost"], 48 * (4 + 8 + 16))
        self.assertEqual(len({(q["episode_id"], q["probe_id"]) for q in report["queries"]}), 144)
        # Generation two reuses the same rows and probes instead of charging twice.
        self.assertLessEqual(model.history[-1]["revealed_cells"], 144)
        for budget in (0, 1, 30, 48, 77, 145):
            for strategy in ("counterexample", "random", "enumeration"):
                result = GreedyProgramSynthesizer(strategy=strategy, search_budget_cells=budget).fit(episodes())
                self.assertLessEqual(result.search_report["revealed_cells"], budget)
                self.assertTrue(all(q["split"] != "test" for q in result.search_report["queries"]))

    def test_inheritance_is_monotone_and_stable(self):
        model = GreedyProgramSynthesizer(strategy="enumeration", search_budget_cells=144).fit(episodes())
        self.assertEqual(len(model.selected_probes), 2)
        previous = []
        previous_regret = float("inf")
        for generation in model.history:
            selected = generation["selected_probes"]
            self.assertEqual(selected[:len(previous)], previous)
            self.assertLessEqual(generation["development_regret"], previous_regret)
            previous, previous_regret = selected, generation["development_regret"]
        reverse = GreedyProgramSynthesizer(strategy="enumeration", search_budget_cells=144).fit(list(reversed(episodes())))
        self.assertEqual(model.state_dict(), reverse.state_dict())

    def test_promotion_requires_development_gain(self):
        rows = episodes()
        for row in rows:
            if row["split"] == "development":
                row["repair_losses"] = [0.0, 1.0, 1.0, 1.0]
        model = GreedyProgramSynthesizer(strategy="enumeration").fit(rows)
        self.assertEqual(model.selected_probes, [])
        self.assertEqual(model.search_report["selected_development_regret"], 0.0)

    def test_json_roundtrip_and_test_evaluation_is_frozen(self):
        model = GreedyProgramSynthesizer(strategy="enumeration").fit(episodes())
        state = model.state_dict()
        restored = GreedyProgramSynthesizer.from_state_dict(json.loads(json.dumps(state)))
        self.assertEqual(model.evaluate(episodes()), restored.evaluate(episodes()))
        self.assertEqual(model.state_dict(), state)
        altered = episodes()
        for row in altered:
            if row["split"] == "test":
                row["repair_losses"] = list(reversed(row["repair_losses"]))
        self.assertEqual(model.predict(episodes()), model.predict(altered))
        model.evaluate(altered)
        self.assertEqual(model.state_dict(), state)

    def test_test_only_selected_probe_access_and_cost(self):
        rows = episodes()
        model = GreedyProgramSynthesizer(max_probes=1, search_budget_cells=100).fit_fixed(rows, [P1])
        for row in rows:
            if row["split"] == "test":
                row["probe_features"] = SelectiveMapping(row["probe_features"], [P1])
                row["probe_costs"] = SelectiveMapping(row["probe_costs"], [P1])
        result = model.evaluate(rows)
        self.assertEqual(result["evaluation_revealed_cells"], 24)
        self.assertEqual(result["evaluation_probe_cost"], 24 * 4)
        self.assertEqual(result["search_revealed_cells"], 24)

    def test_no_probe_is_not_the_hindsight_oracle(self):
        rows = episodes()
        for row in rows:
            row["probe_features"] = ForbiddenMapping(row["probe_features"])
            row["probe_costs"] = ForbiddenMapping(row["probe_costs"])
        baseline = evaluate_no_probe(rows)
        oracle = evaluate_oracle(rows)
        self.assertGreater(baseline["mean_regret"], 0)
        self.assertEqual(baseline["evaluation_probe_cost"], 0)
        self.assertEqual(oracle["mean_regret"], 0)
        self.assertIsNone(oracle["evaluation_probe_cost"])
        self.assertEqual(oracle["benchmark_type"], "hindsight_oracle_not_deployable")

    def test_fixed_expert_helper(self):
        rows = episodes()
        report = evaluate_fixed_probes(rows, [P1])
        self.assertEqual(report["selected_probes"], [P1])
        self.assertEqual(report["search_revealed_cells"], 24)
        self.assertEqual(evaluate_fixed_probes(rows, []), evaluate_no_probe(rows))

    def test_seed_repeatability_and_invalid_split_ids(self):
        for strategy in ("counterexample", "random", "enumeration"):
            a = GreedyProgramSynthesizer(strategy=strategy, seed=17, search_budget_cells=100).fit(episodes())
            b = GreedyProgramSynthesizer(strategy=strategy, seed=17, search_budget_cells=100).fit(episodes())
            self.assertEqual(a.state_dict(), b.state_dict())
        rows = episodes()
        rows[24]["id"] = rows[0]["id"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            GreedyProgramSynthesizer().fit(rows)

    def test_counterexample_scouts_are_paid_and_training_only(self):
        model = GreedyProgramSynthesizer(search_budget_cells=144).fit(episodes())
        scouts = [q for q in model.search_report["queries"] if q["phase"] == "counterexample_scout"]
        self.assertGreater(len(scouts), 0)
        self.assertTrue(all(q["split"] == "discovery" for q in scouts))
        self.assertEqual(model.search_report["revealed_cells"], len(model.search_report["queries"]))


if __name__ == "__main__":
    unittest.main()
