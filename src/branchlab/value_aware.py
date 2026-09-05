"""One-probe diagnostic policies with a cost-aligned objective and a skip gate.

An episode adds ``budget_losses`` to the v1 table: each integer-cost string maps
to action dev-text losses at the *actual remaining-budget horizon*. Discovery
fits all models, whole-discovery-seed cross-fitting supplies gate labels, and
development promotes complete policies by net-budget dev loss. No test-text
outcomes belong in this table. Former v1 holdouts are development data for v2.

This is a fixed finite policy-search implementation, not an RSI claim. Probe
cost must be a fixed positive integer for each DSL production. Search charges
unique episode/probe vectors before reading them; physical table construction
cost remains separate. A skipped probe is never read, including at evaluation.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .synthesis import _Ridge, _matrix, _split, _vector


Episode = Mapping[str, Any]


def _probe_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Probe IDs must be nonempty catalog strings")
    return value


def _cost(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Probe costs must be positive integers")
    number = float(value)
    if not np.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError("Probe costs must be positive integers")
    return int(number)


def _labels(rows: Sequence[Episode], costs: Sequence[int] | int) -> np.ndarray:
    if isinstance(costs, int):
        costs = [costs] * len(rows)
    if len(costs) != len(rows):
        raise ValueError("Every episode needs one budget-label cost")
    values = [_vector(e["budget_losses"][str(cost)], f"budget_losses[{cost}]")
              for e, cost in zip(rows, costs)]
    if not values or len({v.size for v in values}) != 1 or values[0].size < 1:
        raise ValueError("Budget labels need consistent, nonempty action vectors")
    return np.stack(values)


def _gate_features(logs: np.ndarray, free_predictions: np.ndarray) -> np.ndarray:
    ordered = np.sort(free_predictions, axis=1)
    margin = ordered[:, 1] - ordered[:, 0] if ordered.shape[1] > 1 else np.zeros(len(logs))
    return np.column_stack((logs, margin))


class _Ledger:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.cache: dict[tuple[str, str], np.ndarray] = {}
        self.queries: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return self.budget - len(self.cache)

    def missing(self, rows: Sequence[Episode], probe: str) -> int:
        return sum((str(e["id"]), probe) not in self.cache for e in rows)

    def read(self, rows: Sequence[Episode], probe: str, cost: int, phase: str) -> np.ndarray:
        # Check the entire requested subset before touching a feature value.
        if self.missing(rows, probe) > self.remaining:
            raise RuntimeError("Insufficient observation budget")
        values = []
        for e in rows:
            if e["split"] not in ("discovery", "development"):
                raise ValueError("Search must not read test observations")
            key = (str(e["id"]), probe)
            if key not in self.cache:
                actual_cost = _cost(e["probe_costs"][probe])
                if actual_cost != cost:
                    raise ValueError("A probe production must have a fixed cost")
                value = _vector(e["probe_features"][probe], probe)
                if not value.size:
                    raise ValueError("Probe observations cannot be empty")
                self.cache[key] = value.copy()
                self.queries.append({"episode_id": key[0], "split": e["split"],
                                     "probe_id": probe, "cost": cost, "phase": phase,
                                     "scalars": value.size})
            values.append(self.cache[key])
        if len({len(v) for v in values}) != 1:
            raise ValueError("Inconsistent probe feature dimensions")
        return np.stack(values)

    def report(self) -> dict[str, Any]:
        return {"budget_cells": self.budget, "revealed_cells": len(self.cache),
                "revealed_scalars": sum(q["scalars"] for q in self.queries),
                "replay_probe_cost": sum(q["cost"] for q in self.queries),
                "queries": copy.deepcopy(self.queries),
                "cost_scope": "Unique vector queries in replay; actual table collection is separate"}


class ValueAwareDiagnosticLearner:
    """Fit a conditional single-probe policy with an explicit no-probe fallback.

    Search order can be counterexample, random, or enumeration. All share the
    same cross-fitted ridge heads, gate, observation cap, and development metric.
    ``gate_margin`` is fixed before fit; this class never tunes it on a holdout.
    """

    def __init__(self, *, strategy: str = "counterexample", search_budget_cells: int = 150,
                 seed: int = 0, ridge_alpha: float = 1.0, gate_margin: float = 0.0,
                 min_improvement: float = 0.0, counterexample_count: int = 6) -> None:
        if strategy not in ("counterexample", "random", "enumeration"):
            raise ValueError("Unknown candidate ordering")
        if isinstance(search_budget_cells, bool) or int(search_budget_cells) != search_budget_cells or search_budget_cells < 0:
            raise ValueError("search_budget_cells must be a nonnegative integer")
        if ridge_alpha <= 0 or not np.isfinite(ridge_alpha):
            raise ValueError("ridge_alpha must be positive and finite")
        if gate_margin < 0 or not np.isfinite(gate_margin):
            raise ValueError("gate_margin must be finite and nonnegative")
        if min_improvement < 0 or not np.isfinite(min_improvement):
            raise ValueError("min_improvement must be finite and nonnegative")
        if counterexample_count < 3:
            raise ValueError("counterexample_count must be at least three")
        self.strategy, self.search_budget_cells, self.seed = strategy, int(search_budget_cells), int(seed)
        self.ridge_alpha, self.gate_margin = float(ridge_alpha), float(gate_margin)
        self.min_improvement, self.counterexample_count = float(min_improvement), int(counterexample_count)
        self.selected_probe: str | None = None
        self.probe_cost = 0
        self.always_probe = False
        self.base_budget_cost = 0
        self.search_report: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self._free: _Ridge | None = None
        self._head: _Ridge | None = None
        self._gate: _Ridge | None = None
        self._crossfit_folds: list[dict[str, Any]] = []

    @property
    def selected_probes(self) -> list[str]:
        return [] if self.selected_probe is None else [self.selected_probe]

    def _setup(self, discovery: Sequence[Episode], budget_cost: int = 0) -> tuple[np.ndarray, np.ndarray]:
        x, y = _matrix(discovery, "log_features"), _labels(discovery, budget_cost)
        self._free = _Ridge.fit(x, y, self.ridge_alpha)
        self._head = self._gate = None
        self.selected_probe, self.probe_cost, self.always_probe = None, 0, False
        self.base_budget_cost = budget_cost
        self.history, self._crossfit_folds = [], []
        return x, y

    @staticmethod
    def _folds(discovery: Sequence[Episode]) -> list[tuple[list[int], list[int]]]:
        seeds = sorted({e["seed"] for e in discovery}, key=str)
        if len(seeds) < 3:
            raise ValueError("Conditional gates require at least three discovery training seeds")
        return [([i for i, e in enumerate(discovery) if e["seed"] != seed],
                 [i for i, e in enumerate(discovery) if e["seed"] == seed]) for seed in seeds]

    def _free_oof(self, discovery: Sequence[Episode], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        output = np.empty_like(y)
        self._crossfit_folds = []
        for train, holdout in self._folds(discovery):
            model = _Ridge.fit(x[train], y[train], self.ridge_alpha)
            output[holdout] = model.predict(x[holdout])
            self._crossfit_folds.append({"training_seeds": sorted({discovery[i]["seed"] for i in train}, key=str),
                                        "holdout_seed": discovery[holdout[0]]["seed"],
                                        "training_ids": [str(discovery[i]["id"]) for i in train],
                                        "holdout_ids": [str(discovery[i]["id"]) for i in holdout]})
        return output

    def _fit_probe(self, discovery: Sequence[Episode], x: np.ndarray, y0: np.ndarray,
                   free_oof: np.ndarray | None, probe_values: np.ndarray, cost: int,
                   always_probe: bool = False) -> tuple[_Ridge, _Ridge | None, list[float]]:
        yp = _labels(discovery, cost)
        if yp.shape != y0.shape:
            raise ValueError("All budget horizons must share the same action menu")
        xp = np.concatenate((x, probe_values), axis=1)
        head = _Ridge.fit(xp, yp, self.ridge_alpha)
        if always_probe:
            return head, None, []
        if free_oof is None:
            raise ValueError("A conditional probe requires out-of-fold free predictions")
        paid_oof = np.empty_like(yp)
        for train, holdout in self._folds(discovery):
            paid_oof[holdout] = _Ridge.fit(xp[train], yp[train], self.ridge_alpha).predict(xp[holdout])
        index = np.arange(len(discovery))
        gain = y0[index, np.argmin(free_oof, axis=1)] - yp[index, np.argmin(paid_oof, axis=1)]
        gate = _Ridge.fit(_gate_features(x, free_oof), gain[:, None], self.ridge_alpha)
        return head, gate, gain.tolist()

    def _candidate_predictions(self, rows: Sequence[Episode], probe: str, cost: int,
                               head: _Ridge, gate: _Ridge | None, ledger: _Ledger,
                               always_probe: bool = False) -> tuple[np.ndarray, np.ndarray]:
        assert self._free is not None
        x = _matrix(rows, "log_features")
        prediction = self._free.predict(x)
        use = np.ones(len(rows), dtype=bool) if always_probe else gate.predict(_gate_features(x, prediction))[:, 0] > self.gate_margin
        selected = np.flatnonzero(use)
        if len(selected):
            values = ledger.read([rows[i] for i in selected], probe, cost, "development_gate_positive")
            prediction[selected] = head.predict(np.concatenate((x[selected], values), axis=1))
        return prediction, use

    def _candidate_cost(self, rows: Sequence[Episode], probe: str) -> int:
        cost = _cost(rows[0]["probe_costs"][probe])
        if any(_cost(e["probe_costs"][probe]) != cost for e in rows[1:]):
            raise ValueError("Each candidate must use one fixed DSL execution cost")
        return cost

    def _order(self, candidates: list[str], discovery: Sequence[Episode], development_size: int,
               y0: np.ndarray, oof: np.ndarray, costs: Mapping[str, int], ledger: _Ledger) -> list[str]:
        if self.strategy == "random":
            np.random.default_rng(self.seed).shuffle(candidates)
            return candidates
        if self.strategy == "enumeration":
            return candidates
        selected = np.argmin(oof, axis=1)
        regret = y0[np.arange(len(y0)), selected] - y0.min(axis=1)
        residual = y0 - oof
        ids = sorted(range(len(y0)), key=lambda i: (-float(regret[i]), -float(np.linalg.norm(residual[i])), str(discovery[i]["id"])))[:self.counterexample_count]
        rows, target = [discovery[i] for i in ids], residual[ids]
        allowance = max(0, min(ledger.remaining // 4, ledger.remaining - len(discovery) - development_size))
        ranked, spent = [], 0
        for probe in candidates:
            needed = ledger.missing(rows, probe)
            if len(rows) < 3 or spent + needed > allowance:
                continue
            values = ledger.read(rows, probe, costs[probe], "discovery_counterexample_scout")
            spent += needed
            error = null = 0.0
            # Candidate-order heuristic only. Whole-seed cross-fitting remains
            # mandatory for every policy gate and final development promotion.
            for heldout in range(len(rows)):
                mask = np.arange(len(rows)) != heldout
                estimate = _Ridge.fit(values[mask], target[mask], self.ridge_alpha).predict(values[heldout:heldout + 1])[0]
                error += float(np.sum((estimate - target[heldout]) ** 2))
                null += float(np.sum((target[mask].mean(axis=0) - target[heldout]) ** 2))
            ranked.append(((null - error) / len(rows), probe))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        ordered = [p for _, p in ranked]
        return ordered + [p for p in candidates if p not in ordered]

    def fit(self, episodes: Iterable[Episode], candidate_ids: Sequence[str] | None = None) -> "ValueAwareDiagnosticLearner":
        all_rows = list(episodes)
        discovery, development = _split(all_rows, "discovery"), _split(all_rows, "development")
        if {str(e["id"]) for e in discovery} & {str(e["id"]) for e in development}:
            raise ValueError("Discovery and development IDs must be disjoint")
        if {e["seed"] for e in discovery} & {e["seed"] for e in development}:
            raise ValueError("Discovery and development training seeds must be disjoint")
        x, y0 = self._setup(discovery)
        ydev = _labels(development, 0)
        if ydev.shape[1] != y0.shape[1]:
            raise ValueError("Discovery/development action menus differ")
        candidates = sorted(set(candidate_ids)) if candidate_ids is not None else sorted(discovery[0]["probe_features"].keys())
        for probe in candidates:
            _probe_id(probe)
        costs = {p: self._candidate_cost(discovery + development, p) for p in candidates}
        ledger = _Ledger(self.search_budget_cells)
        assert self._free is not None
        basepred = self._free.predict(_matrix(development, "log_features"))
        baseline = float(np.mean(ydev[np.arange(len(development)), np.argmin(basepred, axis=1)]))
        best_loss = baseline
        self.history = [{"candidate": None, "development_net_loss": baseline, "promoted": False}]
        oof = self._free_oof(discovery, x, y0) if candidates else None
        ordered = self._order(candidates, discovery, len(development), y0, oof, costs, ledger) if candidates else []
        for probe in ordered:
            if ledger.missing(discovery, probe) > ledger.remaining:
                continue
            values = ledger.read(discovery, probe, costs[probe], "discovery_head_and_crossfit")
            head, gate, targets = self._fit_probe(discovery, x, y0, oof, values, costs[probe])
            assert gate is not None
            use = gate.predict(_gate_features(_matrix(development, "log_features"), basepred))[:, 0] > self.gate_margin
            needed_rows = [e for i, e in enumerate(development) if use[i]]
            if ledger.missing(needed_rows, probe) > ledger.remaining:
                self.history.append({"candidate": probe, "status": "insufficient_budget_for_gate_positive_development",
                                     "promoted": False, "revealed_cells": len(ledger.cache)})
                continue
            prediction, use = self._candidate_predictions(development, probe, costs[probe], head, gate, ledger)
            yc = ydev.copy()
            if np.any(use):
                yc[use] = _labels([e for i, e in enumerate(development) if use[i]], costs[probe])
            loss = float(np.mean(yc[np.arange(len(development)), np.argmin(prediction, axis=1)]))
            promoted = loss < best_loss - self.min_improvement - 1e-12
            if promoted:
                best_loss = loss
                self.selected_probe, self.probe_cost = probe, costs[probe]
                self._head, self._gate = head, gate
            self.history.append({"candidate": probe, "development_net_loss": loss,
                                 "development_probe_episodes": int(np.sum(use)),
                                 "promoted": promoted, "revealed_cells": len(ledger.cache),
                                 "crossfit_net_gains": targets})
        self.search_report = ledger.report()
        self.search_report.update({"strategy": self.strategy, "candidate_count": len(candidates),
                                   "baseline_development_net_loss": baseline, "selected_development_net_loss": best_loss,
                                   "selection_split": "development", "fit_split": "discovery",
                                   "gate_crossfit_unit": "whole discovery training seed",
                                   "crossfit_folds": copy.deepcopy(self._crossfit_folds),
                                   "test_used_for_selection": False, "objective": "cost-adjusted remaining-horizon dev-text loss"})
        return self

    def fit_fixed(self, episodes: Iterable[Episode], probe_id: str | None = None, *,
                  always_probe: bool = False, budget_cost: int = 0) -> "ValueAwareDiagnosticLearner":
        """Fit a preregistered expert/logs policy, without development selection.

        ``budget_cost>0`` with no probe selects a shorter repair horizon from
        that label entry while actually acquiring no observation. Predictions
        separate ``budget_costs`` (horizon routing) from real ``probe_costs``.
        This supplies the logs-only shorter-training control, not a fake charge.
        """
        if isinstance(budget_cost, bool) or not isinstance(budget_cost, int) or budget_cost < 0:
            raise ValueError("budget_cost must be a nonnegative integer")
        if probe_id is not None and budget_cost:
            raise ValueError("An explicit shorter-horizon baseline cannot also buy a probe")
        discovery = _split(list(episodes), "discovery")
        x, y0 = self._setup(discovery, budget_cost)
        ledger = _Ledger(self.search_budget_cells)
        if probe_id is not None:
            _probe_id(probe_id)
            cost = self._candidate_cost(discovery, probe_id)
            values = ledger.read(discovery, probe_id, cost, "fixed_discovery_head")
            oof = None if always_probe else self._free_oof(discovery, x, y0)
            head, gate, targets = self._fit_probe(discovery, x, y0, oof, values, cost, always_probe)
            self.selected_probe, self.probe_cost, self.always_probe = probe_id, cost, always_probe
            self._head, self._gate = head, gate
            self.history = [{"candidate": probe_id, "crossfit_net_gains": targets, "selection": "preregistered_fixed"}]
        self.search_report = ledger.report()
        self.search_report.update({"strategy": "fixed_always_probe" if always_probe else "fixed_conditional",
                                   "fit_split": "discovery", "selection_split": None,
                                   "gate_crossfit_unit": "whole discovery training seed" if probe_id and not always_probe else None,
                                   "crossfit_folds": copy.deepcopy(self._crossfit_folds), "test_used_for_selection": False})
        return self

    def predict(self, episodes: Iterable[Episode], *, split: str = "test") -> dict[str, Any]:
        """Gate before any probe/cost read. No budget labels are accessed here."""
        if self._free is None:
            raise RuntimeError("Fit or load the learner before predicting")
        rows = _split(episodes, split)
        x = _matrix(rows, "log_features")
        prediction = self._free.predict(x)
        use = np.zeros(len(rows), dtype=bool)
        if self.selected_probe is not None:
            use = np.ones(len(rows), dtype=bool) if self.always_probe else self._gate.predict(_gate_features(x, prediction))[:, 0] > self.gate_margin
            indices = np.flatnonzero(use)
            if len(indices):
                values = []
                for i in indices:
                    row = rows[i]
                    if _cost(row["probe_costs"][self.selected_probe]) != self.probe_cost:
                        raise ValueError("Prediction probe cost differs from the fitted DSL")
                    values.append(_vector(row["probe_features"][self.selected_probe], self.selected_probe))
                prediction[indices] = self._head.predict(np.concatenate((x[indices], np.stack(values)), axis=1))
        chosen = np.argmin(prediction, axis=1).tolist()
        probe_ids = [self.selected_probe if active else None for active in use]
        costs = [self.probe_cost if active else 0 for active in use]
        budget_costs = [self.probe_cost if active else self.base_budget_cost for active in use]
        ids = [str(e["id"]) for e in rows]
        return {"episode_ids": ids, "chosen_repairs": chosen, "probe_ids": probe_ids, "probe_costs": costs,
                "budget_costs": budget_costs,
                "selected_probes": self.selected_probes, "evaluation_revealed_cells": int(np.sum(use)),
                "evaluation_probe_cost": sum(costs), "mean_probe_cost": float(np.mean(costs)),
                "search_revealed_cells": self.search_report["revealed_cells"],
                "search_replay_probe_cost": self.search_report["replay_probe_cost"],
                "search_report": copy.deepcopy(self.search_report),
                "decisions": [{"episode_id": eid, "action": a, "probe_id": p, "probe_cost": c, "budget_cost": b}
                              for eid, a, p, c, b in zip(ids, chosen, probe_ids, costs, budget_costs)]}

    def evaluate(self, episodes: Iterable[Episode], *, split: str = "test") -> dict[str, Any]:
        """Report net-budget dev-text loss only; never a final test-text score."""
        rows = list(episodes)
        prediction = self.predict(rows, split=split)
        target = _labels(_split(rows, split), prediction["budget_costs"])
        losses = target[np.arange(len(target)), prediction["chosen_repairs"]].tolist()
        return {**prediction, "mean_dev_loss": float(np.mean(losses)), "dev_losses": losses,
                "evaluation_target": "dev-text loss at each decision's remaining-budget horizon", "split": split}

    def state_dict(self) -> dict[str, Any]:
        if self._free is None:
            raise RuntimeError("Cannot serialize an unfitted learner")
        parameters = {k: getattr(self, k) for k in ("strategy", "search_budget_cells", "seed", "ridge_alpha",
                      "gate_margin", "min_improvement", "counterexample_count")}
        return {"format_version": 1, "learner": "value_aware_single_probe", "parameters": parameters,
                "selected_probe": self.selected_probe, "probe_cost": self.probe_cost, "always_probe": self.always_probe,
                "base_budget_cost": self.base_budget_cost,
                "free": self._free.state_dict(), "head": self._head.state_dict() if self._head else None,
                "gate": self._gate.state_dict() if self._gate else None, "history": copy.deepcopy(self.history),
                "search_report": copy.deepcopy(self.search_report)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ValueAwareDiagnosticLearner":
        if state.get("format_version") != 1 or state.get("learner") != "value_aware_single_probe":
            raise ValueError("Unsupported value-aware state format")
        obj = cls(**state["parameters"])
        obj.selected_probe, obj.probe_cost, obj.always_probe = state["selected_probe"], state["probe_cost"], state["always_probe"]
        obj.base_budget_cost = state.get("base_budget_cost", 0)
        obj._free = _Ridge.from_state_dict(state["free"])
        obj._head = _Ridge.from_state_dict(state["head"]) if state["head"] is not None else None
        obj._gate = _Ridge.from_state_dict(state["gate"]) if state["gate"] is not None else None
        obj.history, obj.search_report = copy.deepcopy(state["history"]), copy.deepcopy(state["search_report"])
        return obj


def evaluate_fixed_value_aware(episodes: Iterable[Episode], probe_id: str | None = None, *,
                              always_probe: bool = False, budget_cost: int = 0,
                              split: str = "test", **kwargs: Any) -> dict[str, Any]:
    rows = list(episodes)
    model = ValueAwareDiagnosticLearner(**kwargs).fit_fixed(rows, probe_id, always_probe=always_probe, budget_cost=budget_cost)
    return model.evaluate(rows, split=split)
