"""Budgeted, auditable diagnostic-program search over real training episodes.

An episode is a mapping with ``id``, ``seed``, ``split`` (discovery,
development, or test), ``log_features``, ``probe_features``, ``repair_losses``,
and ``probe_costs``. A probe vector is a *paid* observation: search accesses it
only through a query ledger. One revealed cell means one episode/probe vector,
not one scalar and not one physical training branch. Search replay costs and
evaluation probe costs are reported separately from the actual cost of creating
the complete input table, which must be recorded by its producer.

The finite DSL consists of action, horizon, and readout. A program concatenates
at most ``max_probes`` such observations with free training logs. Its ridge
predictor chooses the repair with lowest predicted continuation loss. Generation
g+1 inherits generation g's promoted observations. This implements diagnostic
program accumulation, not evidence that the synthesizer recursively improves.

Only discovery episodes fit predictors. Only development regret promotes a
program. Test labels are read exclusively by ``evaluate`` after selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Episode = Mapping[str, Any]
ACTIONS = ("keep", "lr_half", "momentum_zero")
HORIZONS = (2, 4, 8)
READOUTS = ("loss_delta", "recovery_slope", "grad_alignment")


@dataclass(frozen=True, order=True)
class ProbeSpec:
    """Two-branch probe: this action versus keep at the matching horizon."""

    action: str
    steps: int
    readout: str

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown probe action: {self.action}")
        if self.steps not in HORIZONS:
            raise ValueError(f"Probe steps must be one of {HORIZONS}")
        if self.readout not in READOUTS:
            raise ValueError(f"Unknown probe readout: {self.readout}")

    @property
    def id(self) -> str:
        return f"{self.action}:{self.steps}:{self.readout}"

    @classmethod
    def from_id(cls, probe_id: str) -> "ProbeSpec":
        action, steps, readout = probe_id.split(":")
        return cls(action, int(steps), readout)


def candidate_probe_specs() -> list[ProbeSpec]:
    """The 18 nontrivial productions; keep is the reference action."""
    return [ProbeSpec(a, h, r) for a in ACTIONS[1:] for h in HORIZONS for r in READOUTS]


def _vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return arr


def _matrix(rows: Sequence[Episode], key: str) -> np.ndarray:
    vectors = [_vector(row[key], key) for row in rows]
    if not vectors or len({len(v) for v in vectors}) != 1:
        raise ValueError(f"{key} dimensions must agree across nonempty episodes")
    return np.stack(vectors)


def _split(episodes: Iterable[Episode], split: str) -> list[Episode]:
    # Do not validate unrelated episodes: in particular, do not read their labels.
    rows = sorted((e for e in episodes if e["split"] == split), key=lambda e: str(e["id"]))
    if not rows:
        raise ValueError(f"No {split!r} episodes supplied")
    ids = [str(e["id"]) for e in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate episode IDs in {split}")
    return rows


class _BudgetExhausted(RuntimeError):
    pass


class _QueryLedger:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.cache: dict[tuple[str, str], np.ndarray] = {}
        self.scalar_count = 0
        self.replay_cost = 0.0
        self.records: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return self.budget - len(self.cache)

    def missing(self, rows: Sequence[Episode], probes: Sequence[str]) -> int:
        return sum((str(e["id"]), p) not in self.cache for e in rows for p in probes)

    def read(self, episode: Episode, probe: str, phase: str) -> np.ndarray:
        key = (str(episode["id"]), probe)
        if key in self.cache:
            return self.cache[key]
        if episode["split"] not in ("discovery", "development"):
            raise ValueError("Search queries may never read test features")
        if self.remaining <= 0:
            raise _BudgetExhausted("Probe query budget exhausted")
        # Access happens after the budget and split gates, never before them.
        vector = _vector(episode["probe_features"][probe], f"probe {probe}")
        if vector.size == 0:
            raise ValueError("Probe vectors must not be empty")
        cost = float(episode["probe_costs"][probe])
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("Probe costs must be finite and nonnegative")
        self.cache[key] = vector.copy()
        self.scalar_count += len(vector)
        self.replay_cost += cost
        self.records.append({"episode_id": key[0], "split": episode["split"],
                             "probe_id": probe, "phase": phase, "cost": cost,
                             "scalars": len(vector)})
        return self.cache[key]

    def features(self, rows: Sequence[Episode], probes: Sequence[str], phase: str) -> np.ndarray:
        if self.missing(rows, probes) > self.remaining:
            raise _BudgetExhausted("Insufficient budget for this complete candidate evaluation")
        logs = _matrix(rows, "log_features")
        columns = [logs]
        for probe in probes:
            vectors = [self.read(e, probe, phase) for e in rows]
            if len({len(v) for v in vectors}) != 1:
                raise ValueError(f"Inconsistent vector dimension for {probe}")
            columns.append(np.stack(vectors))
        return np.concatenate(columns, axis=1)

    def summary(self) -> dict[str, Any]:
        return {"budget_cells": self.budget, "revealed_cells": len(self.cache),
                "revealed_scalars": self.scalar_count, "replay_probe_cost": self.replay_cost,
                "cost_interpretation": "sum of supplied episode/probe costs; replay accounting, not actual collection savings",
                "queries": copy.deepcopy(self.records)}


@dataclass
class _Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float) -> "_Ridge":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        z = (x - mean) / scale
        intercept = y.mean(axis=0)
        centered = y - intercept
        coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ centered)
        return cls(mean, scale, coef, intercept)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != len(self.mean):
            raise ValueError("Feature dimension differs from the fitted program")
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def state_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).tolist() for name in ("mean", "scale", "coef", "intercept")}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_Ridge":
        return cls(*(np.asarray(state[k], dtype=float) for k in ("mean", "scale", "coef", "intercept")))


def _metrics(y: np.ndarray, predictions: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    if y.ndim != 2 or y.shape != predictions.shape or y.shape[1] == 0:
        raise ValueError("Repair loss dimensions must match model predictions")
    chosen = np.argmin(predictions, axis=1)
    losses = y[np.arange(len(y)), chosen]
    oracle = np.min(y, axis=1)
    regret = losses - oracle
    return {"n_episodes": len(y), "mean_loss": float(np.mean(losses)),
            "mean_oracle_loss": float(np.mean(oracle)), "mean_regret": float(np.mean(regret)),
            "repair_accuracy": float(np.mean(chosen == np.argmin(y, axis=1))),
            "episode_ids": list(ids), "chosen_repairs": chosen.tolist(),
            "selected_losses": losses.tolist(), "regrets": regret.tolist()}


class GreedyProgramSynthesizer:
    """Greedy inherited DSL search; development data alone decides promotion.

    ``counterexample`` scouts features on discovery states with the largest
    current repair regret and ranks them by leave-one-out residual prediction.
    This reads and charges those observations. ``random`` and ``enumeration``
    vary candidate order under the identical cell budget and promotion rule.
    The candidate set and ridge alpha are fixed before observing development.
    No strategy uses test labels or test probes for selection.

    The supplied seed influences random ordering only. Call ``fit`` once, then
    ``evaluate`` on a frozen holdout. A second fit starts a new search; inherited
    generations live inside one fit and are recorded in ``history``.
    """

    def __init__(self, *, strategy: str = "counterexample", max_probes: int = 2,
                 search_budget_cells: int = 1000, seed: int = 0,
                 ridge_alpha: float = 1.0, min_improvement: float = 0.0,
                 counterexample_count: int = 6, max_generations: int | None = None) -> None:
        if strategy not in ("counterexample", "random", "enumeration"):
            raise ValueError("Unknown search strategy")
        if max_probes < 0 or search_budget_cells < 0:
            raise ValueError("Program size and query budget must be nonnegative")
        if ridge_alpha <= 0 or not np.isfinite(ridge_alpha):
            raise ValueError("ridge_alpha must be finite and positive")
        if min_improvement < 0 or not np.isfinite(min_improvement):
            raise ValueError("min_improvement must be finite and nonnegative")
        if counterexample_count < 2:
            raise ValueError("counterexample_count must be at least two")
        if max_generations is not None and max_generations < 0:
            raise ValueError("max_generations must be nonnegative")
        self.strategy = strategy
        self.max_probes = int(max_probes)
        self.search_budget_cells = int(search_budget_cells)
        self.seed = int(seed)
        self.ridge_alpha = float(ridge_alpha)
        self.min_improvement = float(min_improvement)
        self.counterexample_count = int(counterexample_count)
        self.max_generations = self.max_probes if max_generations is None else int(max_generations)
        self.selected_probes: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.search_report: dict[str, Any] = {}
        self._model: _Ridge | None = None

    def _counterexample_order(self, candidates: list[str], discovery: list[Episode],
                              dev_count: int, y: np.ndarray, prediction: np.ndarray,
                              ledger: _QueryLedger) -> tuple[list[str], list[dict[str, Any]]]:
        chosen = np.argmin(prediction, axis=1)
        regret = y[np.arange(len(y)), chosen] - y.min(axis=1)
        residual = y - prediction
        # Regret is primary; prediction error breaks zero-regret ties deterministically.
        priority = sorted(range(len(y)), key=lambda i: (-float(regret[i]),
                          -float(np.linalg.norm(residual[i])), str(discovery[i]["id"])))
        count = min(self.counterexample_count, len(discovery))
        indices = priority[:count]
        scout_rows = [discovery[i] for i in indices]
        target = residual[indices]
        # Reserve enough cells for at least one full train/development candidate.
        allowance = max(0, min(ledger.remaining // 4,
                              ledger.remaining - len(discovery) - dev_count))
        spent = 0
        scores: list[dict[str, Any]] = []
        for probe in candidates:
            missing = ledger.missing(scout_rows, [probe])
            if count < 3 or spent + missing > allowance:
                continue
            before = len(ledger.cache)
            features = np.stack([ledger.read(e, probe, "counterexample_scout") for e in scout_rows])
            spent += len(ledger.cache) - before
            loo_error = 0.0
            null_error = 0.0
            for holdout in range(count):
                mask = np.arange(count) != holdout
                model = _Ridge.fit(features[mask], target[mask], self.ridge_alpha)
                pred = model.predict(features[holdout:holdout + 1])[0]
                loo_error += float(np.sum((target[holdout] - pred) ** 2))
                null_error += float(np.sum((target[holdout] - target[mask].mean(axis=0)) ** 2))
            utility = (null_error - loo_error) / count
            scores.append({"probe_id": probe, "counterexample_utility": utility,
                           "episode_ids": [str(e["id"]) for e in scout_rows]})
        scores.sort(key=lambda score: (-score["counterexample_utility"], score["probe_id"]))
        ordered = [s["probe_id"] for s in scores]
        ordered.extend(p for p in candidates if p not in ordered)
        return ordered, scores

    def fit(self, episodes: Iterable[Episode], candidate_ids: Sequence[str] | None = None) -> "GreedyProgramSynthesizer":
        episodes = list(episodes)
        discovery = _split(episodes, "discovery")
        development = _split(episodes, "development")
        self._check_split_ids(discovery, development)
        y_train = _matrix(discovery, "repair_losses")
        y_dev = _matrix(development, "repair_losses")
        if y_train.shape[1] != y_dev.shape[1] or y_train.shape[1] == 0:
            raise ValueError("At least one repair with consistent dimensions is required")
        if candidate_ids is None:
            # The catalog is metadata, not the feature values themselves.
            common = set(discovery[0]["probe_features"].keys())
            for row in discovery[1:] + development:
                common.intersection_update(row["probe_features"].keys())
            candidates = sorted(common)
        else:
            candidates = sorted(set(candidate_ids))
        for probe in candidates:
            ProbeSpec.from_id(probe)
        ledger = _QueryLedger(self.search_budget_cells)
        rng = np.random.default_rng(self.seed)
        self.selected_probes = []
        self.history = []
        model = _Ridge.fit(_matrix(discovery, "log_features"), y_train, self.ridge_alpha)
        dev_prediction = model.predict(_matrix(development, "log_features"))
        current = _metrics(y_dev, dev_prediction, [str(e["id"]) for e in development])
        baseline = current["mean_regret"]
        self.history.append({"generation": 0, "selected_probes": [],
                             "development_regret": baseline, "promoted": False,
                             "revealed_cells": 0})

        for generation in range(1, min(self.max_probes, self.max_generations) + 1):
            remaining = [p for p in candidates if p not in self.selected_probes]
            if not remaining:
                break
            scouts: list[dict[str, Any]] = []
            if self.strategy == "random":
                rng.shuffle(remaining)
            elif self.strategy == "counterexample":
                x_current = ledger.features(discovery, self.selected_probes, "inherited_program")
                remaining, scouts = self._counterexample_order(
                    remaining, discovery, len(development), y_train, model.predict(x_current), ledger)
            best: tuple[float, str, _Ridge] | None = None
            trials: list[dict[str, Any]] = []
            for candidate in remaining:
                program = self.selected_probes + [candidate]
                needed = ledger.missing(discovery + development, program)
                if needed > ledger.remaining:
                    continue
                x_train = ledger.features(discovery, program, "candidate_fit")
                x_dev = ledger.features(development, program, "candidate_development")
                trial_model = _Ridge.fit(x_train, y_train, self.ridge_alpha)
                result = _metrics(y_dev, trial_model.predict(x_dev), [str(e["id"]) for e in development])
                trial_regret = result["mean_regret"]
                trials.append({"probe_id": candidate, "program": program,
                               "development_regret": trial_regret,
                               "revealed_cells": len(ledger.cache)})
                # Stable tie breaking is independent of evaluation ordering.
                if best is None or (trial_regret, candidate) < (best[0], best[1]):
                    best = (trial_regret, candidate, trial_model)
            promoted = best is not None and best[0] < current["mean_regret"] - self.min_improvement - 1e-12
            if promoted:
                assert best is not None
                self.selected_probes.append(best[1])
                model = best[2]
                current = {"mean_regret": best[0]}
            self.history.append({"generation": generation, "selected_probes": self.selected_probes.copy(),
                                 "development_regret": current["mean_regret"],
                                 "promoted": bool(promoted), "scouts": scouts, "trials": trials,
                                 "revealed_cells": len(ledger.cache)})
            if not promoted:
                break
        self._model = model
        self.search_report = ledger.summary()
        self.search_report.update({"strategy": self.strategy, "candidate_count": len(candidates),
                                   "baseline_development_regret": baseline,
                                   "selected_development_regret": current["mean_regret"],
                                   "selection_split": "development", "predictor_fit_split": "discovery",
                                   "test_used_for_selection": False,
                                   "synthesizer_update": False})
        return self

    @staticmethod
    def _check_split_ids(discovery: Sequence[Episode], development: Sequence[Episode]) -> None:
        if {str(e["id"]) for e in discovery} & {str(e["id"]) for e in development}:
            raise ValueError("Discovery and development episode IDs must be disjoint")

    def fit_fixed(self, episodes: Iterable[Episode], probe_ids: Sequence[str]) -> "GreedyProgramSynthesizer":
        """Fit a preregistered expert/no-probe program without selecting on dev.

        The caller fixes probe IDs before inspecting development or test results.
        Only discovery observations are queried and charged in this comparator.
        """
        discovery = _split(list(episodes), "discovery")
        probes = list(probe_ids)
        if len(set(probes)) != len(probes) or len(probes) > self.max_probes:
            raise ValueError("Fixed probes must be unique and within max_probes")
        for probe in probes:
            ProbeSpec.from_id(probe)
        ledger = _QueryLedger(self.search_budget_cells)
        x = ledger.features(discovery, probes, "fixed_program_fit")
        y = _matrix(discovery, "repair_losses")
        if y.shape[1] == 0:
            raise ValueError("At least one repair is required")
        self._model = _Ridge.fit(x, y, self.ridge_alpha)
        self.selected_probes = probes
        self.history = [{"generation": 0, "selected_probes": probes.copy(), "promoted": False}]
        self.search_report = ledger.summary()
        self.search_report.update({"strategy": "fixed", "selection_split": None,
                                   "predictor_fit_split": "discovery", "test_used_for_selection": False,
                                   "synthesizer_update": False})
        return self

    def predict(self, episodes: Iterable[Episode], *, split: str = "test") -> dict[str, Any]:
        """Read only selected features; do not access repair labels."""
        if self._model is None:
            raise RuntimeError("Fit or load a program before predicting")
        rows = _split(episodes, split)
        columns = [_matrix(rows, "log_features")]
        total_cost = 0.0
        scalar_count = 0
        for probe in self.selected_probes:
            vectors = [_vector(e["probe_features"][probe], f"probe {probe}") for e in rows]
            columns.append(np.stack(vectors))
            scalar_count += sum(v.size for v in vectors)
            costs = [float(e["probe_costs"][probe]) for e in rows]
            if any(not np.isfinite(c) or c < 0 for c in costs):
                raise ValueError("Probe costs must be finite and nonnegative")
            total_cost += sum(costs)
        predictions = self._model.predict(np.concatenate(columns, axis=1))
        return {"episode_ids": [str(e["id"]) for e in rows],
                "predicted_repair_losses": predictions.tolist(),
                "chosen_repairs": np.argmin(predictions, axis=1).tolist(),
                "selected_probes": self.selected_probes.copy(),
                "evaluation_revealed_cells": len(rows) * len(self.selected_probes),
                "evaluation_revealed_scalars": scalar_count,
                "evaluation_probe_cost": total_cost,
                "mean_evaluation_probe_cost": total_cost / len(rows)}

    def evaluate(self, episodes: Iterable[Episode], *, split: str = "test") -> dict[str, Any]:
        """Score a frozen program; evaluation never updates search or parameters."""
        episodes = list(episodes)
        prediction = self.predict(episodes, split=split)
        rows = _split(episodes, split)
        metrics = _metrics(_matrix(rows, "repair_losses"),
                           np.asarray(prediction.pop("predicted_repair_losses")), prediction["episode_ids"])
        return {**prediction, **metrics, "split": split, "benchmark_type": "deployable_program",
                "search_revealed_cells": self.search_report["revealed_cells"],
                "search_replay_probe_cost": self.search_report["replay_probe_cost"]}

    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable frozen model, ancestry, and search ledger (no labels)."""
        if self._model is None:
            raise RuntimeError("Cannot serialize an unfitted program")
        params = {k: getattr(self, k) for k in ("strategy", "max_probes", "search_budget_cells",
                  "seed", "ridge_alpha", "min_improvement", "counterexample_count", "max_generations")}
        return {"format_version": 1, "parameters": params, "selected_probes": self.selected_probes.copy(),
                "model": self._model.state_dict(), "history": copy.deepcopy(self.history),
                "search_report": copy.deepcopy(self.search_report)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "GreedyProgramSynthesizer":
        if state.get("format_version") != 1:
            raise ValueError("Unsupported synthesis state version")
        instance = cls(**state["parameters"])
        instance.selected_probes = list(state["selected_probes"])
        instance._model = _Ridge.from_state_dict(state["model"])
        instance.history = copy.deepcopy(state["history"])
        instance.search_report = copy.deepcopy(state["search_report"])
        return instance


def evaluate_no_probe(episodes: Iterable[Episode], *, split: str = "test",
                      ridge_alpha: float = 1.0) -> dict[str, Any]:
    """Fit only free logs on discovery; this is not an oracle repair selector."""
    rows = list(episodes)
    model = GreedyProgramSynthesizer(max_probes=0, search_budget_cells=0, ridge_alpha=ridge_alpha)
    model.fit_fixed(rows, [])
    return model.evaluate(rows, split=split)


def evaluate_fixed_probes(episodes: Iterable[Episode], selected_probe_ids: Sequence[str], *,
                          split: str = "test", ridge_alpha: float = 1.0,
                          search_budget_cells: int | None = None) -> dict[str, Any]:
    """Evaluate a caller-preregistered expert program, charging its discovery fit.

    If no budget is supplied, allocate exactly one observation per discovery
    episode and selected probe. This does not select a program or tune alpha.
    """
    rows = list(episodes)
    probes = list(selected_probe_ids)
    if search_budget_cells is None:
        search_budget_cells = sum(e["split"] == "discovery" for e in rows) * len(probes)
    model = GreedyProgramSynthesizer(max_probes=len(probes),
                                    search_budget_cells=search_budget_cells, ridge_alpha=ridge_alpha)
    model.fit_fixed(rows, probes)
    return model.evaluate(rows, split=split)


def evaluate_oracle(episodes: Iterable[Episode], *, split: str = "test") -> dict[str, Any]:
    """Hindsight lower bound that reads all repair outcomes; not deployable.

    No zero-cost claim is made: obtaining the labels required the producer to
    execute all repair continuations. Their actual cost is outside this table.
    """
    rows = _split(episodes, split)
    y = _matrix(rows, "repair_losses")
    metrics = _metrics(y, y, [str(e["id"]) for e in rows])
    return {**metrics, "split": split, "benchmark_type": "hindsight_oracle_not_deployable",
            "uses_heldout_repair_labels_for_decisions": True,
            "evaluation_probe_cost": None,
            "cost_interpretation": "all repair labels required; collection cost is external, not zero"}
