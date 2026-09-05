# Development mechanism audit: how long does a repair remain predictable?

Date: 2026-09-05. This audit is proposed after the completed v2 NOGO and a recorded post-run predictor-capacity sweep. It is development exploration, not a new confirmation attempt or a replacement of the failed 64-update criterion.

## Motivation

On the existing development states, all eight always-paid heads lose to the 64-update free baseline. Lower-dimensional heads and a five-value ridge regularization sweep do not beat the strongest free head. The small one-update probe may measure an effect that disappears or reverses before the endpoint. The next experiment measures that timescale directly before training new model seeds or proposing a third method.

## Frozen collection

Restore all seven existing v2 seed checkpoints and all eight original declared conditions. Use identical model, optimizer, RNG, document ordering, current gradient, actions and known training batches. Never load the text-test stream. Read only the same 4,096-target-token dev window beginning at token offset 8,192.

For each action, collect dev loss at total continuation updates 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63 and 64. Hash inputs/code before replay, retain actual gradient, optimizer and evaluation counts, and require the re-collected 63/64 losses to agree with the original evidence within 1e-6 absolute error. Any mismatch or nonfinite outcome makes the replay invalid; preserve the failed run instead of overwriting it.

## Descriptive analysis

At endpoint horizons 4, 8, 16, 32 and 64, reconstruct a paired fixed-budget table: no acquisition leaves H updates, and a two-forward acquisition leaves H-1 updates. The common first gradient/update is included in both budgets. Keep a fixed 32-unit final-evaluation reserve, as in v2. Do not pretend a single physical table replay is a deployed-policy cost saving.

Report all horizon results, including logs-only at both horizons, a logs-only joint action/horizon controller, fixed expert and conditional expert, and counterexample/random/enumerated search. Include ridge alpha 1 and 10 for free and paid models equally; alpha 10 is included because the previous development sweep improved the free baseline. Report per-seed results, action-oracle headroom and acquisition rates. Correlated conditions are not independent replicates.

There is no promotion-to-confirmation gate in this descriptive audit, and no best horizon is automatically declared successful. If a shorter recovery timescale appears useful, the next development protocol must explain the resulting operational task, costs and comparator fairness explicitly. Confirmation still requires a subsequently frozen method and entirely fresh seeds/windows under GOAL.md.
