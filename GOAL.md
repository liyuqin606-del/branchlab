# Active research goal: make the diagnostic useful

Started 2026-09-05 after the v0.1.0 NOGO release. The user requested a persistent goal and continued iteration until the method works. v0.1.0 and its evidence remain unchanged.

Completion requires a deployable training diagnostic that improves independently evaluated model loss after paying its diagnostic cost, compared with logs-only selection, a prespecified expert, and random/enumerated search with matched search resources. A positive engineering result does not establish full RSI.

## Development and confirmation

All v0.1 seeds 0–7 and both of its text windows are now development data. Their outcomes can diagnose failures and guide changes, but cannot confirm a new claim. Every new development run is tagged as development before it starts. Failed ideas remain in the iteration ledger; they are never relabeled as confirmatory.

Before a confirmation run, freeze the chosen method, baselines, data windows, model configurations, held-out seed list, budget accounting, practical effect threshold, statistic, failure handling and exact promotion criterion. Confirmation uses new seed groups and document windows that have not selected the method. Evaluate seeds as independent units; state variants within a seed are correlated. Use at least 4,096 target tokens per final text-loss evaluation, with identical windows across paired methods.

Repeated confirmations must not turn 'keep trying' into a false-positive guarantee. Confirmation attempt k allocates at most alpha_k=0.05/[k(k+1)] across all required primary comparisons, with a prespecified correction. The sum over attempts is at most 0.05. A confirmation can fail; its data then joins development, while later confirmation uses a fresh reserved cohort and the next alpha allocation. The method must also replicate on a second fresh cohort before the goal is marked complete. No claimed superiority may rely only on a replay oracle or on a software test.

## Iteration 1: economic value of diagnostics

The first change will address v0.1's objective mismatch. It selected programs using fixed-24-step dev regret, then evaluated their cost at a different continuation horizon. It always executed a selected probe. Investigate a decision to skip a probe and optimize net loss at the actual remaining training horizon. Improve evaluation coverage and the diversity of legitimate training states only under an explicitly changed development protocol; never manufacture a win by hiding available logs or showing baselines less history.

The current code branch is `codex/value-aware-diagnostics`. No paid compute is authorized. Use the existing local machine and bounded development pilots. A successful version will receive an audited public GitHub release; intermediate findings and failures will be retained.
