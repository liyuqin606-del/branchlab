# v2 post-run synthesis diagnosis

Development only: this analysis reads the 56 episodes with discovery seeds 8–11 and development seeds 12–14. It does not access confirmation text, states, or seeds. All post-hoc settings are retained; no result is promoted to confirmation.

## Why the learner returned no probe

- The promotion guard worked: every evaluated conditional policy either matched or worsened development loss. Individual gates did acquire probes; the final promoted policy correctly fell back to logs-only.
- At alpha 1, full logs contain 92 features but there are only 32 discovery episodes from four independent seeds. The gain gate adds a margin feature. Its predictions extrapolate strongly: momentum_zero:1 predicts mean gain +0.021163 while its realized mean gain is -0.001987 and buys on 24/24 development states. lr_half:1 predicts -0.012553 while realized gain is -0.002331 and never buys.
- Heads also lack incremental value. All eight always-paid heads lose to full-budget logs. Relative to a matched 63-step free head, seven are unchanged or worse; momentum_zero:1 improves only 0.000022884.

## Remaining headroom

| Quantity | Development mean |
| --- | ---: |
| Logs at 64 updates | 3.418438588 |
| Free logs at 63 updates | 3.420448684 |
| Clairvoyant action oracle at 64 updates | 3.411281755 |
| Clairvoyant action oracle after paying for a probe, 63 updates | 3.411067046 |
| Clairvoyant choice to skip or use a paid action oracle | 3.410468344 |
| Best clairvoyant gate around an actually fitted probe head | 3.416534238 |

Logs choose a non-oracle 64-step action in 15/24 states, but mean regret is just 0.007156833. The perfect acquisition gate for the existing fitted heads has at most 0.001904350 mean improvement. Hindsight choices are ceilings, not deployable results.

## Requested capacity sweep

Evaluated alpha {0.01, 0.1, 1, 10, 100} across full92, first14aggregate, first14+last9candidate, and last9candidate; free and paid heads share each setting. Every gate uses whole-discovery-seed OOF labels. All 160 rows are in postrun_capacity_sweep.json.

| Free subset | Alpha | Free loss | Best paid head | Best conditional gate | Same-horizon signal in OOF and dev / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full92 | 0.01 | 3.423939334 | 3.421426545 | 3.421543897 | 1 |
| full92 | 0.1 | 3.423952984 | 3.421078475 | 3.420785785 | 0 |
| full92 | 1 | 3.418438588 | 3.420425800 | 3.418438588 | 0 |
| full92 | 10 | 3.416384583 | 3.418544463 | 3.416574147 | 0 |
| full92 | 100 | 3.416642992 | 3.418704653 | 3.416642992 | 0 |
| first14aggregate | 0.01 | 3.420568543 | 3.417225251 | 3.418643677 | 1 |
| first14aggregate | 0.1 | 3.420199148 | 3.418604144 | 3.420721852 | 2 |
| first14aggregate | 1 | 3.420611699 | 3.418439211 | 3.419740453 | 0 |
| first14aggregate | 10 | 3.419035982 | 3.418438553 | 3.418617161 | 0 |
| first14aggregate | 100 | 3.421013990 | 3.421034201 | 3.421341934 | 0 |
| first14_plus_last9candidate | 0.01 | 3.418414279 | 3.421200677 | 3.419731799 | 0 |
| first14_plus_last9candidate | 0.1 | 3.419309358 | 3.423148361 | 3.419962380 | 0 |
| first14_plus_last9candidate | 1 | 3.418537314 | 3.420319992 | 3.416608971 | 1 |
| first14_plus_last9candidate | 10 | 3.418578325 | 3.418206864 | 3.417954961 | 0 |
| first14_plus_last9candidate | 100 | 3.419509462 | 3.419732367 | 3.419870578 | 0 |
| last9candidate | 0.01 | 3.419335859 | 3.418249241 | 3.418456175 | 0 |
| last9candidate | 0.1 | 3.419335859 | 3.418417181 | 3.419233383 | 1 |
| last9candidate | 1 | 3.419335859 | 3.418417181 | 3.418061752 | 0 |
| last9candidate | 10 | 3.419012435 | 3.418417181 | 3.418222129 | 0 |
| last9candidate | 100 | 3.418683742 | 3.419086828 | 3.418924124 | 0 |

The best free model across this post-hoc grid is full92/alpha10 at 3.416384583. The best conditional probe policy is 3.416574147 and the best always-paid head is 3.417225251. None surpasses the strongest free development reference. Of 160 probe/settings pairs, 52 improve net loss in discovery OOF and development, but only six still improve against a matched-horizon free model in both. Most apparent benefits are repair-duration/model-fitting changes rather than added probe information.

The largest within-setting gate improvement, 0.003167199, occurs against the weak full92/alpha0.1 free model (3.423952984). It is driven by seed13; seeds12 and14 get worse. It must not be framed as a general diagnostic improvement.

## One bounded next algorithm

Use a shared-head residual-value policy, with capacity selected by discovery-seed CV. Keep all free logs in the strongly regularized free predictors; do not delete available information from the baseline. For a paid head, reuse the matched-horizon free predictor and fit only a scalar-probe residual correction to relative action outcomes, using nested whole-seed cross-fitting for residual and gain labels. Restrict the acquisition gate to two interpretable free predictions: current action margin and predicted free63-versus-free64 loss gap. This reduces the new gate from 93 coefficients to a small calibrated decision without hiding logs from the free baseline. Random/enumeration and experts must receive the same predictor, correction, gate family, and query budgets.

Merely centering action losses or renaming them pairwise advantages is not a new algorithm: with the same linear ridge features/penalty, subtracting a common per-state target is a linear output transformation and does not change the action argmin. The substantive change above is shared/shrunk residual capacity and a smaller gate, not that reparameterization.

Before another confirmation cohort, inspect the lifetime of repair effects at shorter horizons and increase independent discovery seeds if the signal exists but varies by seed. Adding more correlated conditions to the existing four training seeds will not resolve the main effective-sample limitation. The current tiny advantage ceilings do not justify claiming that parameter tuning alone has made the method work.
