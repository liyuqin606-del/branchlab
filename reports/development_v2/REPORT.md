# Development iteration 2 — NOGO

This is method development, not independent confirmation. No test text was loaded. The run evaluated seven new development seeds (8–14), eight conditions per seed, and 4,096 fixed dev target tokens per final loss.

| Policy | Mean development loss | Acquisitions / 24 |
| --- | ---: | ---: |
| Logs only, 64 updates | 3.418439 | 0 |
| Logs only, 63 updates | 3.420449 | 0 |
| Fixed next-batch expert | 3.420769 | 24 |
| Fixed conditional expert | 3.418439 | 0 |
| Logs joint action/horizon | 3.419715 | 0 |
| Counterexample / random (3 seeds) / enumeration | 3.418439 | 0 |

All search strategies retained the no-probe policy. This removes unnecessary diagnostic spending but establishes no improvement over logs-only selection. The new 92-dimensional free log vector includes exact first-order candidate update summaries, exposed to every policy. Paid observations compare alternative and reference weights on a future known training batch after one shared-gradient update. Gates are cross-fitted by whole discovery seed. Search and deployment costs remain separate.

The physical run used 11,760 gradient batches, 672 probe forward batches and 5,376 dev evaluation forward batches (3,010,560 trained tokens; 41,328 analytic forward-batch equivalents). Collection took 282.90 seconds on the local CPU with four Torch threads, with no recorded failures. Candidate optimizer arithmetic is recorded separately. These figures are physical table-construction costs; no net wall-clock training speedup is claimed.

`summary.json` contains all decisions and seed-level outcomes; `episodes.json`, `curves.json`, and `programs/` permit numerical replay. `freeze.json` records pre-collection code/config/input hashes, and `validation.json` verifies source identity. Full training histories and checkpoints remain under the local `artifacts/development_v2/` directory. No new public model release is claimed by this report.

The confirmation ledger is still empty. All outcomes here may guide the next development design and cannot be reused as confirmation of it.
