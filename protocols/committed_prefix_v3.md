# Development screen: committed four-step prefix and broader calibration

2026-09-05. This is a new observation screen on the same old development data. V 2 and its lifetime audit remain negative/inconclusive; no confirmation is attempted. The lifetime audit found no advantage over all simple controls at any examined horizon. One-update scalar contrasts are therefore replaced here by a fixed multi-update, multi-batch observation, not by another horizon or ridge sweep.

## Fixed design

Use every original v2 state (seeds 8–14, eight conditions each), its checkpoint, document order, and the same dev evaluation window of 4096 target tokens at offset 8192. No test text. The free proposer is the full 92-feature ridge model with alpha 10 fitted exclusively on discovery seeds 8–11 against original 64-update labels. Freeze its parameters and top-two action proposals before collecting new observations; stable ties follow keep,lr_half,momentum_zero. This alpha is a development-derived strong baseline, not an independent finding.

A deployed diagnosis materializes the proposed pair from one shared current gradient, then trains each for three more ordinary updates. Evaluate both saved four-update states on four known training calibration batches at origin-relative positions 65,81,97,113. Position 1 is the first committed update. Verify nonoverlapping target indices and no wrapping; position 65 may share a boundary context token with position 64. Retain each raw calibration loss. The fixed expert chooses the pair member with lower mean calibration loss; ties retain the free incumbent. No threshold, learned acquisition gate or post-hoc batch subset is added.

Restore the selected complete four-update checkpoint and continue it; never repeat the selected prefix, action or common gradient. Record signatures of weights, optimizer state, RNG and stream before/after restoring the prefix. Prefix training metrics and inexpensive parameter-level gradient/moment statistics are open to all prefix controllers; the paid calibration forwards are the only additional observations withheld from no-calibration controllers.

## Costs and data collection

Keep the 224-unit total cap with 32 final-evaluation units reserved and 3 proxy units per gradient batch. A two-candidate four-step prefix adds 9 units for the losing branch after the shared first gradient. Eight calibration forwards add 8. Total excess 17 leaves 58 retained updates and one unspent unit. A prefix-only controller pays 9 and retains 61 updates. Origin-only controllers retain 64 or may stop earlier at 58/61 for free. These are analytic forward-batch equivalents, not measured FLOPs or wall-time savings.

For table construction only, replay all three actions to 64, save the four-step observations, and evaluate dev loss at 58,61,64. The logical policy reads only its frozen pair. The full physical collection is 10640 gradient batches,168 candidate optimizer steps,672 calibration forwards,8064 dev forwards and 40656 proxy units; no baseline retraining. Require every 64 endpoint to match the original within 1e-6. Preserve all input/code/protocol hashes, attempted/completed counts and failed runs. A mismatch or incomplete run is invalid, not a scientific NOGO. All calibrations use training data only.

## Comparators and descriptive checks

Report every constant repair at 58/61/64; origin-log ridge controllers (alpha 10 and 100) at 64 and joint action/stopping controllers over all three horizons. All fit discovery only. The frozen proposer remains alpha 10 regardless of these outcomes.

Compare prefix-log ridge heads at 61 and 58, each restricted to the same frozen action pair. Expose origin 92 features, pair identity, four steps of both candidates' training loss/gradient norms, and each candidate's step 4 parameter gradient norm, first-moment norm and alignment. Report both compact (without the parameter detail) and full feature forms, at alpha 10 and 100. These fixed settings strengthen controls; no free evidence is hidden from them. Also include a direct prefix-training-loss expert choosing the smaller mean of the paired steps 2–4 at 61 and 58.

The paid fixed expert has no predictor fit and reads only the proposed pair's four calibration scores. Describe its pair-ranking agreement/regret at 58 and 64, macro-averaged by training seed. Do not confuse 64-step predictive quality with its actual 58-step budget result.

## Frozen development decisions

First require source/state/endpoint validity. Then calculate the clairvoyant skip-or-acquire upper bound using the frozen proposer's chosen 64-step action as fallback and the better of its pair's 58-step endpoints after acquisition. If mean development improvement is less than 0.002 nats/token, reject this fixed observation primitive before building a policy search. This is an oracle ceiling, never a deployable result.

If that ceiling permits progress, the fixed calibration expert must improve development loss by at least 0.002 over every listed origin-log joint controller and prefix-log 61 controller, with strictly positive paired improvement on each development seed for each comparison. It must also beat every prefix-log 58 controller in aggregate. Otherwise this fixed observation-and-rule screen is NOGO. Passing would only support another development step; it cannot satisfy the active goal's learned-method, search-baseline, independent-confirmation, replication or RSI requirements.

Do not rescue a failure by selecting a condition, horizon, calibration subset or new threshold after seeing these results. If this fixed primitive fails, pivot the operational design—for example, synthesize telemetry rules from already committed training steps—rather than continue tuning this paid branch observation.
