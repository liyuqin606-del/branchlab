# Repair-lifetime development findings

The replay is valid: all 336 original endpoint values at 63/64 updates are bitwise identical. All 56 states were replayed at 12 times, with no missing or nonfinite outcomes. The separate audit consumed 10,640 gradient batches and 32,256 dev forwards, or 64,176 analytic forward-batch units (2,723,840 trained tokens, 409.22 seconds local CPU). It did not retrain the baseline or access test text.

Shortening the horizon does not rescue the proposed search mechanism:

| Horizon | Descriptive finding |
| ---: | --- |
| 4 | CE improves over its logs head by 0.000397, but constant momentum reset is better by 0.002030. |
| 8 | Alpha 10 CE improves over logs by 0.000818, but is identical to the fixed conditional expert, enumeration and two random runs. |
| 16 | CE retains the no-probe policy. |
| 32 | The small alpha 10 gain of0.000156 is shared by the fixed conditional expert and all search controls; constant LR-half is stronger. |
| 64 | CE retains the no-probe policy; a free joint action/horizon controller is slightly stronger. |

These are comparisons within an explicitly exploratory horizon/regularization audit, not independent results. They do not justify a new confirmation run or a positive RSI claim. All 100 method/configuration records and 40 constant-control records remain available, including every seed and decision.

The next development screen changes the physical observation: retain a four-step branch prefix and read four training calibration batches. Its losing prefix and forwards will cost 17 extra units, leaving 58 retained updates. A prefix-log-only controller gets 61 updates; free origin controllers get 64 or may stop early. This expensive observation will be rejected if even its conditional oracle lacks 0.002 nats/token headroom, or if the fixed expert cannot clear the frozen controls. See the separately frozen committed-prefix protocol; none of the failed lifetime criteria is replaced.
