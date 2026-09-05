# Development repair-lifetime audit

Development repair-timescale audit; every requested setting retained; no winner or confirmation gate selected

All losses are macro-averaged over development training seeds. The same seeds recur across rows and horizons. No confirmation gate or winner is selected.

| Horizon | Alpha | Method | Search seed | Dev loss | Probe acquisitions | Mean probe cost | Search cells |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | logs_only | - | 3.507430660 | 0 | 0.000 | 0 |
| 4 | 1 | logs_only_short | - | 3.507288314 | 0 | 0.000 | 0 |
| 4 | 1 | fixed_expert | - | 3.508288908 | 24 | 2.000 | 32 |
| 4 | 1 | fixed_conditional_expert | - | 3.507623712 | 18 | 1.500 | 32 |
| 4 | 1 | logs_joint_action_horizon | - | 3.507011152 | 0 | 0.000 | 0 |
| 4 | 1 | counterexample | 0 | 3.507033659 | 23 | 1.917 | 165 |
| 4 | 1 | random | 0 | 3.507033659 | 23 | 1.917 | 163 |
| 4 | 1 | random | 1 | 3.507033659 | 23 | 1.917 | 159 |
| 4 | 1 | random | 2 | 3.507033659 | 23 | 1.917 | 163 |
| 4 | 1 | enumeration | 0 | 3.507182621 | 22 | 1.833 | 158 |
| 4 | 10 | logs_only | - | 3.507430660 | 0 | 0.000 | 0 |
| 4 | 10 | logs_only_short | - | 3.508288908 | 0 | 0.000 | 0 |
| 4 | 10 | fixed_expert | - | 3.508288908 | 24 | 2.000 | 32 |
| 4 | 10 | fixed_conditional_expert | - | 3.508034253 | 23 | 1.917 | 32 |
| 4 | 10 | logs_joint_action_horizon | - | 3.508034253 | 0 | 0.000 | 0 |
| 4 | 10 | counterexample | 0 | 3.507033659 | 23 | 1.917 | 166 |
| 4 | 10 | random | 0 | 3.507430660 | 0 | 0.000 | 165 |
| 4 | 10 | random | 1 | 3.507033659 | 23 | 1.917 | 165 |
| 4 | 10 | random | 2 | 3.507033659 | 23 | 1.917 | 165 |
| 4 | 10 | enumeration | 0 | 3.507430660 | 0 | 0.000 | 165 |
| 8 | 1 | logs_only | - | 3.496584961 | 0 | 0.000 | 0 |
| 8 | 1 | logs_only_short | - | 3.499763881 | 0 | 0.000 | 0 |
| 8 | 1 | fixed_expert | - | 3.500476246 | 24 | 2.000 | 32 |
| 8 | 1 | fixed_conditional_expert | - | 3.498224742 | 17 | 1.417 | 32 |
| 8 | 1 | logs_joint_action_horizon | - | 3.497654518 | 0 | 0.000 | 0 |
| 8 | 1 | counterexample | 0 | 3.496584961 | 0 | 0.000 | 170 |
| 8 | 1 | random | 0 | 3.496432887 | 11 | 0.917 | 179 |
| 8 | 1 | random | 1 | 3.496584961 | 0 | 0.000 | 171 |
| 8 | 1 | random | 2 | 3.496584961 | 0 | 0.000 | 166 |
| 8 | 1 | enumeration | 0 | 3.496584961 | 0 | 0.000 | 180 |
| 8 | 10 | logs_only | - | 3.496620923 | 0 | 0.000 | 0 |
| 8 | 10 | logs_only_short | - | 3.499529187 | 0 | 0.000 | 0 |
| 8 | 10 | fixed_expert | - | 3.499557817 | 24 | 2.000 | 32 |
| 8 | 10 | fixed_conditional_expert | - | 3.495802432 | 4 | 0.333 | 32 |
| 8 | 10 | logs_joint_action_horizon | - | 3.497690480 | 0 | 0.000 | 0 |
| 8 | 10 | counterexample | 0 | 3.495802432 | 4 | 0.333 | 159 |
| 8 | 10 | random | 0 | 3.495802432 | 4 | 0.333 | 174 |
| 8 | 10 | random | 1 | 3.495802432 | 4 | 0.333 | 177 |
| 8 | 10 | random | 2 | 3.496171261 | 4 | 0.333 | 171 |
| 8 | 10 | enumeration | 0 | 3.495802432 | 4 | 0.333 | 177 |
| 16 | 1 | logs_only | - | 3.484909005 | 0 | 0.000 | 0 |
| 16 | 1 | logs_only_short | - | 3.489710156 | 0 | 0.000 | 0 |
| 16 | 1 | fixed_expert | - | 3.491112882 | 24 | 2.000 | 32 |
| 16 | 1 | fixed_conditional_expert | - | 3.486509574 | 6 | 0.500 | 32 |
| 16 | 1 | logs_joint_action_horizon | - | 3.489690047 | 0 | 0.000 | 0 |
| 16 | 1 | counterexample | 0 | 3.484909005 | 0 | 0.000 | 174 |
| 16 | 1 | random | 0 | 3.484909005 | 0 | 0.000 | 154 |
| 16 | 1 | random | 1 | 3.484909005 | 0 | 0.000 | 156 |
| 16 | 1 | random | 2 | 3.484909005 | 0 | 0.000 | 166 |
| 16 | 1 | enumeration | 0 | 3.484909005 | 0 | 0.000 | 156 |
| 16 | 10 | logs_only | - | 3.482521018 | 0 | 0.000 | 0 |
| 16 | 10 | logs_only_short | - | 3.490984081 | 0 | 0.000 | 0 |
| 16 | 10 | fixed_expert | - | 3.490984081 | 24 | 2.000 | 32 |
| 16 | 10 | fixed_conditional_expert | - | 3.488593837 | 17 | 1.417 | 32 |
| 16 | 10 | logs_joint_action_horizon | - | 3.489991912 | 0 | 0.000 | 0 |
| 16 | 10 | counterexample | 0 | 3.482521018 | 0 | 0.000 | 171 |
| 16 | 10 | random | 0 | 3.482521018 | 0 | 0.000 | 177 |
| 16 | 10 | random | 1 | 3.482521018 | 0 | 0.000 | 179 |
| 16 | 10 | random | 2 | 3.482521018 | 0 | 0.000 | 180 |
| 16 | 10 | enumeration | 0 | 3.482521018 | 0 | 0.000 | 178 |
| 32 | 1 | logs_only | - | 3.466151061 | 0 | 0.000 | 0 |
| 32 | 1 | logs_only_short | - | 3.467694871 | 0 | 0.000 | 0 |
| 32 | 1 | fixed_expert | - | 3.467477348 | 24 | 2.000 | 32 |
| 32 | 1 | fixed_conditional_expert | - | 3.466151061 | 0 | 0.000 | 32 |
| 32 | 1 | logs_joint_action_horizon | - | 3.466151061 | 0 | 0.000 | 0 |
| 32 | 1 | counterexample | 0 | 3.466151061 | 0 | 0.000 | 172 |
| 32 | 1 | random | 0 | 3.466151061 | 0 | 0.000 | 160 |
| 32 | 1 | random | 1 | 3.466151061 | 0 | 0.000 | 160 |
| 32 | 1 | random | 2 | 3.466151061 | 0 | 0.000 | 160 |
| 32 | 1 | enumeration | 0 | 3.466151061 | 0 | 0.000 | 160 |
| 32 | 10 | logs_only | - | 3.467231344 | 0 | 0.000 | 0 |
| 32 | 10 | logs_only_short | - | 3.469550561 | 0 | 0.000 | 0 |
| 32 | 10 | fixed_expert | - | 3.468787630 | 24 | 2.000 | 32 |
| 32 | 10 | fixed_conditional_expert | - | 3.467075775 | 1 | 0.083 | 32 |
| 32 | 10 | logs_joint_action_horizon | - | 3.467231344 | 0 | 0.000 | 0 |
| 32 | 10 | counterexample | 0 | 3.467075775 | 1 | 0.083 | 177 |
| 32 | 10 | random | 0 | 3.467075775 | 1 | 0.083 | 165 |
| 32 | 10 | random | 1 | 3.467075775 | 1 | 0.083 | 165 |
| 32 | 10 | random | 2 | 3.467075775 | 1 | 0.083 | 165 |
| 32 | 10 | enumeration | 0 | 3.467075775 | 1 | 0.083 | 165 |
| 64 | 1 | logs_only | - | 3.418438588 | 0 | 0.000 | 0 |
| 64 | 1 | logs_only_short | - | 3.420448684 | 0 | 0.000 | 0 |
| 64 | 1 | fixed_expert | - | 3.420769113 | 24 | 2.000 | 32 |
| 64 | 1 | fixed_conditional_expert | - | 3.418438588 | 0 | 0.000 | 32 |
| 64 | 1 | logs_joint_action_horizon | - | 3.419715462 | 0 | 0.000 | 0 |
| 64 | 1 | counterexample | 0 | 3.418438588 | 0 | 0.000 | 174 |
| 64 | 1 | random | 0 | 3.418438588 | 0 | 0.000 | 163 |
| 64 | 1 | random | 1 | 3.418438588 | 0 | 0.000 | 152 |
| 64 | 1 | random | 2 | 3.418438588 | 0 | 0.000 | 176 |
| 64 | 1 | enumeration | 0 | 3.418438588 | 0 | 0.000 | 169 |
| 64 | 10 | logs_only | - | 3.416384583 | 0 | 0.000 | 0 |
| 64 | 10 | logs_only_short | - | 3.419921625 | 0 | 0.000 | 0 |
| 64 | 10 | fixed_expert | - | 3.419477905 | 24 | 2.000 | 32 |
| 64 | 10 | fixed_conditional_expert | - | 3.416754760 | 5 | 0.417 | 32 |
| 64 | 10 | logs_joint_action_horizon | - | 3.416283396 | 0 | 0.000 | 0 |
| 64 | 10 | counterexample | 0 | 3.416384583 | 0 | 0.000 | 174 |
| 64 | 10 | random | 0 | 3.416384583 | 0 | 0.000 | 156 |
| 64 | 10 | random | 1 | 3.416384583 | 0 | 0.000 | 150 |
| 64 | 10 | random | 2 | 3.416384583 | 0 | 0.000 | 154 |
| 64 | 10 | enumeration | 0 | 3.416384583 | 0 | 0.000 | 152 |

The h-1 logs control uses no observation and leaves three proxy units unspent; a paid two-unit probe also routes to h-1 and leaves one unit unspent. Joint logs can select either horizon for free. This separates short-training effects from paid information.

Per-seed differences, every decision, and nondeployable oracle ceilings are retained in summary.json. Query ledgers and fitted model states are in programs/. Physical curve-collection cost remains separate from this analysis replay.
