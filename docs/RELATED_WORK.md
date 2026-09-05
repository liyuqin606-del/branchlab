# Scope, sources, and related work

BranchLab is an independent educational and research prototype. It implements a
small decoder-only language model and studies diagnostic-program selection using
local, restorable training branches. It is not a Stanford course submission, an
official Marin component, or a reproduction of a frontier model's training run.

This document carries forward the sources checked during the project's
2026-09-05 design work. It is a targeted overlap discussion, not an exhaustive
novelty search, an independent reproduction of the cited work, or a claim that
each live URL will retain the same contents. The original design and source
audit remain available in [the design note](../ideas/innovation_v2.md) and
[the data audit](../reports/source_audit.md).

## What the Stanford and Marin resources contribute

| Resource | Actual role in BranchLab | Claim that this does not support |
| --- | --- | --- |
| [Stanford CS336](https://cs336.stanford.edu/) and [Assignment 1](https://github.com/stanford-cs336/assignment1-basics) | Engineering reference for tokenization, a decoder-only Transformer, optimization, recovery, and measurement | Completion of official course assessment, Stanford affiliation, or an implementation of every course assignment |
| [Marin](https://marin.community/) and its [32B retrospective](https://github.com/marin-community/marin/blob/main/docs/reports/marin-32b-retro.md) | Public examples that motivate inspecting numerical behavior, optimizer state, recovery, and data order | Reproduction of the same failure mechanism or recovery benefit in a model of Marin's scale |
| [Delphi](https://openathena.ai/blog/delphi/) and [the published tables](https://huggingface.co/datasets/marin-community/delphi-blog-data) | Auditable configuration and endpoint data for descriptive analysis | A complete stream of interventions, counterfactual repair outcomes, or exactly restorable training states |
| [Marin 535B launch note](https://openathena.ai/blog/marin-535b-launch-note/) and [run issue #8435](https://github.com/marin-community/marin/issues/8435) | A dated public engineering case from the initial source review | A result from BranchLab or a statement about that run's current completion status |

Marin's website describes its Stanford origins and current Open Athena
development. This project therefore identifies the organizations and resources
separately rather than describing all current Marin work as a Stanford course
artifact.

The Delphi audit pinned dataset revision
`78f8e0c57d76326876e93d925354aad156188f08` and checked six tables totaling 4,117
rows. Rows include different record types and may overlap in run identity; they
are not 4,117 independent training trajectories. The checked tables did not
establish the availability of full optimizer state, random-number-generator
state, data cursors, or a complete per-step intervention history. See the audit
for its actual schema, hashes, and limitations.

The released diagnostic experiment uses locally generated intervention
observations. It does not train the synthesizer on fabricated Delphi
counterfactuals. A demonstrated benefit from Marin-derived initialization would
require a separate prior-versus-no-prior experiment; the release does not claim
that benefit.

## Nearest overlaps and the remaining question

| Prior work | Relevant overlap identified in the design review | Consequence for BranchLab's claims |
| --- | --- | --- |
| [AutoTTS](https://arxiv.org/html/2605.08083) | Controller synthesis using precollected inference trajectories and probe signals | A controller that reads a probe is insufficient to establish novelty; BranchLab makes the training intervention, horizon, and readout explicit in a finite DSL |
| [DoVer](https://arxiv.org/html/2512.06749) | Checkpointing, interventions, restoration, and differential utility in agent debugging | Cloning a state and comparing interventions are engineering mechanisms, not a new principle introduced here |
| [Step-DAD](https://arxiv.org/html/2507.14057) | Updating an experimental-design policy during experimentation | Adaptive experiment selection has prior art; the pertinent comparison is search efficiency under the same observations and budget |
| [GoBOED](https://arxiv.org/abs/2605.26093) | Experimental design targeting subsequent decision quality | Scoring an observation through repair regret is not independently a novelty claim |
| [Counterexample-guided automata learning](https://sws.cs.ru.nl/publications/papers/fvaan/CEGAR12/FM.pdf) | Counterexamples refine an observation abstraction | Using counterexamples to improve what a system observes is an established idea |
| [STOP](https://arxiv.org/abs/2310.02304) and [Hyperagents](https://arxiv.org/abs/2603.19461) | Programs or meta-improvers participate in improving the improvement procedure | Accumulating probes does not establish an improved meta-improver, let alone full recursive self-improvement |

These are source-level comparisons, not claims of benchmark superiority or
implementation equivalence. The release compares transparent local baselines;
it does not reproduce every cited system.

## The implemented mechanism

The released finite language has 18 nontrivial candidates:

```text
action   := lr_half | momentum_zero
horizon  := 2 | 4 | 8 optimizer updates
readout  := loss_delta | recovery_slope | grad_alignment
probe    := action branch contrasted with a keep branch at the same horizon
program  := free state/log features plus at most two selected probes
```

`keep` is a legal primitive and the comparison branch, not an extra informative
candidate. The action and readout operators are written by the developer.
Selecting and concatenating members of this finite language does not invent a
new primitive, generate unrestricted code, or change the language model's
architecture.

The implementation's `grad_alignment` identifier denotes a model-update
alignment contrast against the `keep` branch. The reference is the clipped
gradient from the last completed baseline update, which was already available
as a training log. It is not a freshly computed gradient at the intervened
state; this distinction matters when interpreting the diagnostic feature.

The repair predictor is a ridge model fitted on discovery episodes. The
counterexample strategy prioritizes discovery episodes with high current repair
regret, pays for a small number of candidate observations, and ranks those
candidates using leave-one-out residual prediction. Complete candidate programs
are promoted only when they reduce development regret. A promoted program is
inherited by the next search generation. Test episodes neither generate nor
promote candidates.

Enumeration and random ordering use the same language, predictor, cache rule,
and query-budget unit. The fixed expert and free-log baselines clarify whether
search is useful at all. A hindsight repair oracle is explicitly nondeployable:
it requires observing all future repair outcomes.

The release also includes a `full_enumeration` reference with a larger search
cap. It is explicitly not an equal-search-budget comparator. The actual
revealed cells, rather than only configured caps, appear in the report. Repeating
a deterministic counterexample/enumeration policy with a different search seed
does not produce a new independent training repeat.

The bounded question is whether counterexample ordering discovers useful
diagnostic programs more efficiently than these alternatives on the measured
training states. The complete input table's real collection cost is recorded
separately from hypothetical replay queries. Multiple readouts can share a
physical branch; per-probe replay accounting conservatively charges the
provided cost and does not turn shared computation into an unsupported saving.

## Evidence required for stronger claims

- Reduced repair regret at a fixed continuation horizon supports better repair
  selection within the tested action menu. It does not by itself establish a
  reduction in total training computation.
- In this pilot, regret is measured on development text at a fixed repair
  horizon for held-out training seeds. The primary fixed-budget result is
  cross-entropy on a separate test-text batch, after charging diagnostic costs
  and using the remaining proxy budget for repair. These two metrics must not
  be conflated.
- Improved loss after charging diagnostics and allocating the remaining budget
  supports the stated budget proxy on the tested setup. Forward/optimizer-step
  equivalents do not establish a hardware wall-clock speedup.
- Multiple states from one seed remain correlated. A seed-isolated pilot does
  not establish a general result for other model scales, data sources, or
  natural training failures.
- Inheriting selected probes demonstrates program reuse. The synthesizer's
  own generation and promotion rules remain fixed in this release.
- Recursive improvement would require updated and frozen synthesizers, equal
  histories and initial libraries, equal search budgets, and held-out evidence
  that the updated synthesizer produces better descendants. That experiment is
  not implemented here. BranchLab does not claim full RSI or SOTA.

The machine-generated release report states the observed results, including
negative or inconclusive outcomes. Repository completion and passing software
tests are separate from support for the research hypothesis.
