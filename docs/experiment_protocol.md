# Experiment Protocol

## M9 execution conditions

The harness technically supports `BASELINE`, `L1`, `L2`, `L3`, `L1_L2`,
`L1_L3`, `L2_L3`, and `FULL`. The initial primary comparison is restricted to
`BASELINE`, `L1`, `L1_L2`, and `FULL`; the other conditions remain executable
for engineering checks and later analysis.

`BASELINE` disables Layer 1 contract injection, Layer 2 intervention, and Layer
3 reset. The consistency monitor remains in passive measurement mode: it may
record raw drift indicators, but it cannot repair, block, reset, rewrite output,
or change routing. `FULL` enables all three role-consistency layers. Safety and
observability remain enabled in every condition and are not experimental
guardrail layers.

## Invariant controls

Across conditions, the harness fixes the three role-agent names and
responsibilities, graph routing topology, model configuration, prompt version
except for Layer 1 content, problem input, and execution environment. Each run
records its condition, seed, model configuration, trace identifier, raw drift
indicators, configuration source, and a SHA-256 hash of the complete resolved
configuration.

## Current M9 scope

M9 repeats a supplied single-turn input and exports one observation per
repetition to CSV. Simulated learner profiles and automated multi-turn learner
behavior are deferred to M10. Statistical analysis and final sample-size,
dataset, and human-annotation decisions remain deferred to the research
evaluation protocol.
