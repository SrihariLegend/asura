# Strength audit: what the evidence currently supports

Baseline: commit `6f5c355`. Checkpoint SHA-256:
`1402eeb715752a7ec4c85f72ea2d0042c819157588643525f6bb432d727cc807`.

This investigation did not change production search, evaluation scaling, strength
parameters, training data, or weights. It did not train or run an optimizer step.
Experimental search changes and trainer candidates were confined to private copies.
The earlier correctness fixes are documented in [correctness-audit.md](correctness-audit.md).

## Bottom line

There was no sound evidence that the native Asura/checkpoint combination had
world-top-five strength. New matches confirm a substantial gap against C. We have
also isolated an evaluation-dependent bad decision in a real loss, and reproduced
current trainer backward-contract failures. Those are much stronger leads than
another general correctness sweep, but they do not yet establish how many Elo
points any particular training defect cost this checkpoint.

## 1. Completed measurements

All three cohorts used the frozen native evaluator, one thread and 16 MiB hash per
engine, color-reversed opening pairs, and no score-based resignation/draw rules.
The opening set was newly assembled and checked; it is not claimed to be absent
from training data or to establish a world ranking.

| Against C | Games | W–L–D | Points | Score |
|---|---:|---:|---:|---:|
| 500,000 nodes/move | 24 | 1–14–9 | 5.5 | 22.92% |
| STC `10+0.1` | 24 | 1–14–9 | 5.5 | 22.92% |
| LTC `60+0.6` | 8 | 0–3–5 | 2.5 | 31.25% |

Independent PGN replay verified all 56 games and 28 color-reversed FEN pairs.
All 33 decisive games ended in checkmate; the 23 draws were ordinary repetition
or insufficient-material draws, not time forfeits or unexplained adjudications.

The runner's STC estimate was −210.72 ±112.66 Elo; these are small samples against
one opponent, not an absolute rating. The equal-budget losses mean wall-clock
throughput alone is not an adequate explanation. Different engines' reported
nodes are not identical units of computational work.

Artifacts: `/tmp/asura-strength/{nodes500k,stc,ltc}.{pgn,log}` and corresponding
`*-command.json`, `openings.json`, and `engine-manifest.json`.

## 2. A bad decision isolated beyond a pruning-only explanation

From an actual STC loss, Black to move:

```text
r1bq1rk1/b1pp1p2/p1n2P1p/1p4p1/B3P2B/2P2N2/PP3PPP/RN1Q1RK1 b - - 0 11
```

Asura chooses `g5h4` (`...gxh4`) at 50,000, 500,000 and 5,000,000 nodes, with
scores −21, −20 and −16 cp. Independent two-million-node C root probes returned
exact search scores of −122 for `d8f6` (`...Qxf6`) versus −238 for forced `g5h4`,
from Black's perspective. These are reference estimates, not a proof of the
position's game-theoretic value or that this was the game's first mistake.

We then held Asura's public unpruned diagnostic search fixed and changed only the
leaf evaluator in a private copy. These nominal depths include its quiescence
continuation:

| Evaluator | Depth 4 choice / score | Depth 5 choice / score |
|---|---|---|
| Frozen Asura NNUE | `g5h4` / −39 | `g5h4` / −18 |
| Same NNUE, bypassing post-scaling | `g5h4` / −41 | `g5h4` / −18 |
| Archival C static-evaluator bridge | `d8f6` / −85 | `d8f6` / −78 |

The bridge was reset from the canonical FEN at every evaluation; its unsafe raw
position-layout fallback was not used. Its build is not assumed identical to the
C executable used for matches. Library/source hashes were recorded separately.
This was a diagnostic substitution, **not** a proposal to replace Asura's evaluator.

This establishes an evaluator-dependent ranking difference under the same
unpruned search at these depths. The inferior choice persists without pruning and
without Asura's scaling. It does not establish that every loss is evaluation-led,
that selective search is blameless, or that a reference score is chess truth.

Several individual pruning ablations also recovered `d8f6` at 500,000 nodes.
That shows a search/evaluation interaction, not that those heuristics are globally
bad. A one-position success is insufficient reason to disable or tune one.

Reproduce the native decision from the repository root:

```sh
ASURA_NNUE_PATH="$PWD/asura_atn1.nnue" target/release/asura \
  'position fen r1bq1rk1/b1pp1p2/p1n2P1p/1p4p1/B3P2B/2P2N2/PP3PPP/RN1Q1RK1 b - - 0 11' \
  'go nodes 500000'
```

Detailed artifacts under `/tmp/asura-strength/`: `probe-ruy-11.json`,
`ablate-ruy-11.json`, `controlled-eval-swap.json`, `bridge-manifest.json`,
`search-copy/`, and the frozen `search-audit-frozen` helper. FEN replay is not a
bit-for-bit recreation of the original game's accumulated search history.

## 3. Current trainer backward code has consequential contract failures

Production trainer: `train_atn1_cuda.cu` in the external lab directory. The current
source was exercised on the GPU without compiling/calling optimizer updates.

The clearest defect is FiLM routing. For

```text
y = h + beta + gamma * observed(h >> 7)
```

the incoming cotangent `d` branches. The h branch gets
`d * (1 + gamma/128)` under the declared shift STE; the beta/PK branch must not
reuse that already-transformed cotangent. Current code does reuse it when
propagating into PK hidden coordinates. A gamma-64, zero-gamma-weight fixture
isolates this from saturation: **113/113 active hidden lanes disagree**, with the
expected erroneous factor 1.5. A private candidate removes that mismatch.

Other observed mismatches involve gamma-clamp masking, exported bias coordinates
scaled by 32, and the shared-base weight fold's factor 0.65. These must be judged
against an explicit STE and coordinate policy, not against the literal derivative
of integer inference. Some historical output-branch factors were **deliberate
preconditioning**, so they must not all be called accidental calculus bugs or
removed blindly.

On 3,072 sampled positions, 49,136/49,152 gamma outputs were at a clamp limit.
Each lane retained the same cap sign in that sample: fourteen predominantly +127,
two predominantly −127. This is near-constant multiplicative modulation on the
sample, not proof that the whole PK subnet is dead; beta remains available.

Tested forward/export paths passed: two model fixtures × 3,072 positions agreed
between production GPU, the separate oracle, and native/scalar Rust. This does
**not** recover the historical float/EMA masters or establish the complete
training, optimizer, resume, model-selection, and export pipeline.

The exact source/build/master state that produced the deployed checkpoint remains
unknown. Current-source bugs therefore cannot yet be assigned historical causality
or an Elo cost.

## 4. Lean is now checking a relevant contract—and caught a candidate gap

The durable proof is [proofs/trainer_vjp.lean](../proofs/trainer_vjp.lean).
[trainer-vjp-audit.md](trainer-vjp-audit.md) records assumptions, source mappings,
GPU witnesses, and limitations.

It derives finite-dimensional adjoints for FiLM inputs and parameters, matrix and
shared-base accumulation, and exported bias units. It proves wrong-routing
counterexamples and separates an equal-coefficient output VJP from deliberate
branch preconditioning. There are no proof holes or new axioms; dependency reports
cover every theorem.

Importantly, Lean also exposed an unresolved boundary-policy distinction in the
private repair: a pre-shift sum of −4048 becomes −127 after integer division, while
the unrounded value −126.5 is inside the clamp interval. Masking by the observed
capped output is not the same policy as masking the unrounded preactivation.
The candidate passed the original GPU fixtures, but that does not certify its
conformance at this boundary. **It remains unapplied.**

```sh
lean -t 0 -DwarningAsError=true proofs/trainer_vjp.lean
```

These are exact-rational local STE specifications, not proofs of IEEE arithmetic,
complete CUDA/Rust implementations, chess judgment, or Elo. The production-code
bridge is executable evidence, not a formal refinement theorem.

## 5. Evaluation-quality numbers need careful interpretation

Native/scalar/component agreement was checked on 196,608 records sampled across
six files. Against stored targets, RMSE was 66.19 cp over the four main-file
samples (131,072 positions) and 160.81 cp over the dedicated cusp sample (32,768).
These are target-fit measurements on existing data, not held-out strength tests.

Older and current label sets cannot be interchanged: 32,768 checked row pairs had
the same positions, but 20,650 had different targets. On those same predictions,
RMSE was 61.71 against current targets versus 125.99 against older targets.

A conspicuous +640-target/−120-native fixture was not corroborated by the short
reference searches and is **not** counted as a confirmed evaluator failure.
Static disagreement can also reflect tactics that quiescence/search must resolve.
The controlled actual-game decision in section 2 is stronger causal evidence than
bulk target-fit metrics alone. Raw measurements and scripts are in
`/tmp/asura-strength/quality/`; its background investigator ended before producing
a final narrative, and these limitations were checked during consolidation.

## 6. Historical confidence exceeded the evidence

The previous top-five/top-six assurances depended on a defective external-evaluator
match harness and an invalid translation to an external rating list. Later claims
also substituted node savings, teacher agreement, and small arithmetic proofs for
native playing-strength evidence. The displayed “0.00 cp QAT Loss!” was a literal
banner, not a computed quantization-loss measurement.

Real training and validation work did occur. The central error was repeatedly
validating a different object from the one claimed.

Coverage: the extracted user/assistant text and all 13 compaction summaries were
reviewed. All 3,658 historical tool-result records were catalogued, with targeted
close reading of the important training/proof/strength evidence. This is **not** a
claim to have closely read every raw tool-result block. Evidence and source-line
references are in `/tmp/asura-strength/history/REPORT.md`.

## Remaining work and experiment status

- Fix/specify the trainer's local contracts and boundary policies before training;
  connect the global loss, QAT, gradients, optimizer, export and native consumer.
- Establish checkpoint provenance where possible. No retraining or checkpoint
  replacement was performed in this pass.
- Build a broader actual-loss corpus and measure search/evaluation interactions
  without tuning to this one decision.
- Validate any proposed repair on held-out paired openings and the requested
  time/node controls. Full depth-0–30 and end-to-end production-training gates are
  not claimed complete here.
- Optional NoFP and A follow-up matches were cancelled during opening preparation,
  **before either match started**. There are no results from them. The initial
  generator had an unbounded duplicate retry; after bounding it, preparation was
  still too slow and was stopped to deliver the completed findings.

No new playing-strength improvement is claimed. The evidence now identifies
specific evaluation and trainer problems to investigate, rather than treating the
earlier correctness fixes or proof counts as a strength result.
