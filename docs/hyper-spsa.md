# Hyper-SPSA: checked proposals, not a strength certificate

This repair changes the tuner, not the deployed search settings or NNUE weights.
The engine correctness fixes are recorded separately in [correctness-audit.md](correctness-audit.md).
The current evaluator/trainer findings remain in [strength-audit.md](strength-audit.md)
and [trainer-vjp-audit.md](trainer-vjp-audit.md). Training gradients and search-parameter
perturbations are different processes; repairing either does not automatically repair the other.

## What failed

The old tuner at `6f5c355:src/bin/hyper_spsa.rs` applied 16 Hadamard directions
and their negatives, then attributed four mixed interactions to selected pairs.
Distinctness of those four selected patterns does not exclude other interactions.
The actual assignment contains `1 XOR 2 = 8 XOR 11 = 3`.

Two production-derived counterexamples were reproduced before replacement:

- On `f(x) = 7*x[3]*x[7]`, the old estimator reports an interaction of **7** for
  coordinates **0 and 1**, even though neither appears in the objective.
- On noiseless `f(x) = x[0] + 100*x[1]`, the old gradient for coordinate 0 is
  exactly **1**, but its supposed standard error is **25**. It discards a real
  effect because deterministic variation from another coordinate is treated as noise.

The old formulas are retained only in `src/bin/hyper_spsa/legacy.rs` for executable
counterexamples and comparisons. The historical `~300x` campaign claim compared
an assumed 63-hour conventional campaign with a claimed 12-minute Hyper campaign.
It did not demonstrate equal playing-strength improvement at those costs.

## The replacement

`src/tuning.rs` is the small, objective-independent interface:

- `probe_block`: bounded eight-coordinate incumbent plus a fixed objective
  produces observations, exact integer contrasts and **unvalidated** proposals.
- `select_on_panel`: same-panel measurements and fixed protected-score limits
  select a strictly better eligible candidate, or retain the incumbent.

A probe changes two coordinates while holding the other six fixed. Its four
corners determine a bilinear interpolant of that face. Four pairs require at most
**16 additional objective calls**, with duplicate points cached. Seven consecutive
pair partitions cover all 28 pairs. If the incumbent changes between rounds,
those rounds do not reconstruct one full Hessian at a single point.

The contrasts recover step-scaled first/mixed coefficients of a centered quadratic.
For an arbitrary chess objective, they describe only the sampled face. Clipping
can move the face center away from the incumbent. Diagonal curvature and
higher-order interactions are not magically recovered.

Up to **ten distinct proposals** are declared before validation:

1. A normalized direction and its half-step; immobile/outward coordinates are
   masked before normalization. This can retain useful single-axis progress
   when curvature makes every two-axis corner lose.
2. Merged pair winners and their half-step.
3. Four individual pair winners.
4. Two antipodal global endpoint trials, to explore beyond pair-local changes.

Every proposed setting must be evaluated. Gains from separate pairs are not
assumed to add safely. Exploratory trials are not free observations. With known
incumbent scores, there are at most 16 training-panel calls plus 10 validation-panel
calls; panels can have different per-call costs. Label building and baseline
measurements are additional work.

The protected-score limits remain anchored to the original baseline, not renewed
against each successive incumbent. A tolerance of 5 below an initial score of
100 cannot permit a sequence ending at 90. This is a finite-panel property, not
an Elo or population guarantee.

## Native execution

`src/bin/hyper_spsa/evaluator.rs` runs actual bundled-format NNUE and native search.
Missing models, missing move labels, insufficient distinct positions, malformed
inputs and failed worker batches are errors, not passing or partial measurements.
Workers retain allocated state but clear/reset search state between roots. They
finish outstanding work before tunable restoration. Integer metrics are reduced
in position order. The label formulas remain handcrafted **proxies**, not observed
winning/drawing probabilities or independent chess truth.

The CLI preserves six historical parameter groups, with **47 unique parameters**;
shared parameters use intersected bounds. Bounds, model and input paths are
explicit. `tune` uses disjoint training, validation and final-test positions. The
final candidate is fixed before final-test scoring, with no retry on that panel.
Disjoint positions are not necessarily independent games, and exclusion from
NNUE training data is not established. Repeated adaptive validation can still overfit.

The program never overwrites parameter source, data or checkpoints. Printed
candidates still require real, held-out strength matches. All parameters and the
prior SPSA-active flag are restored on normal/error return. Unwinding worker
failures are handled; release `panic=abort`, OOM abort and external process death
cannot be recovered by destructors.

Node budgets use the engine's existing stopping policy: its first completed depth
is exempt. Deadlines are checked between roots, not inside a running root search.
Use an external process timeout for a hard operational limit. Regular-file I/O
has no portable hard-time guarantee either.

## What Lean establishes

`proofs/hyper_spsa.lean` contains **119 checked theorems** covering the counterexamples,
face reconstruction, quadratic specialization, integer widening/clipping/truncation,
normalized proposals, pair schedule, bounded query/proposal counts, and fixed-anchor
selection. Under an exact noiseless affine objective, the normalized proposal makes
strict progress when a coordinate can improve; selection retains that progress if
the proposal satisfies the protected constraints.

The file also proves limitations: finite observations cannot guarantee an arbitrary
unseen objective's optimum; a finite-panel gain need not generalize; a quadratic
need not equal its bilinear interpolant in the interior.

**Noise tradeoff:** with four distinct new observations versus the old 32 observations,
independent equal-variance additive noise and unit perturbation steps, the four-corner
gradient contrast has **eight times** the variance. Halving probe count is not an
equal-accuracy stochastic speedup. The actual position-search adapter uses fixed,
repeatable panels; that does not establish population generalization.

Every theorem has a dependency audit. Only `Classical.choice`, `Quot.sound` and
`propext` occur; no proof holes, new axioms or native-evaluation proof oracle.
These are mathematical specifications, not a formal refinement proof of the whole
Rust program, IEEE arithmetic, NNUE judgment, Elo or a 100–1000x strength-tuning advantage.

`tools/check_hyper_spsa.py` executes theorem-backed Lean definitions against the
compiled public Rust optimizer. The recorded run matched **4,399 cases**:
1,137 contrast, 1,625 bounds/halfstep, 736 fixed-anchor selection and 901 complete
quadratic probe reports. The latter include all 896 combined pair/mask schedule
rounds. Both implementations rejected 28 malformed streams. Panel-shape/multi-metric
errors, arbitrary objectives and native NNUE execution have separate Rust tests.

## Measurements

Machine-readable timings, query counts and input/source hashes are preserved in
[hyper-spsa-results.json](hyper-spsa-results.json). Actual NNUE, six persistent workers,
1 MiB hash per worker, CPUs 12–23; one parameter group, frozen incumbent, alternating
old/new order. Both designs use the **same repaired native adapter and guarded
validation**, not their entire historical/new executables.

| Search-budget pyramid | Repeats | Train/validation/test positions | Probe speedup | Probe + validation speedup |
|---|---:|---:|---:|---:|
| 500 / 1,500 / 4,500 | 6 | 6 / 6 / 6 | 1.96x | 1.66x |
| 50k / 150k / 500k | 3 | 2 / 2 / 2 | 2.10x | 1.79x |

Phase timings exclude shared label construction and incumbent scoring. Actual
probe calls were 32 versus 16; complete cycles were 34 versus 20. On the small
panel neither method accepted a validation gain. On the larger-budget, two-position
validation panel only the repaired method found an eligible proxy gain. Repeated
timings are not independent strength observations. No candidate was adopted.

In constructed noiseless optimization fixtures, 32 seeds per family, the cap was
512 objective calls including validation; unused budget was reported, not treated
as spent. Mean regret (distance in objective value from the known optimum):

| Diagnostic family | Legacy | Repaired |
|---|---:|---:|
| Separable quadratic | 0 | 0 |
| Dense quadratic | 115.56 | 2.66 |
| Integer plateau | 38.56 | 3.75 |
| Three-coordinate barrier | 0 | 0 |

These families helped develop the method, so they are not an untouched algorithm
tournament. The repaired method still missed the optimum in 1/32 dense and 12/32
plateau cases. The additional two-probe comparator is a simple untuned normalized
SPSA baseline, not evidence against a competently tuned conventional campaign.

Native and generic x86-64/non-AVX2 suites each passed **186 tests**, with one explicitly
ignored external-oracle test. Strict whole-library Clippy is not clean because of
pre-existing warnings; no new tuner warnings were observed in the targeted advisory
run. These results do not establish a playing-strength gain.

## Reproduce

```sh
cargo test
cargo run --release --bin hyper_spsa -- check
python3 tools/check_hyper_spsa.py --target-dir /tmp/asura-contract/target

# Actual NNUE mechanical check, not a strength suite.
cargo run --release --bin hyper_spsa -- verify \
  --model asura_atn1.nnue --fens tests/fixtures/hyper_spsa.fens \
  --train 2 --validation 2 --test 2 --scan 20 --threads 3 \
  --nodes 64,128,256 --label-nodes 256 --blocks 2 --max-seconds 30

python3 tools/benchmark_hyper_spsa.py \
  --model asura_atn1.nnue --data /path/to/asura_bt4_64m.bin \
  --output /tmp/new-hyper-benchmark \
  --nodes 500,1500,4500 --label-nodes 6000 --blocks 2
```

The checker needs installed Lean/`leanc` (tested with Lean 4.35.0), Rust and Python
3.11+, uses an isolated Cargo target, and fails explicitly if fixtures/tools are
missing. No dependencies are downloaded. The benchmark copies a bounded dataset
prefix and model into a new artifact directory, records hashes and enforces a
subprocess timeout. Its defaults are an experiment configuration, not CLI defaults.
The CLI currently defaults to 2k/8k/32k nodes and twice the largest tier for labels;
these remain screening budgets, not STC/LTC game controls. No arguments show help
rather than silently starting the historical hard-coded workload.

## Outstanding objective

The target is **real playing-strength improvement at 100–1000x less total tuning
compute than conventional SPSA**. It is not established. The decisive next question
is whether cheap measurements predict real game improvements across candidate
settings. Preparation, calibration games, failed candidates and final independent
validation must all count toward the claimed cost. A faster mistaken scorecard is
not success.

A subsequent [game-audited tuning study](audited-tuning.md) proved an unbiased
residual-correction estimator and completed a 128-game fidelity pilot. Its tested
cheap inputs did not reduce held-out residual variance; no candidate was adopted
and no compute-saving claim resulted. The goal remains open.
