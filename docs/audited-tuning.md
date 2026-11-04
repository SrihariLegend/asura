# Game-audited tuning: a proved estimator, a negative calibration pilot

**The 100–1000× total-compute/playing-strength objective is still unachieved.**
This experiment tested whether cheap predictions could replace most expensive
game measurements without blindly trusting the predictions. The mathematics
checks out under explicit sampling assumptions. The two cheap inputs tested here
provided **no held-out residual-variance reduction**. No settings were adopted.

The production NNUE and search parameters remain unchanged. This is a follow-up
to [the Hyper-SPSA repair](hyper-spsa.md), not a replacement optimizer or a trainer
update. The separate [trainer gradient findings](trainer-vjp-audit.md) remain unresolved.

## 1. What the method would do

For one fixed candidate and a fixed cohort of `N` color-reversed opening pairs:

- `y[i]` is the actual signed paired score: total candidate points in the two games
  minus one. Its possible values are `-1, -0.5, 0, 0.5, 1`.
- `p[i]` is any cheap prediction, fixed before selecting the audit indices.
- Draw `k > 0` indices independently and uniformly **with replacement**.

Estimate the candidate's finite-cohort mean with

```text
mean(all cheap predictions) + mean(actual result - prediction on sampled pairs).
```

The correction removes systematic prediction bias **in expectation**, not in every
individual sample. Accurate predictions can make the correction less noisy. A
constant prediction cannot reduce its variance: subtracting the same number from
every result leaves their spread unchanged.

`proofs/audited_tuning.lean` derives the finite product distribution, then proves

```text
E[estimate]   = mean(y)
Var(estimate) = Var(y - p) / k.
```

Here `Var` is a finite-population centered mean square, denominator `N`. The proof
does **not** assume the estimator is unbiased or assume away the covariance terms.
It also proves concrete prediction-only, selective-audit and nonuniform-unweighted
audit counterexamples. Randomness alone is insufficient: for example, sampling
scores `[1,-1]` with probabilities `[3/4,1/4]` gives an unweighted mean of `1/2`, not zero.

The ordinary comparison in these identities is an iid **with-replacement mean**,
not a complete SPSA campaign. A census has zero sampling variance; without-replacement
sampling, deterministic replay caching, shared random numbers and other optimized
baselines require their own accounting. Timed-game determinism is not asserted:
`y` is a frozen mathematical conditional cohort. RNG correctness, unseen openings,
adaptive winner selection, optional stopping and confidence intervals are not proved.
An unbiased candidate estimate alone does not make selecting its maximum safe.

## 2. A quantitative requirement, not a speedup claim

If an ordinary mean uses at least `s*k` draws, while `k` residual audits must have
no greater variance, the proof establishes the necessary condition

```text
Var(y - p) <= Var(y) / s.
```

Thus **100× fewer audit draws requires at most 1% residual variance**; **1000×
requires at most 0.1%**. These are necessary sampling-variance conditions for this
estimator, not universal limits on every possible tuning method.

For a fixed linear control variate `p = beta*x`, the proof completes the square:

```text
minimum residual variance = Var(y) * (1 - rho_squared).
```

With positive variances, even an oracle coefficient needs squared correlation
at least **0.99 / 0.999** for those respective sample reductions. This is explained
variance, not classification accuracy. The oracle coefficient uses true cohort
covariance; learning it is not free. Nonlinear/richer predictors are not bounded
by the original scalar feature's linear correlation.

Total cost must additionally include

```text
N * cheap-prediction cost + calibration + k * high-fidelity pair cost
  + independent confirmation + other preparation/overhead.
```

Each pair cost includes **both games**. The Lean file proves that positive overhead
prevents interpreting an exact `s`-fold game-call reduction as an exact `s`-fold
all-in saving against a game-only baseline. A real comparison must include the
baseline's overheads too, use common compute units and equivalent strength endpoints.
Parallel elapsed time is not the sum of per-game elapsed times.

All **55 theorems** compile with Lean 4.35.0 and warnings treated as errors. Every
theorem has a transitive dependency audit; only `Classical.choice`, `Quot.sound`
and `propext` occur. No proof holes, local axioms or native-evaluation proof oracle.
This is not a formal refinement proof of the chess engine or a strength guarantee.

## 3. Registered native experiment

Frozen engine source: `6221d252ccb1ee911cfdf3a477ae1b943e37d244`, plus the standalone
`tools/tuning_calibration_probe.rs` driver. Checkpoint SHA-256:
`1402eeb715752a7ec4c85f72ea2d0042c819157588643525f6bb432d727cc807`.

Four one-step corners of the first pair in parameter block 2 were fixed before
observing any games:

| Profile | LMR table scale | Singular beta margin |
|---|---:|---:|
| baseline | 418 | 26 |
| ll | 404 | 22 |
| lh | 404 | 30 |
| hl | 432 | 22 |
| hh | 432 | 30 |

All other settings were held at baseline. Candidate and opponent used the same
runtime-enabled parameter path. Eight 50k-node public-UCI checks established that
this baseline path matched the production-default path in move, score, depth and
node count. Candidates use temporary process-local settings; no source is rewritten.

The pilot reused the first eight frozen opening families from the earlier audit:

- Each candidate played both colors from each opening against baseline.
- **64 complete games at 5,000 nodes/move**, followed by **64 at 500,000**.
- One thread and 16 MiB hash per engine; eight simultaneous games on CPUs 12–23.
- No score-based resignation/draw or maximum-move adjudication. Natural terminal
  results only; complete pairs required. Hard controller limit: 1,800 seconds.
- Before all games, the actual native position adapter scored the eight initial
  positions at **500 / 1,500 / 4,500 nodes**, with 6,000-node frozen labels,
  four workers and 1 MiB hash per worker. Scorecard roots reset search state;
  games retain normal within-game search history. The scalar feature was
  candidate utility minus baseline utility.
- A separate affine predictor for each cheap source was fitted on the **first four
  opening families** (16 candidate/pair observations), then evaluated on the other
  four (16 observations). Predictions were clipped to `[-1,1]`. This fit, split,
  analysis and stopping policy were registered before the games.

**Scope limitation:** these are early-opening, neutral-score fixtures, not a
representative middlegame/conversion/style corpus. Opening reuse is disclosed;
exclusion from NNUE training is not established. The shared four held-out opening
families do not create 16 independent game-distribution samples. These are
small-cohort diagnostics, not population correlation estimates or Elo tests.

## 4. Results

All **128 games / 64 color-reversed pairs** passed independent python-chess replay:
legal moves, expected initial positions, swapped colors and natural outcomes.
There were no time forfeits or unexplained adjudications. At 5k nodes, 43/64 games
were decisive; at 500k, 13/64 were decisive. Their different draw behavior is one
reason not to equate the cheaper games with the target game control.

The position scorecard gave **exactly the same scores for all four perturbations**
at all eight positions. It supplied no ranking signal in this pilot. That is not
proof that the objective is constant on broader positions, nor proof it discarded
a genuine overall strength improvement: none was established here.

Held-out residual spread, centered **within each candidate** and pooled with equal
cohort sizes (sum of residual centered squares / sum of outcome centered squares):

| Cheap input | Residual-variance ratio | Required for 100× audits | Required for 1000× audits |
|---|---:|---:|---:|
| Native position scorecard | **1.0000** | <=0.01 | <=0.001 |
| 5k-node game results, calibrated | **1.0183** | <=0.01 | <=0.001 |

The first removed none of the spread. The second increased it by about **1.8%**.
Even a hindsight best common linear slope on the held-out, within-candidate
centered cheap-game feature had squared correlation only `1/1404` (about 0.00071).
That last number is an **in-sample oracle diagnostic**, not a deployable fitted
model, a confidence bound, or a restriction on richer predictors.

For transparency, all-game scores against baseline—not candidate adoption gates:

| Profile | 5k points / 16 games | 500k points / 16 games |
|---|---:|---:|
| ll | 7.5 | 8.0 |
| lh | 5.0 | 7.5 |
| hl | 8.5 | 6.5 |
| hh | 7.5 | 7.5 |

The cheap-game leader was not the high-budget leader. These small samples do not
establish which setting is globally strongest. **No candidate was adopted.**

### Cost, including negative work

- Complete registered run: **654.36 s elapsed**, **4,687.45 child CPU seconds**.
- Cheap games: **50.09 child CPU seconds**, **47,915,154 reported nodes**.
- Target games: **4,634.75 child CPU seconds**, **3,873,510,619 reported nodes**.
- Native scorecard including label construction: **1.53 child CPU seconds**.
- Two administrative setup failures before any game: **2.81 child CPU seconds**,
  retained with exact registered sources and logs. The first assumed optional UCI
  PV output; the second used a Fastchess config-discard flag that removed the engine
  and book arguments. Neither caused a game result to be discarded or a hypothesis
  to change. Position scores were identical across the corrected setup.

The cheaper game cohort used about **92.5× less measured child CPU time**, but did
not provide the required prediction quality. This is precisely why speed per
measurement is not the target result. The study itself played **every** registered
game, so it demonstrated **no audit-game saving**. Historical opening construction,
cached builds and interactive development/proof/debug costs were not completely
instrumented; they are not assumed free. There is no all-in SPSA campaign comparison.

## 5. Reproducibility and next decision

- [tuning-calibration-results.json](tuning-calibration-results.json): coefficients,
  exact fractions, pair observations, costs and source/check hashes.
- [tuning-calibration-artifacts.tar.gz](tuning-calibration-artifacts.tar.gz): 128-game
  PGNs, registrations, logs, scorecards, original runners (including failed setup),
  checksums and proof-check data. No model or executable binaries included.
- `tools/analyze_tuning_calibration.py`: exact-rational registered split/analysis.
- `tools/check_audited_tuning.py`: all-theorem audit and optional finite enumeration
  of frozen game cohorts in Python and compiled Lean.

```sh
ART=$(mktemp -d)
tar -xzf docs/tuning-calibration-artifacts.tar.gz -C "$ART"
python3 tools/analyze_tuning_calibration.py "$ART/tuning-calibration" \
  --output "$ART/analysis.json"
python3 tools/check_audited_tuning.py --analysis "$ART/analysis.json" \
  --work-dir "$ART/proof-check"
```

The recorded executable check matched **24/24 cohort/sample-count rows** (eight
predictor/candidate cohorts, counts 1–3). It evaluates all ordered audit tuples,
including repeated indices, and checks the expectation and variance identities.
It does not pretend that reusing already-played games is a new independent match,
or that this is a completed audited-SPSA implementation.

**Decision:** do not put this cheap-game substitute into the tuner on the strength
of this pilot. Keep the valid residual-audit mathematics, but first seek a more
informative predictor and use fresh, broader held-out game families. A richer
position/decision-quality surrogate might nominate good candidates without
predicting individual game outcomes; this experiment neither validates nor rules
out that different route. Any promising method still needs an equal-total-compute
comparison with competently configured regular SPSA and independent strength gates.
