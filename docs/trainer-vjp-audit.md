# Checked local QAT contracts

The proof is preserved in the repository; ancillary logs and reproduction scripts named below are audit artifacts under `/tmp/asura-strength/lean/`. To check the durable proof directly, run:

```sh
lean -t 0 -DwarningAsError=true proofs/trainer_vjp.lean
```

## Result and artifacts

**`proofs/trainer_vjp.lean` passes Lean 4.35.0, with trust level 0 and warnings treated as errors.** It is self-contained apart from the installed `Std`. Its 46 checked theorem declarations include finite-sum infrastructure, substantive local adjoints, and counterexamples. This is **not** 46 verified production functions, a complete CUDA proof, a Rust validation, or a strength result.

- [`proofs/trainer_vjp.lean`](../proofs/trainer_vjp.lean): complete specification and proofs.
- `/tmp/asura-strength/lean/check.sh`: original repeatable CPU-only check.
- `/tmp/asura-strength/lean/lean-check.log`: original dependency reports for **every** theorem; `repo-check.log` records the independent check of the repository copy.
- The same temporary directory contains `lean-check.exit`, `axiom-audit.log`, `environment.txt`, and `source-provenance.txt`: original status, dependency audit, compiler/affinity, and inspected input hashes.

Repository inspected at `6f5c355959ce4b86e50c7bdf8aa7c7c2dc25f0c9`; working tree status was clean. During that investigator run, no repository, shared trainer, checkpoint, or GPU-contract artifact was edited. The proof and these notes were subsequently preserved in the repository. No training, optimizer update, tuning, match, commit, or reset was run. Only Lean checks and read-only source/artifact inspection were performed here. Inherited CPU affinity **12–23** was retained. No dependency download was needed.

## What is specified, rather than silently assumed

All mathematical scalars are exact `Rat`; the small shifted-clamp model uses unbounded `Int`. `Vec n = Fin n → Rat`. `total` recursively sums each coordinate exactly once. Empty dimensions are allowed. The underlying pairing is the ordinary coordinate dot product, not a weighted metric.

The integer forward is not a differentiable real network. These are **declared straight-through tangent maps and their algebraic adjoints**, not claims of real/Fréchet derivatives of quantization. No real analysis library or differentiability theorem is used. In particular:

1. `roundSTE dx = dx`. This is a policy for a free quantization coordinate, not the literal derivative of rounding.
2. `shift5STE dx = dx/32`; `shift7STE dx = dx/128`.
3. `gammaMask u = 1` exactly when `-127 < u < 127`, otherwise `0`; `clampSTE u du = gammaMask u * du`. Zero at either boundary is an explicit policy. Strict plateau constancy is proved separately, not postulated.
4. In FiLM, `t` is the **observed** `h >> 7` primal value, while its tangent is `dh/128`. It is not replaced by `h/128` in the gamma producer's coefficient. Similarly, `gamma` is the observed clamped integer gate; `u` is the unrounded gamma affine sum divided by 32. Algebraic identities hold for arbitrary rational `gamma,t,u`, and do not hide an equality identifying rounded and unrounded primal values.
5. For each local adjoint, primal activations, coefficients, masks, active feature identities, and the row-to-base map are fixed. Cotangents and tangent directions are arbitrary, independently of any global loss. FiLM's two independent inputs are raw dense `h` and post-activation PK `a`. Propagation through earlier PK clipping/saturating additions is outside this local input space; the GPU hidden-lane comparison uses interior lanes only.
6. Bias/weight and folded-base statements concern the affine **pre-rounding** export coordinates, with identity weight STE. They do not cover i8 clipping, FT residual projection, or a quantizer's saturation boundaries. Applying them to production requires the corresponding free-coordinate policy or an interior fixture. Exporter `:728–729` can actually project the residual master in place; unchanged-master observations in the fixtures are not a general nonmutation theorem.
7. The fold uses any total `Fin r → Fin b` map, with no injectivity, surjectivity, or single-use assumption. Thus multiple residual rows can share a base. `rowScale` is `1` on a supplied PSQ classification and `13/20` otherwise. Correctness of the production feature classification/map is not assumed to be proved by the type.
8. Nonzero hypotheses in counterexample families exclude cancellation/zero-seed cases, rather than asserting a desired mismatch. They are: nonzero correct beta-path cotangent for routing, `d*t ≠ 0` for missing mask, `a*g ≠ 0` for bias units, `g0+g1 ≠ 0` for the two-row fold, and `g ≠ 0` for unequal output-path factors. There is also an unconditional explicit wrong-branch witness. Plateau hypotheses are explicit inequalities on preactivations/perturbations; no theorem assumes its adjoint identity as a premise.

The only logical axioms in the complete dependency audit are Lean's standard `propext`, `Classical.choice`, and `Quot.sound`. There are **no new axioms, proof holes, admitted goals, native-evaluation proof oracles, or imported implementation-correctness assumptions**. `grind`, `omega`, rewriting and induction construct kernel-checked proofs; the one `decide` use is a small boundary-policy check, not the substance of the adjoints. The shell/Python dependency audit is bookkeeping, not a premise of any theorem.

## Exact proven claims

### 1. Independent finite-dimensional adjoint infrastructure

`total_congr`, `total_zero`, `total_add`, `total_scale`, `total_swap`, `total_single`, `total_select`, and `dot_basis` establish the finite-sum rules, exchange of two finite sums, fiber selection, and isolation of a coordinate by a basis perturbation.

`matVec_linear` proves linearity for an arbitrary finite matrix. `matrix_adjoint` proves, for every matrix `M`, vector `x` and cotangent `g`:

```text
Σj g[j] (Σk M[j,k] x[k]) = Σk (Σj M[j,k] g[j]) x[k].
```

Neither side is assumed correct by a typeclass or external verifier.

### 2. FiLM routing, parameters, and wrong-branch counterexamples

`filmPrimitiveT` declares the product/shift/clamp STE:

```text
T(dh,dβ,du) = dh + dβ + gamma*(dh/128) + t*mask(u)*du.
```

`film_primitive_adjoint` proves the three pullbacks are:

```text
h: d*(1 + gamma/128)     beta: d     u: d*t*mask(u).
```

For arbitrary output count `m`, PK width `n`, and coefficient matrices `B,G`, define

```text
K[j,k] = B[j,k] + t[j]*mask(u[j])/32 * G[j,k].
```

`film_inputs_decompose`, `film_inputs_linear`, and `film_inputs_adjoint` prove the two-input linear tangent and its full pairing identity:

```text
T(dh,da)[j] = (1+gamma[j]/128)*dh[j] + Σk K[j,k]*da[k]
<d,T(dh,da)> = <d*(1+gamma/128),dh> + <Kᵀd,da>.
```

The PK branch consumes **d**, not the h-branch cotangent. `wrongRoutedPK` explicitly substitutes the latter even while retaining the correct clamp mask. With `G=0` and every gamma `64`, `wrong_route_beta_factor` proves its PK result is **3/2** times the correct result for arbitrary dimensions, `B,t,u,d`. `wrong_route_beta_nonzero` proves inequality whenever the selected correct coordinate is nonzero. `wrong_route_not_adjoint` uses a basis perturbation to prove failure of the **whole finite-dimensional adjoint pairing**, without defining a global objective. `wrong_branch_counterexample` is the unconditional scalar instance: beta-only tangent gives `1`, but reusing `filmH 64 1` gives `3/2`.

`film_params_adjoint` additionally proves the joint beta/gamma **master-parameter** pullback for one output and any finite hidden width. For the tangent

```text
(32*dβbias + Σk a[k]*dB[k])
 + t*mask(u)*(32*dγbias + Σk a[k]*dG[k])/32,
```

the pullback is:

```text
βbias: 32*d          B[k]: d*a[k]
γbias: d*t*mask(u)    G[k]: (d*t*mask(u)/32)*a[k].
```

This distinguishes the gamma producer's bias cancellation of `/32` from its weight and hidden-input coefficients.

### 3. Strict clamp plateaus, including a boundary caveat

`gamma_interior`, `gamma_strict_plateau_mask`, and `gamma_boundary_policy` characterize the declared mask. `clamp_upper_local` constructs positive radius `(u-127)/2` for every `u>127`, and proves the rational clamp unchanged for **every** perturbation in that neighborhood. `clamp_lower_local` does the same with `(-127-u)/2` for `u<-127`. These are genuine local-constancy statements over rational neighborhoods, not merely two sampled values or a real derivative claim.

`clamp_plateau_tangent` proves the declared clamp tangent is zero throughout either strict plateau. `missing_mask_nonzero` proves omission of that mask gives a different cotangent whenever `d*t` is nonzero.

`shifted_upper`, `shifted_lower`, and `shifted_plateau` prove the unbounded-integer `clamp(s / 32)` is constant at the corresponding cap when both sums stay in the conservative strict region `s>4064` or `s<-4064`. Here Lean integer division by positive 32 is floor division. Equivalence to machine shifts, overflow absence, and float-to-integer conversion are not proved.

**Important additional checked counterexample:** `lower_edge_mask_difference` proves

```text
shiftedGamma(-4048) = -127, but gammaMask(-4048/32) = 1.
```

Thus a policy masking by the **observed capped output** is not globally identical to this pre-rounding policy. For integer sums `-4063…-4033`, the observed output is already `-127`, although the unrounded `u` lies strictly inside `(-127,-126)`.

The private GPU candidate masks using `pk_gamma <= -127 || pk_gamma >= 127`; the diagnostic's expected hidden pullback instead tests unrounded `gtot/32.f`. They agree on the tested strict-plateau and gamma-64 fixtures, but passing those fixtures does **not** establish their global equivalence. This counterexample is a policy/specification distinction, not a claim that a new GPU run failed. Choose and document the intended boundary policy before treating the candidate as general conformance.

Finally, `saturated_film_routes` proves strict gamma saturation removes the gamma-producer tangent **but retains beta and h**. `film_h_multiplier_positive` proves `1+gamma/128 > 0` throughout `[-127,127]`. Neither theorem says other network masks or the global cotangent are nonzero.

### 4. Exported bias scale 32 and equivalent directions

`affine32_exact_increment` proves the exact affine increment for `z=32*b+a*w` is `32*db+a*dw`. `affine32_adjoint` proves the master-coordinate pullback `(32*g,a*g)`.

`bias_weight_equivalent` proves, for **every** rational `a,delta,g`, both equal tangent effects and equal directional pullbacks for

```text
(db,dw) = (a*delta/32,0)  versus  (0,delta).
```

`wrong_bias_factors` proves replacing `32*g` by `64*g` or `1*g` breaks that equivalence whenever `a*g≠0`. This is invariant under any common upstream rescaling; it does not need a loss or a choice of centipawn units. A coordinate-dependent preconditioner remains a separate policy, not this VJP.

### 5. Folded shared-base adjoint, including multiple rows

`fold_exact_increment` proves the affine tangent of

```text
phi[i] = s[i]*base[map(i)] + residual[i].
```

`fold_adjoint` proves for arbitrary finite maps and values:

```text
Σi g[i]*(s[i]*db[map(i)] + dr[i])
 = Σb (Σi:map(i)=b s[i]*g[i])*db[b] + Σi g[i]*dr[i].
```

`fold_13_20_adjoint` specializes to the weight-fold scales `1` and `13/20`. The fiber sum is defined explicitly by an equality test; row collisions accumulate, rather than being assumed absent.

`fold_two_rows` proves two residual rows sharing one base contribute `(13/20)*(g0+g1)`. `fold_unscaled_wrong` proves unscaled accumulation differs whenever that sum is nonzero. `fold_secant_units` checks the associated exact affine units: base `+20 → +13`, residual `+20 → +20`. The general adjoint, not this last numeric example, is the substantive proof. The PSQT fold has its own `1/0.5` and export `16` factors and is **not** identified with this weight-fold specialization.

### 6. Equal output coefficients versus deliberate preconditioning

`output_sum_adjoint` proves the cotangents of `out0 + skip + quad` are `(g,g,g)` for arbitrary incoming `g`. A common downstream `600/16384` factor can be included in that seed; it cannot justify unequal local coefficients.

`branchPolicy ps pq` separately maps `(go,gs,gq)` to `(go,ps*gs,pq*gq)`. `historicalBranchPolicy` chooses `ps=1/4`, `pq=3/20`. `policy_weighted_adjoint` proves this reweighted rule is the adjoint of the **different** tangent `do + ps*ds + pq*dq`. `policy_not_sum_pullback` proves it is not the pullback of the equal-coefficient sum for a nonzero seed. `policy_positive_pairing` proves the quadratic pairing with this diagonal branch policy is nonnegative. It is a coherent positive branch-space preconditioner, not thereby an exact derivative of the inference graph, nor a theorem about descent after shared-parameter mixing/Adam.

History matters: `/tmp/asura-strength/history/raw-training-1.md:487–517` contains the deliberate edit replacing exact skip gradients by **0.22** and quad gradients by **0.15**, with a balancing rationale. `/tmp/asura-strength/history/REPORT.md:171` records the subsequent full rewrite at historical calls **L6065/L6066**, source **954–955,1020,1033**, using **0.25/0.15** as seen in current production. These factors must not be erased blindly as accidental calculus mistakes. Specify the correct local VJP first and the chosen branch policy separately. No recommendation to deploy the private candidate or retrain follows from these proofs. Constant coordinate factors can also be partly normalized by Adam; discrepancy ratios are not training-damage or Elo ratios.

## Actual source bridge inspected

Paths below are locations read, **not formal imports**. The full production functions (quantizer/reducer, forward/backward kernel and serializer) were inspected, not just comments or proposed contracts.

Production path: `/home/tom/.cache/asura_spsa/nnue_lab/train_atn1_cuda.cu`

SHA-256: `8d3811c89958782ac172905e02f7101272a41a6f9ee455520e7fdf6bd7dd5d69`.

| Contract | Production locations | Rust counterparts inspected, relative to frozen repository |
|---|---|---|
| Free quantization / folding | `quant_i8`/`quant_i16 :656–664`; `sync_master_to_quantized_kernel :712–769`, especially `:727–731`; reducer `accumulate_base_grads_from_ft_kernel :666–710`, especially unscaled weight addition `:689–697` | `src/nnue.rs:379–466` loader, especially exported FT rows `:399–406`; `src/nnue/accumulator.rs:26–67` row accumulation; `src/nnue.rs:656–678` full refresh. Rust consumes already-folded weights; there is no corresponding Rust training adjoint here. |
| Dense bias units | Export `:757–763`; backward FC3 `:1058–1059`, FC2 `:1077`, FC1 `:1109` | `src/nnue/forward/scalar.rs:18–41,61–66,79–87`; same integer biases/weights in `src/nnue/forward/vectorized.rs:83–109,172–173,206–211` |
| PK beta/gamma producers | Export `:747–751`; PK activation and outputs `:847–886`; direct parameter gradients `:1104–1119` | `src/nnue/accumulator.rs:382–485`, including vector gamma shift/clamp `:456–458` and scalar `:485` |
| FiLM h versus PK inputs | Forward `:924–927`; h branch `:1106`; wrong reuse at `:1131–1134`; hidden interior condition `:1127` | `src/nnue/forward/scalar.rs:44–47`; vectorized `:117–124` |
| Equal output-path coefficients | Forward `:963–967`; skip backward factors `:1101–1102`; quad backward factors `:1167,1180` | Scalar `:49,90`; vectorized `:127,213` |
| Binary export/consumer | `save_quantized_model :1244–1273` | `src/nnue.rs:310–466`, serializer and loader |

The current production direct beta parameter branch already uses post-FiLM `dx1`. The wrong-branch finding concerns reuse of `d_x1_raw` when propagating into **PK hidden** coordinates. Gamma producer parameter and hidden gradients additionally omit the clamp mask. These are different issues; the specification isolates them.

`proofs/search_invariants.lean` was read completely. It contains small compensation/correction/division/capacity contracts with explicit geometry bounds; it does not contain a QAT tangent or CUDA/Rust trainer proof. It is neither imported nor relabeled as evidence for the new contracts. The new file also does not rely on theorem-number comments in production source.

## Empirical IEEE/GPU bridge — evidence already run by the GPU investigator

`/tmp/asura-strength/gpu-contract/REPORT.md` and `CONTRACTS.md` were read fully. The diagnostic source, private candidate patch and logs, fractional fixture, and Rust comparison harness were also inspected. **GPU diagnostics were not rerun here**, because they write to the shared GPU-contract directory. The following results are attributed to those existing artifacts, not to Lean:

| Existing executable evidence | What it bridges |
|---|---|
| `diagnostic.log`: gamma-64, zero-gamma-weight isolation has **113/113** active hidden-lane mismatches, ratio **1.50000036**; candidate has **0/113**, max error `4.76837158e-7` | The wrong-routing counterexample family actually occurs through the instrumented production backward path, isolated from clamp handling. |
| Real sample 12, pre-shift gamma sum **19637**; master gamma bias ±20 leaves gamma **127**, FiLM value **2367**, evaluation **−44**; original gamma-bias gradient **0.433059454**, candidate zero | A strict plateau with large margin and a nonzero unmasked producer cotangent, not an infinitesimal finite difference of integer cp. |
| Original full hidden pullback **113/113** mismatches; candidate **0/113**, max error `2.98023224e-8` | Finite local PK pullback using actual captured registers, on the selected interior hidden lanes. |
| Equivalent perturbations through actual exporter/forward: FC3 weight/bias gradient ratio about **32**, FC2 and FC1 bias/weight ratio about **2**, quad/bias ratio about **4.8**; candidate ratios about **1** | Matching pre-rounding coordinate units/parallel-path effects in the tested exported directions. These are not estimates of Elo damage. No separately isolated direct-skip-ratio experiment was claimed. |
| Reducer probe: incoming **1.25**, original base result **1.25**, expected/candidate **0.8125**. Actual exporter: base +20 gives +13, residual +20 gives +20 | Export scale and its missing reducer factor on a nonzero production probe. Lean additionally handles arbitrary row collisions; the probe alone does not test all fibers. |
| Round trip: **0 / 73,614,400** payload bytes changed. Fractional fixture: **68,350,080** FT/PSQT values checked, **0** mismatches, **61,737,151** nonzero FT weights | Current exporter/serializer on quantized-grid and nondegenerate fractional fixtures. Reconstruction with shared bases zero is not recovery of the historical float master. |
| Two models × **3,072** positions: production GPU versus oracle and Rust scalar/native outputs, **0 mismatches** | Sampled forward/export/consumer consistency, independent of the mathematical adjoint proofs. Sample covers **15/16** stacks, lacks stack 13, and uses optimism/clock zero. |
| Original local-contract aggregate FAIL, candidate PASS; memcheck reports **0 errors** | Executable local regressions and sampled memory-safety evidence, not proof of complete GPU correctness or candidate completeness. |

Float arithmetic is not rational arithmetic: e.g. `0.65f` is approximately `0.6499999761581421`, not exactly `13/20`, and `0.15f` is not exactly `3/20`. Fused multiply-add, nearest-even conversion (`__float2int_rn`), accumulation order/atomic adds, and compiler flags matter. The fractional fixture explicitly compares against fused float and nearest-even host computations. Powers of two and the small integer fixtures avoid some, but not all, numerical ambiguities.

For gradient comparisons, `close_grad` uses

```text
abs(actual-expected) <= 1e-5 * (1 + max(abs(actual),abs(expected))).
```

The hidden-lane loops use the closely related `1e-5*(1+abs(expected))` threshold. The report records stable verdicts across repeats and a fast-math build. These empirical tolerances distinguish the observed large factors/sign changes from low-order float noise. **No formal IEEE error bound, reduction-order bound, compiler correctness theorem, or universal tolerance theorem is proved here.** Near a quantization/clamp boundary, a tiny error may change a discrete result; the boundary counterexample above also shows why a matching fixture cannot settle the STE policy globally.

Reported gamma at a limit: **49,136 / 49,152 observations = 99.96745%**. This is a finite-sample **at-cap count**, not proof that every such observation is strictly outside the unrounded clamp interval, nor that the PK subnet or global network is dead. Beta remains a distinct route. No historical training causality, game-loss attribution, or Elo conclusion is established.

## What remains unverified

- Translation/refinement of these definitions to complete CUDA or Rust programs; actual feature maps and masks on all inputs; shuffle/reduction indexing, races/synchronization, finite-width overflow, SIMD saturation, serialization for all values, or every batch/device/compiler configuration.
- Full quantizer/projection behavior, all activation STE choices, and the output-based versus pre-rounding gamma-mask boundary policy. The candidate is not globally certified by this specification.
- The global surrogate: production logs `p_diff²` at `:1021`, but seeds backward with a custom teacher/cp/focal/Bellman/auxiliary mixture at `:1039–1053`. We treat each incoming local cotangent as given, not as the derivative of that logged loss. Phase/post-scaling, PSQT/draw supervision, optimizer, accumulation over steps, EMA, resume, and model selection are outside the proofs.
- Historical checkpoint provenance: the exact exporting float/EMA state and source/build/run manifest remain unavailable. Current-source defects need not have produced the frozen checkpoint.
- Any claim about chess correctness, search quality, match outcomes, ranking, or strength. No generic algebraic proof is labeled as validation of Rust or chess.

## Exact check command and observed output

The original investigator script and the repository-copy check both passed.
The original source was named `/tmp/asura-strength/lean/LocalQAT.lean`; its contents
are preserved without changes as `proofs/trainer_vjp.lean`.

```sh
lean -t 0 -DwarningAsError=true proofs/trainer_vjp.lean
```

Actual script output:

```text
Lean exit status: 0
Checked dependency reports for all 46 theorems.
Axiom union: Classical.choice, Quot.sound, propext
No proof holes, custom axioms, or native evaluation proof oracle.
PASS: kernel-checked local specification; no implementation/strength certification.
```

Lean output is the 46 complete dependency reports in `lean-check.log`, with **no errors or warnings**. Representative substantive lines:

```text
'LocalQAT.film_inputs_adjoint' depends on axioms: [propext, Classical.choice, Quot.sound]
'LocalQAT.wrong_route_not_adjoint' depends on axioms: [propext, Classical.choice, Quot.sound]
'LocalQAT.fold_13_20_adjoint' depends on axioms: [propext, Classical.choice, Quot.sound]
```

Compiler: `Lean 4.35.0`, commit `eb6cafb7290e8bd5f2937522d8d804f58f02bafc`, x86_64 release. Trust level 0 requests imported-module checking as well as checking the new proof terms. No `--run`, native proof decision procedure, repository build, or generated training executable was used.

`trainer_vjp.lean` SHA-256: `2d76cf0f352df50d6de75ac00bf36034a5d18eccf16fa22d666fda84643611b7`.
