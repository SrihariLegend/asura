import Std

/-!
Independent exact-rational local QAT specification. No implementation is imported.
Rounding/shift tangents are DECLARED STEs, not derivatives of integer evaluation.
Finite sums include every coordinate once; dimensions may be zero. Incoming
cotangents are arbitrary and do not presuppose any global training loss.
-/
namespace LocalQAT

abbrev Vec (n : Nat) := Fin n → Rat

def total : {n : Nat} → Vec n → Rat
  | 0, _ => 0
  | n+1, f => f 0 + total (fun i : Fin n => f i.succ)

def dot {n} (x y : Vec n) : Rat := total (fun i => x i * y i)

theorem total_congr {n} {f g : Vec n} (h : ∀ i, f i = g i) : total f = total g := by
  have : f = g := funext h
  rw [this]

theorem total_zero (n) : total (fun _ : Fin n => (0 : Rat)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [total, ih]; grind

theorem total_add {n} (f g : Vec n) :
    total (fun i => f i + g i) = total f + total g := by
  induction n with
  | zero => simp only [total]; grind
  | succ n ih => simp only [total, ih]; grind

theorem total_scale {n} (c : Rat) (f : Vec n) :
    total (fun i => c * f i) = c * total f := by
  induction n with
  | zero => simp only [total]; grind
  | succ n ih => simp only [total, ih]; grind

theorem total_swap {m n} (f : Fin m → Fin n → Rat) :
    total (fun i => total (f i)) = total (fun j => total (fun i => f i j)) := by
  induction m with
  | zero => simp only [total, total_zero]
  | succ m ih => simp only [total, total_add, ih]

theorem total_single {n} (j : Fin n) (x : Rat) :
    total (fun i => if i = j then x else 0) = x := by
  induction n with
  | zero => exact Fin.elim0 j
  | succ n ih =>
    refine Fin.cases ?_ (fun k => ?_) j
    · simp [total, Fin.succ_ne_zero, total_zero]
      grind
    · simp [total, ih]
      grind

def matVec {m n} (M : Fin m → Vec n) (x : Vec n) : Vec m :=
  fun i => dot (M i) x

def transpose {m n} (M : Fin m → Vec n) (g : Vec m) : Vec n :=
  fun k => total (fun j => M j k * g j)

/-- Arbitrary finite matrix, arbitrary perturbation and cotangent. -/
theorem matrix_adjoint {m n} (M : Fin m → Vec n) (x : Vec n) (g : Vec m) :
    dot g (matVec M x) = dot (transpose M g) x := by
  unfold dot matVec transpose
  simp only [dot, ← total_scale]
  rw [total_swap]
  apply total_congr
  intro k
  rw [Rat.mul_comm, ← total_scale]
  apply total_congr
  intro j
  grind

/-- Rational linearity, independent of the adjoint theorem. -/
theorem matVec_linear {m n} (M : Fin m → Vec n) (x y : Vec n) (c : Rat) :
    matVec M (fun k => c*x k + y k) =
      fun j => c * matVec M x j + matVec M y j := by
  funext j
  unfold matVec dot
  calc
    total (fun k => M j k * (c*x k + y k)) =
        total (fun k => c*(M j k*x k) + M j k*y k) := by
          apply total_congr; intro k; grind
    _ = _ := by rw [total_add, total_scale]

/-! STE primitives. The primal rounding operator is deliberately not assigned a
real derivative. Its declared tangent is identity; shifts declare reciprocal
scale. Saturation is a separate mask, fixed at the primal observation. -/
def roundSTE (dx : Rat) : Rat := dx
def shift5STE (dx : Rat) : Rat := dx / 32
def shift7STE (dx : Rat) : Rat := dx / 128

def clamp127 (x : Rat) : Rat :=
  if x < -127 then -127 else if 127 < x then 127 else x

def gammaMask (u : Rat) : Rat := if -127 < u ∧ u < 127 then 1 else 0

def clampSTE (u du : Rat) : Rat := gammaMask u * du

/-- Zero at the two boundaries is a policy, not a classical derivative claim. -/
theorem gamma_boundary_policy : gammaMask (-127) = 0 ∧ gammaMask 127 = 0 := by
  decide

theorem gamma_interior {u : Rat} (h : -127 < u ∧ u < 127) : gammaMask u = 1 := by
  simp only [gammaMask, ite_eq_left h]

theorem gamma_strict_plateau_mask {u : Rat} (h : 127 < u ∨ u < -127) :
    gammaMask u = 0 := by
  unfold gammaMask
  split <;> grind

/-- A genuine local-constancy statement over rational neighborhoods, not a derivative
of a rounded network: the displayed positive radius is constructed, not assumed. -/
theorem clamp_upper_local {u : Rat} (h : 127 < u) :
    0 < (u-127)/2 ∧ ∀ e : Rat, -(u-127)/2 < e → e < (u-127)/2 →
      clamp127 (u+e) = clamp127 u := by
  unfold clamp127
  grind

theorem clamp_lower_local {u : Rat} (h : u < -127) :
    0 < (-127-u)/2 ∧ ∀ e : Rat, -(-127-u)/2 < e → e < (-127-u)/2 →
      clamp127 (u+e) = clamp127 u := by
  unfold clamp127
  grind

theorem clamp_plateau_tangent {u : Rat} (h : 127 < u ∨ u < -127) (du : Rat) :
    clampSTE u du = 0 := by
  simp only [clampSTE, gamma_strict_plateau_mask h]
  grind

/-- Unbounded-integer analogue of arithmetic shift by 5 followed by clamp.
No machine overflow or instruction equivalence is asserted. -/
def shiftedGamma (s : Int) : Int :=
  let v := s / 32
  if v < -127 then -127 else if 127 < v then 127 else v

theorem shifted_upper {s : Int} (h : 4064 < s) : shiftedGamma s = 127 := by
  simp only [shiftedGamma]
  split <;> (try split) <;> omega

theorem shifted_lower {s : Int} (h : s < -4064) : shiftedGamma s = -127 := by
  simp only [shiftedGamma]
  split <;> (try split) <;> omega

theorem shifted_plateau {s e : Int}
    (h : (4064 < s ∧ 4064 < s+e) ∨ (s < -4064 ∧ s+e < -4064)) :
    shiftedGamma (s+e) = shiftedGamma s := by
  rcases h with h | h
  · rw [shifted_upper h.1, shifted_upper h.2]
  · rw [shifted_lower h.1, shifted_lower h.2]

/-- Boundary-bridge warning: testing the shifted/clamped output for a cap is
NOT always the same STE policy as testing the unrounded preactivation.
This is a deliberate checked counterexample, not a claimed GPU failure. -/
theorem lower_edge_mask_difference :
    shiftedGamma (-4048) = -127 ∧ gammaMask ((-4048 : Rat)/32) = 1 := by
  constructor
  · simp only [shiftedGamma]
    split <;> (try split) <;> omega
  · apply gamma_interior
    constructor <;> grind

/-! FiLM: two input branches, h and PK activations a. t is the OBSERVED h>>7;
gamma is the OBSERVED clamped gamma. t need not equal h/128. The value used
in the product rule is t; the declared tangent of the shift is dh/128.
u is the gamma affine preactivation divided by 32, before clamp/shift rounding. -/
def filmPrimitiveT (gamma t u dh dbeta du : Rat) : Rat :=
  dh + dbeta + gamma * shift7STE dh + t * clampSTE u du

def filmH (gamma d : Rat) : Rat := d * (1 + gamma/128)
def filmBeta (d : Rat) : Rat := d
def filmU (t u d : Rat) : Rat := d * t * gammaMask u

theorem film_primitive_adjoint (gamma t u dh dbeta du d : Rat) :
    d * filmPrimitiveT gamma t u dh dbeta du =
      filmH gamma d * dh + filmBeta d * dbeta + filmU t u d * du := by
  unfold filmPrimitiveT filmH filmBeta filmU shift7STE clampSTE
  grind

/-- PK input coefficient after BOTH producers beta and gamma. -/
def pkMatrix {m n} (B G : Fin m → Vec n) (t u : Vec m) : Fin m → Vec n :=
  fun j k => B j k + t j * gammaMask (u j) / 32 * G j k

def filmInputsT {m n} (B G : Fin m → Vec n) (gamma t u dh : Vec m) (da : Vec n) : Vec m :=
  fun j => filmPrimitiveT (gamma j) (t j) (u j) (dh j)
    (dot (B j) da) (shift5STE (dot (G j) da))

def filmInputsH {m} (gamma d : Vec m) : Vec m := fun j => filmH (gamma j) (d j)
def filmInputsPK {m n} (B G : Fin m → Vec n) (t u d : Vec m) : Vec n :=
  transpose (pkMatrix B G t u) d

theorem film_inputs_decompose {m n} (B G : Fin m → Vec n)
    (gamma t u dh : Vec m) (da : Vec n) (j : Fin m) :
    filmInputsT B G gamma t u dh da j =
      (1 + gamma j / 128) * dh j + matVec (pkMatrix B G t u) da j := by
  unfold filmInputsT filmPrimitiveT shift7STE clampSTE shift5STE matVec pkMatrix dot
  have h := total_congr (fun k : Fin n =>
    show (B j k + t j * gammaMask (u j) / 32 * G j k) * da k =
      B j k * da k + (t j * gammaMask (u j) / 32) * (G j k * da k) by grind)
  rw [h, total_add, total_scale]
  grind

/-- Independent finite-dimensional adjoint for all outputs/hidden lanes at once.
The h cotangent does NOT get reused as the PK cotangent. -/
theorem film_inputs_adjoint {m n} (B G : Fin m → Vec n)
    (gamma t u dh d : Vec m) (da : Vec n) :
    dot d (filmInputsT B G gamma t u dh da) =
      dot (filmInputsH gamma d) dh + dot (filmInputsPK B G t u d) da := by
  unfold dot
  have h := total_congr (fun j : Fin m =>
    show d j * filmInputsT B G gamma t u dh da j =
      filmInputsH gamma d j * dh j + d j * matVec (pkMatrix B G t u) da j by
        rw [film_inputs_decompose]; unfold filmInputsH filmH; grind)
  rw [h, total_add]
  exact congrArg (fun x => total (fun j => filmInputsH gamma d j * dh j) + x)
    (matrix_adjoint (pkMatrix B G t u) da d)

/-- Linearity of the declared two-input tangent; all primal registers are fixed. -/
theorem film_inputs_linear {m n} (B G : Fin m → Vec n)
    (gamma t u dh eh : Vec m) (da ea : Vec n) (c : Rat) :
    filmInputsT B G gamma t u (fun j => c*dh j + eh j) (fun k => c*da k + ea k) =
      fun j => c * filmInputsT B G gamma t u dh da j + filmInputsT B G gamma t u eh ea j := by
  funext j
  simp only [film_inputs_decompose, matVec_linear]
  grind

/-- Saturating gamma suppresses the gamma producer, NOT beta or h. -/
theorem saturated_film_routes {u : Rat} (h : 127 < u ∨ u < -127)
    (gamma t dh dbeta du : Rat) :
    filmPrimitiveT gamma t u dh dbeta du = (1+gamma/128)*dh + dbeta := by
  unfold filmPrimitiveT shift7STE
  rw [clamp_plateau_tangent h]
  grind

/-- The h-branch multiplier is strictly positive throughout the allowed gamma range.
This is a local fact, not a claim about the rest of a network or its loss seed. -/
theorem film_h_multiplier_positive {gamma : Rat} (h : -127 ≤ gamma ∧ gamma ≤ 127) :
    0 < 1+gamma/128 := by grind

/-- Isolate the routing error while giving the wrong map the CORRECT mask. -/
def wrongRoutedPK {m n} (B G : Fin m → Vec n) (gamma t u d : Vec m) : Vec n :=
  transpose (pkMatrix B G t u) (filmInputsH gamma d)

/-- General 1.5 factor: arbitrary dimensions, B, t, u, d; G=0, gamma=64.
It cannot be excused by gamma saturation or by the choice of global loss. -/
theorem wrong_route_beta_factor {m n} (B : Fin m → Vec n) (t u d : Vec m) (k : Fin n) :
    wrongRoutedPK B (fun _ _ => 0) (fun _ => 64) t u d k =
      (3/2 : Rat) * filmInputsPK B (fun _ _ => 0) t u d k := by
  unfold wrongRoutedPK filmInputsPK transpose filmInputsH filmH pkMatrix
  rw [← total_scale]
  apply total_congr
  intro j
  grind

theorem wrong_route_beta_nonzero {m n} (B : Fin m → Vec n) (t u d : Vec m) (k : Fin n)
    (h : filmInputsPK B (fun _ _ => 0) t u d k ≠ 0) :
    wrongRoutedPK B (fun _ _ => 0) (fun _ => 64) t u d k ≠
      filmInputsPK B (fun _ _ => 0) t u d k := by
  rw [wrong_route_beta_factor]
  grind

/-- Explicit wrong-branch witness without a global objective. -/
theorem wrong_branch_counterexample :
    (1 : Rat) * filmPrimitiveT 64 2 64 0 1 0 ≠ filmH 64 1 * 1 := by
  unfold filmPrimitiveT filmH shift7STE clampSTE
  grind

/-- Missing mask gives a false nonzero gamma cotangent on ANY strict plateau,
provided the actual incoming product d*t is nonzero (not a desired-result premise). -/
theorem missing_mask_nonzero {u : Rat} (h : 127 < u ∨ u < -127)
    (d t : Rat) (hn : d*t ≠ 0) : filmU t u d = 0 ∧ d*t ≠ filmU t u d := by
  unfold filmU
  rw [gamma_strict_plateau_mask h]
  constructor <;> grind

/-- Evaluation on a basis perturbation isolates a cotangent coordinate. -/
theorem dot_basis {n} (x : Vec n) (k : Fin n) :
    dot x (fun i => if i = k then 1 else 0) = x k := by
  unfold dot
  calc
    _ = total (fun i => if i = k then x k else 0) := by
      apply total_congr
      intro i
      by_cases h : i = k
      · subst i; simp
      · simp only [ite_eq_right h]; grind
    _ = _ := total_single k (x k)

/-- The wrong routing fails the full finite-dimensional adjoint pairing on a
basis direction, not merely an isolated displayed coefficient. -/
theorem wrong_route_not_adjoint {m n} (B : Fin m → Vec n) (t u d : Vec m) (k : Fin n)
    (h : filmInputsPK B (fun _ _ => 0) t u d k ≠ 0) :
    dot d (filmInputsT B (fun _ _ => 0) (fun _ => 64) t u (fun _ => 0)
      (fun i => if i = k then 1 else 0)) ≠
    dot (filmInputsH (fun _ => 64) d) (fun _ => 0) +
      dot (wrongRoutedPK B (fun _ _ => 0) (fun _ => 64) t u d)
        (fun i => if i = k then 1 else 0) := by
  rw [film_inputs_adjoint, dot_basis, dot_basis]
  have := wrong_route_beta_nonzero B t u d k h
  grind

/-! Scale-32 exported bias and weight directions, in master coordinates.
This affine map is before rounding and assumes a free/unsaturated weight STE. -/
def affine32 (a b w : Rat) : Rat := 32*b + a*w
def affine32T (a db dw : Rat) : Rat := 32*roundSTE db + a*roundSTE dw
def biasPB (g : Rat) : Rat := 32*g
def weightPB (a g : Rat) : Rat := a*g

theorem affine32_exact_increment (a b w db dw : Rat) :
    affine32 a (b+db) (w+dw) - affine32 a b w = affine32T a db dw := by
  unfold affine32 affine32T roundSTE
  grind

theorem affine32_adjoint (a db dw g : Rat) :
    g * affine32T a db dw = biasPB g * db + weightPB a g * dw := by
  unfold affine32T roundSTE biasPB weightPB
  grind

/-- Allows the diagnostic's delta=16 as well as delta=1. -/
theorem bias_weight_equivalent (a delta g : Rat) :
    affine32T a (a*delta/32) 0 = affine32T a 0 delta ∧
    biasPB g * (a*delta/32) = weightPB a g * delta := by
  unfold affine32T roundSTE biasPB weightPB
  constructor <;> grind

/-- Unit mistakes 64 and 1, for every nonzero a*g, not just a numeric fixture. -/
theorem wrong_bias_factors (a g : Rat) (h : a*g ≠ 0) :
    (64*g)*(a/32) ≠ weightPB a g ∧ (1*g)*(a/32) ≠ weightPB a g := by
  unfold weightPB
  constructor <;> grind

/-- Joint beta/gamma producer parameter tangent for one FiLM output, with
arbitrary finite hidden activations. Both MASTER biases export with scale 32.
The gamma shift cancels that 32 only for its bias, not for its weights. -/
def filmParamsT {n} (a : Vec n) (t u dbeta dgamma : Rat) (dB dG : Vec n) : Rat :=
  (32*dbeta + dot a dB) + t * clampSTE u (shift5STE (32*dgamma + dot a dG))

theorem film_params_adjoint {n} (a dB dG : Vec n) (t u dbeta dgamma d : Rat) :
    d * filmParamsT a t u dbeta dgamma dB dG =
      (32*d)*dbeta + dot (fun k => d*a k) dB +
      (d*t*gammaMask u)*dgamma + dot (fun k => (d*t*gammaMask u/32)*a k) dG := by
  unfold filmParamsT clampSTE shift5STE dot
  have hb : total (fun k => d*a k*dB k) = d * total (fun k => a k*dB k) := by
    rw [← total_scale]
    apply total_congr; intro k; grind
  have hg : total (fun k => (d*t*gammaMask u/32)*a k*dG k) =
      (d*t*gammaMask u/32) * total (fun k => a k*dG k) := by
    rw [← total_scale]
    apply total_congr; intro k; grind
  rw [hb, hg]
  grind

/-! Shared-base fold: every residual row has a map into a finite base set.
No injectivity, surjectivity, or unique use is assumed. Aliasing is the point. -/
def foldMap {r b} (map : Fin r → Fin b) (s : Vec r) (base : Vec b) (res : Vec r) : Vec r :=
  fun i => s i * base (map i) + res i

def foldBasePB {r b} (map : Fin r → Fin b) (s g : Vec r) : Vec b :=
  fun k => total (fun i => if map i = k then s i * g i else 0)

theorem total_select {b} (k : Fin b) (c : Rat) (x : Vec b) :
    total (fun j => (if k = j then c else 0) * x j) = c * x k := by
  calc
    _ = total (fun j => if j = k then c * x k else 0) := by
      apply total_congr
      intro j
      by_cases h : j = k
      · subst j; simp
      · have h' : k ≠ j := Ne.symm h
        simp only [ite_eq_right h, ite_eq_right h']; grind
    _ = _ := total_single k (c*x k)

/-- Transpose identity with genuine accumulation across all rows in each fiber. -/
theorem fold_adjoint {r b} (map : Fin r → Fin b) (s g dr : Vec r) (db : Vec b) :
    dot g (foldMap map s db dr) = dot (foldBasePB map s g) db + dot g dr := by
  have base : total (fun i => g i * (s i * db (map i))) =
      dot (foldBasePB map s g) db := by
    unfold dot foldBasePB
    calc
      _ = total (fun i => total (fun k => (if map i = k then s i*g i else 0)*db k)) := by
        apply total_congr
        intro i
        rw [total_select]
        grind
      _ = _ := by
        rw [total_swap]
        apply total_congr
        intro k
        rw [Rat.mul_comm, ← total_scale]
        apply total_congr
        intro i
        grind
  unfold dot foldMap
  calc
    _ = total (fun i => g i*(s i*db (map i)) + g i*dr i) := by
      apply total_congr; intro i; grind
    _ = _ := by rw [total_add, base]; rfl

theorem fold_exact_increment {r b} (map : Fin r → Fin b) (s : Vec r)
    (base db : Vec b) (res dr : Vec r) (i : Fin r) :
    foldMap map s (fun k => base k+db k) (fun k => res k+dr k) i -
      foldMap map s base res i = foldMap map s db dr i := by
  unfold foldMap
  grind

def rowScale {r} (isPSQ : Fin r → Bool) : Vec r := fun i => if isPSQ i then 1 else 13/20

/-- The production weight-fold scale specialization, not the separate PSQT map. -/
theorem fold_13_20_adjoint {r b} (map : Fin r → Fin b) (isPSQ : Fin r → Bool)
    (g dr : Vec r) (db : Vec b) :
    dot g (foldMap map (rowScale isPSQ) db dr) =
      dot (foldBasePB map (rowScale isPSQ) g) db + dot g dr :=
  fold_adjoint map (rowScale isPSQ) g dr db

/-- Two distinct residual rows share one base: both contributions are required. -/
theorem fold_two_rows (g0 g1 : Rat) :
    foldBasePB (fun _ : Fin 2 => (0 : Fin 1)) (fun _ => 13/20)
      (fun i => if i = 0 then g0 else g1) 0 = (13/20 : Rat)*(g0+g1) := by
  simp [foldBasePB, total]
  grind

theorem fold_unscaled_wrong (g0 g1 : Rat) (h : g0+g1 ≠ 0) :
    g0+g1 ≠ foldBasePB (fun _ : Fin 2 => (0 : Fin 1)) (fun _ => 13/20)
      (fun i => if i = 0 then g0 else g1) 0 := by
  rw [fold_two_rows]
  grind

theorem fold_secant_units : (13/20 : Rat)*20 = 13 ∧ (1 : Rat)*20 = 20 := by grind

/-! Equal-coefficient output sum versus an explicit branch policy.
The policy is NOT silently identified with the mathematical pullback. -/
def outputSum (o s q : Rat) : Rat := o+s+q
def outputPB (g : Rat) : Rat × Rat × Rat := (g,g,g)

def branchPolicy (ps pq : Rat) (v : Rat × Rat × Rat) : Rat × Rat × Rat :=
  (v.1, ps*v.2.1, pq*v.2.2)

def historicalBranchPolicy := branchPolicy (1/4) (3/20)

theorem output_sum_adjoint (g do_ ds dq : Rat) :
    g * outputSum do_ ds dq =
      (outputPB g).1*do_ + (outputPB g).2.1*ds + (outputPB g).2.2*dq := by
  unfold outputSum outputPB
  grind

/-- The reweighted branch rule IS the adjoint of a DIFFERENT tangent map. -/
theorem policy_weighted_adjoint (ps pq g do_ ds dq : Rat) :
    g * (do_ + ps*ds + pq*dq) =
      (branchPolicy ps pq (outputPB g)).1*do_ +
      (branchPolicy ps pq (outputPB g)).2.1*ds +
      (branchPolicy ps pq (outputPB g)).2.2*dq := by
  unfold branchPolicy outputPB
  grind

theorem policy_not_sum_pullback (g : Rat) (h : g ≠ 0) :
    (historicalBranchPolicy (outputPB g)).2.1 ≠ g ∧
    (historicalBranchPolicy (outputPB g)).2.2 ≠ g := by
  unfold historicalBranchPolicy branchPolicy outputPB
  constructor <;> grind

/-- Positive branch factors are a coherent preconditioner in branch space.
This proves no statement about Adam or the mixed parameter-space update. -/
theorem policy_positive_pairing (o s q : Rat) :
    0 ≤ o*o + s*((historicalBranchPolicy (o,s,q)).2.1) +
      q*((historicalBranchPolicy (o,s,q)).2.2) := by
  unfold historicalBranchPolicy branchPolicy
  have square (x : Rat) : 0 ≤ x*x := by
    by_cases h : 0 ≤ x
    · exact Rat.mul_nonneg h h
    · have hn : 0 ≤ -x := by grind
      have := Rat.mul_nonneg hn hn
      grind
  have := square o
  have := square s
  have := square q
  grind

end LocalQAT

-- Kernel dependency audit: no custom axioms or executable-evaluation oracle.
#print axioms LocalQAT.total_congr
#print axioms LocalQAT.total_zero
#print axioms LocalQAT.total_add
#print axioms LocalQAT.total_scale
#print axioms LocalQAT.total_swap
#print axioms LocalQAT.total_single
#print axioms LocalQAT.matrix_adjoint
#print axioms LocalQAT.matVec_linear
#print axioms LocalQAT.gamma_boundary_policy
#print axioms LocalQAT.gamma_interior
#print axioms LocalQAT.gamma_strict_plateau_mask
#print axioms LocalQAT.clamp_upper_local
#print axioms LocalQAT.clamp_lower_local
#print axioms LocalQAT.clamp_plateau_tangent
#print axioms LocalQAT.shifted_upper
#print axioms LocalQAT.shifted_lower
#print axioms LocalQAT.shifted_plateau
#print axioms LocalQAT.lower_edge_mask_difference
#print axioms LocalQAT.film_primitive_adjoint
#print axioms LocalQAT.film_inputs_decompose
#print axioms LocalQAT.film_inputs_adjoint
#print axioms LocalQAT.film_inputs_linear
#print axioms LocalQAT.saturated_film_routes
#print axioms LocalQAT.film_h_multiplier_positive
#print axioms LocalQAT.wrong_route_beta_factor
#print axioms LocalQAT.wrong_route_beta_nonzero
#print axioms LocalQAT.wrong_branch_counterexample
#print axioms LocalQAT.missing_mask_nonzero
#print axioms LocalQAT.dot_basis
#print axioms LocalQAT.wrong_route_not_adjoint
#print axioms LocalQAT.affine32_exact_increment
#print axioms LocalQAT.affine32_adjoint
#print axioms LocalQAT.bias_weight_equivalent
#print axioms LocalQAT.wrong_bias_factors
#print axioms LocalQAT.film_params_adjoint
#print axioms LocalQAT.total_select
#print axioms LocalQAT.fold_adjoint
#print axioms LocalQAT.fold_exact_increment
#print axioms LocalQAT.fold_13_20_adjoint
#print axioms LocalQAT.fold_two_rows
#print axioms LocalQAT.fold_unscaled_wrong
#print axioms LocalQAT.fold_secant_units
#print axioms LocalQAT.output_sum_adjoint
#print axioms LocalQAT.policy_weighted_adjoint
#print axioms LocalQAT.policy_not_sum_pullback
#print axioms LocalQAT.policy_positive_pairing
