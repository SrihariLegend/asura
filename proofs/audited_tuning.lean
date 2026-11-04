import Std

/-! Exact finite probability design for one fixed candidate and a frozen cohort.
No claim that timed chess is deterministic: outcomes and predictors are parameters.
Only the independent uniform-with-replacement design is implemented here. -/
namespace AuditedTuning

abbrev Vec (n : Nat) := Fin n → Rat

def total : {n : Nat} → Vec n → Rat
  | 0, _ => 0
  | n+1, f => f 0 + total (fun i : Fin n => f i.succ)

theorem total_congr {n} {f g : Vec n} (h : ∀ i, f i = g i) : total f = total g := by
  rw [funext h]

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

theorem total_const (n : Nat) (c : Rat) : total (fun _ : Fin n => c) = n*c := by
  induction n with
  | zero => simp [total]
  | succ n ih => simp only [total, ih, Rat.natCast_add, Rat.natCast_ofNat]; grind

theorem total_nonneg {n} (f : Vec n) (h : ∀ i, 0 ≤ f i) : 0 ≤ total f := by
  induction n with
  | zero => simp [total]
  | succ n ih =>
    have h0 := h 0
    have ht := ih (fun i => f i.succ) (fun i => h i.succ)
    simp only [total]; grind

def avg {n} (f : Vec n) : Rat := (1 / (n : Rat)) * total f

theorem nat_pos {n : Nat} (h : 0 < n) : (0 : Rat) < n := by
  exact Rat.natCast_pos.mpr h

theorem avg_congr {n} {f g : Vec n} (h : ∀ i, f i = g i) : avg f = avg g := by
  unfold avg; rw [total_congr h]

theorem avg_add {n} (f g : Vec n) : avg (fun i => f i+g i) = avg f+avg g := by
  unfold avg; rw [total_add]; grind

theorem avg_scale {n} (c : Rat) (f : Vec n) : avg (fun i => c*f i) = c*avg f := by
  unfold avg; rw [total_scale]; grind

theorem avg_const {n} (hn : 0 < n) (c : Rat) : avg (fun _ : Fin n => c) = c := by
  have hp := nat_pos hn
  unfold avg; rw [total_const]
  have hc := Rat.div_mul_cancel (a := (1 : Rat)) (Rat.ne_of_gt hp)
  grind

theorem avg_sub {n} (f g : Vec n) : avg (fun i => f i-g i) = avg f-avg g := by
  have he : (fun i => f i-g i) = (fun i => f i + (-1)*g i) := by funext i; grind
  rw [he, avg_add, avg_scale]; grind

theorem avg_nonneg {n} (hn : 0 < n) (f : Vec n) (h : ∀ i, 0 ≤ f i) : 0 ≤ avg f := by
  have hp := nat_pos hn
  have hi : 0 ≤ (1 : Rat) / n := by
    have := Rat.inv_pos.mpr hp
    have := Rat.le_of_lt this
    grind [Rat.div_def]
  exact Rat.mul_nonneg hi (total_nonneg f h)

/-- Each outcome is an ordered k-tuple; repetitions are retained. -/
def Samples (n : Nat) : Nat → Type
  | 0 => Unit
  | k+1 => Fin n × Samples n k

/-- Actual iterated uniform averaging, not an unbiasedness hypothesis.
At each node each of the n branches has weight 1/n. -/
def expect {n : Nat} : {k : Nat} → (Samples n k → Rat) → Rat
  | 0, f => f ()
  | k+1, f => avg (fun i => expect (fun t : Samples n k => f (i,t)))

theorem expect_congr {n k} {f g : Samples n k → Rat} (h : ∀ t, f t = g t) :
    expect f = expect g := by rw [funext h]

theorem expect_const {n} (hn : 0 < n) (k : Nat) (c : Rat) :
    expect (fun _ : Samples n k => c) = c := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [expect, ih, avg_const hn]

theorem expect_add {n k} (f g : Samples n k → Rat) :
    expect (fun t => f t + g t) = expect f + expect g := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [expect, ih, avg_add]

theorem expect_scale {n k} (c : Rat) (f : Samples n k → Rat) :
    expect (fun t => c*f t) = c*expect f := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [expect, ih, avg_scale]

theorem expect_sub {n k} (f g : Samples n k → Rat) :
    expect (fun t => f t-g t) = expect f-expect g := by
  have he : (fun t => f t-g t) = (fun t => f t+(-1)*g t) := by funext t; grind
  rw [he, expect_add, expect_scale]; grind

theorem expect_nonneg {n k} (hn : 0 < n) (f : Samples n k → Rat)
    (h : ∀ t, 0 ≤ f t) : 0 ≤ expect f := by
  induction k with
  | zero => exact h ()
  | succ k ih => exact avg_nonneg hn _ (fun i => ih _ (fun t => h (i,t)))

/-- Unweighted enumeration of every ordered tuple, including repeated indices. -/
def sampleTotal {n : Nat} : {k : Nat} → (Samples n k → Rat) → Rat
  | 0, f => f ()
  | k+1, f => total (fun i => sampleTotal (fun t : Samples n k => f (i,t)))

/-- Explicit mass function: every tuple has weight (1/N)^k. -/
theorem actual_distribution_formula {n k} (f : Samples n k → Rat) :
    expect f = (1/(n : Rat))^k * sampleTotal f := by
  induction k with
  | zero => simp only [expect, sampleTotal, Rat.pow_zero]; grind
  | succ k ih =>
    simp only [expect, sampleTotal, ih, avg, total_scale, Rat.pow_succ]
    grind

theorem tuple_mass_positive {n : Nat} (hn : 0 < n) (k : Nat) :
    0 < (1/(n : Rat))^k := by
  have hi := Rat.inv_pos.mpr (nat_pos hn)
  have hd : (1 : Rat)/n = (n : Rat)⁻¹ := by grind [Rat.div_def]
  rw [hd]
  exact Rat.pow_pos hi

/-- Normalization and positivity certify a genuine finite probability average. -/
theorem probability_normalized {n} (hn : 0 < n) (k : Nat) :
    expect (fun _ : Samples n k => (1 : Rat)) = 1 := expect_const hn k 1

def residual {n} (y p : Vec n) : Vec n := fun i => y i-p i

def energy {n} (v : Vec n) := avg (fun i => v i*v i)
def center {n} (v : Vec n) : Vec n := fun i => v i-avg v
def variance {n} (v : Vec n) := energy (center v)
def randomVariance {n k} (f : Samples n k → Rat) :=
  expect (fun t => (f t-expect f)*(f t-expect f))

def drawSum {n} (v : Vec n) : {k : Nat} → Samples n k → Rat
  | 0, _ => 0
  | _k+1, t => v t.1 + drawSum v t.2

def estimator {n k} (y p : Vec n) (t : Samples n k) :=
  avg p + (1/(k : Rat))*drawSum (residual y p) t

theorem residual_mean {n} (y p : Vec n) : avg (residual y p) = avg y-avg p :=
  avg_sub y p

theorem center_mean {n} (hn : 0 < n) (v : Vec n) : avg (center v) = 0 := by
  unfold center; rw [avg_sub, avg_const hn]; grind

theorem square_nonneg (x : Rat) : 0 ≤ x*x := by
  by_cases h : 0 ≤ x
  · exact Rat.mul_nonneg h h
  · have hn : 0 ≤ -x := by grind
    have := Rat.mul_nonneg hn hn
    grind

theorem variance_nonneg {n} (hn : 0 < n) (v : Vec n) : 0 ≤ variance v :=
  avg_nonneg hn _ (fun i => square_nonneg (center v i))

theorem drawSum_expect {n} (hn : 0 < n) (v : Vec n) (k : Nat) :
    expect (fun t : Samples n k => drawSum v t) = (k : Rat)*avg v := by
  induction k with
  | zero => simp [expect, drawSum]
  | succ k ih =>
    simp only [expect, drawSum, expect_add, expect_const hn, ih,
      avg_add, avg_const hn, Rat.natCast_add, Rat.natCast_ofNat]
    grind

/-- Arbitrary predictors fixed before the k independent draws; no accuracy premise. -/
theorem estimator_unbiased {n k} (hn : 0 < n) (hk : 0 < k) (y p : Vec n) :
    expect (estimator (k := k) y p) = avg y := by
  unfold estimator
  rw [expect_add, expect_const hn, expect_scale, drawSum_expect hn, residual_mean]
  have hc := Rat.div_mul_cancel (a := (1 : Rat)) (Rat.ne_of_gt (nat_pos hk))
  grind

/-- The centered k-sample sum has no cross contribution, by product averaging. -/
theorem centered_sum_energy {n} (hn : 0 < n) (v : Vec n)
    (hz : avg v = 0) (k : Nat) :
    expect (fun t : Samples n k => drawSum v t * drawSum v t) = (k : Rat)*energy v := by
  induction k with
  | zero => simp [expect, drawSum]
  | succ k ih =>
    simp only [expect, drawSum]
    have inner (i : Fin n) :
        expect (fun t : Samples n k => (v i+drawSum v t)*(v i+drawSum v t)) =
          v i*v i + (k : Rat)*energy v := by
      calc
        _ = expect (fun t : Samples n k =>
            v i*v i + (2*v i)*drawSum v t + drawSum v t*drawSum v t) := by
              apply expect_congr; intro t; grind
        _ = _ := by
          rw [expect_add, expect_add, expect_const hn, expect_scale, drawSum_expect hn, ih, hz]
          grind
    simp only [inner, avg_add, avg_const hn]
    unfold energy
    simp only [Rat.natCast_add, Rat.natCast_ofNat]
    grind

theorem drawSum_center {n} (v : Vec n) (k : Nat) (t : Samples n k) :
    drawSum (center v) t = drawSum v t - (k : Rat)*avg v := by
  induction k with
  | zero => simp [drawSum]; grind
  | succ k ih =>
    simp only [drawSum, center, ih, Rat.natCast_add, Rat.natCast_ofNat]
    grind

/-- Exact variance of the actual iid-with-replacement estimator. -/
theorem estimator_variance {n k} (hn : 0 < n) (hk : 0 < k) (y p : Vec n) :
    randomVariance (estimator (k := k) y p) = variance (residual y p)/(k : Rat) := by
  unfold randomVariance
  rw [estimator_unbiased hn hk]
  have hc := Rat.div_mul_cancel (a := (1 : Rat)) (Rat.ne_of_gt (nat_pos hk))
  have point (t : Samples n k) :
      (estimator y p t-avg y)*(estimator y p t-avg y) =
      ((1/(k : Rat))*(1/(k : Rat))) *
        (drawSum (center (residual y p)) t * drawSum (center (residual y p)) t) := by
    rw [drawSum_center, residual_mean]
    unfold estimator
    grind
  rw [expect_congr point, expect_scale, centered_sum_energy hn _ (center_mean hn _)]
  unfold variance
  grind [Rat.div_def]

theorem one_draw_variance {n} (hn : 0 < n) (y p : Vec n) :
    randomVariance (estimator (k := 1) y p) = variance (residual y p) := by
  rw [estimator_variance hn (by decide)]
  grind

/-- A two-opening cohort: predicting zero and auditing only the positive opening
reports 1, whereas the fixed target is 0. Even the score bound is satisfied. -/
def badScores : Vec 2 := fun i => if i = 0 then 1 else -1
def zeroPredictor : Vec 2 := fun _ => 0

theorem selective_audit_counterexample :
    avg badScores = 0 ∧ avg zeroPredictor + residual badScores zeroPredictor 0 = 1 ∧
    avg zeroPredictor + residual badScores zeroPredictor 0 ≠ avg badScores := by
  simp [avg, total, badScores, zeroPredictor, residual]
  grind

theorem counterexample_bounded : ∀ i, -1 ≤ badScores i ∧ badScores i ≤ 1 := by
  intro i; unfold badScores; split <;> decide

theorem predictor_only_counterexample :
    avg (fun _ : Fin 2 => (1 : Rat)) = 1 ∧
    avg (fun _ : Fin 2 => (1 : Rat)) ≠ avg badScores := by
  rw [avg_const (by decide)]
  have h := selective_audit_counterexample
  grind

/-- Random but nonuniform unweighted correction is biased even with full support. -/
def skewWeights : Vec 2 := fun i => if i = 0 then 3/4 else 1/4

theorem skew_weights_valid :
    (∀ i, 0 < skewWeights i) ∧ total skewWeights = 1 := by
  constructor
  · intro i; unfold skewWeights; split <;> grind
  · simp [total, skewWeights]; grind

theorem nonuniform_uncorrected_counterexample :
    total (fun i => skewWeights i * residual badScores zeroPredictor i) = (1/2 : Rat) ∧
    total (fun i => skewWeights i * residual badScores zeroPredictor i) ≠ avg badScores := by
  simp [total, skewWeights, residual, badScores, zeroPredictor, avg]
  grind

/-- A frozen conditional cohort for one candidate. Each coordinate stands for
one already-frozen color-reversed opening-pair result, NOT one random timed run.
Predictors, as fields of this object, do not depend on the subsequent tuple. -/
structure FrozenCohort where
  size : Nat
  positive : 0 < size
  score : Vec size
  prediction : Vec size
  bounded : ∀ i, -1 ≤ score i ∧ score i ≤ 1

theorem frozen_cohort_unbiased (c : FrozenCohort) {k : Nat} (hk : 0 < k) :
    expect (estimator (k := k) c.score c.prediction) = avg c.score :=
  estimator_unbiased c.positive hk c.score c.prediction

theorem frozen_cohort_variance (c : FrozenCohort) {k : Nat} (hk : 0 < k) :
    randomVariance (estimator (k := k) c.score c.prediction) =
      variance (residual c.score c.prediction)/(k : Rat) :=
  estimator_variance c.positive hk c.score c.prediction

/-- Centered covariance, with the same population (denominator N) convention. -/
def covariance {n} (y x : Vec n) := avg (fun i => center y i * center x i)
def linearResidual {n} (y x : Vec n) (beta : Rat) : Vec n := fun i => y i-beta*x i

theorem center_linear {n} (y x : Vec n) (beta : Rat) (i : Fin n) :
    center (linearResidual y x beta) i = center y i-beta*center x i := by
  unfold center linearResidual
  rw [avg_sub, avg_scale]
  grind

theorem control_variance_quadratic {n} (y x : Vec n) (beta : Rat) :
    variance (linearResidual y x beta) =
      variance y - 2*beta*covariance y x + beta*beta*variance x := by
  unfold variance energy
  simp only [center_linear]
  calc
    _ = avg (fun i => (center y i*center y i +
          (-2*beta)*(center y i*center x i)) +
          (beta*beta)*(center x i*center x i)) := by
            apply avg_congr; intro i; grind
    _ = _ := by rw [avg_add, avg_add, avg_scale, avg_scale]; unfold covariance; grind

/-- Exact completion of the square, including the optimal fixed coefficient. -/
theorem control_variance_identity {n} (y x : Vec n) (hx : variance x ≠ 0) (beta : Rat) :
    variance (linearResidual y x beta) =
      variance y - covariance y x*covariance y x/variance x +
      variance x*(beta-covariance y x/variance x)*(beta-covariance y x/variance x) := by
  rw [control_variance_quadratic]
  have hc := Rat.div_mul_cancel (a := covariance y x) hx
  have hd := Rat.div_mul_cancel (a := covariance y x*covariance y x) hx
  grind

theorem optimal_control_variance {n} (y x : Vec n) (hx : variance x ≠ 0) :
    variance (linearResidual y x (covariance y x/variance x)) =
      variance y - covariance y x*covariance y x/variance x := by
  rw [control_variance_identity y x hx]
  grind

theorem fixed_control_optimal {n} (y x : Vec n)
    (hx : 0 < variance x) (beta : Rat) :
    variance (linearResidual y x (covariance y x/variance x)) ≤
      variance (linearResidual y x beta) := by
  rw [optimal_control_variance y x (Rat.ne_of_gt hx),
    control_variance_identity y x (Rat.ne_of_gt hx)]
  have h := Rat.mul_nonneg (Rat.le_of_lt hx)
    (square_nonneg (beta-covariance y x/variance x))
  grind

/-- Orientation matters: n >= s*k and residual/k <= true/n imply the strong
necessary residual <= true/s condition, not the reverse inequality. Counts are
rational here so this also applies to integer counts via casts. -/
theorem sample_reduction_necessary (vr vy n k s : Rat)
    (hr : 0 ≤ vr) (hk : 0 < k) (hs : 0 < s)
    (hbudget : s*k ≤ n) (hmatch : vr/k ≤ vy/n) : vr ≤ vy/s := by
  have hsk := Rat.mul_pos hs hk
  have hn : 0 < n := by grind
  have hkn : 0 ≤ k*n := Rat.mul_nonneg (Rat.le_of_lt hk) (Rat.le_of_lt hn)
  have hm := Rat.mul_le_mul_of_nonneg_right hmatch hkn
  have hb := Rat.mul_le_mul_of_nonneg_left hbudget hr
  have cr := Rat.div_mul_cancel (a := vr) (Rat.ne_of_gt hk)
  have cy := Rat.div_mul_cancel (a := vy) (Rat.ne_of_gt hn)
  have er : vr/k*(k*n) = vr*n := by grind
  have ey : vy/n*(k*n) = vy*k := by grind
  have eb : vr*(s*k) = (vr*s)*k := by grind
  rw [er, ey] at hm
  rw [eb] at hb
  have hmul := Rat.le_trans hb hm
  have hfinal := Rat.le_of_mul_le_mul_right hmul hk
  apply Rat.le_of_mul_le_mul_right (c := s) _ hs
  rw [Rat.div_mul_cancel (Rat.ne_of_gt hs)]
  exact hfinal

/-- At exactly n=s*k, the variance comparison is equivalent to the threshold. -/
theorem exact_sample_ratio (vr vy k s : Rat) (hk : 0 < k) (hs : 0 < s) :
    vr/k ≤ vy/(s*k) ↔ vr ≤ vy/s := by
  have hsk := Rat.mul_pos hs hk
  have cr := Rat.div_mul_cancel (a := vr) (Rat.ne_of_gt hk)
  have cy := Rat.div_mul_cancel (a := vy) (Rat.ne_of_gt hsk)
  have cs := Rat.div_mul_cancel (a := vy) (Rat.ne_of_gt hs)
  have er : vr/k*(s*k) = vr*s := by grind
  constructor
  · intro h
    have hm := Rat.mul_le_mul_of_nonneg_right h (Rat.le_of_lt hsk)
    rw [er, cy] at hm
    apply Rat.le_of_mul_le_mul_right (c := s) _ hs
    rw [cs]
    exact hm
  · intro h
    have hm := Rat.mul_le_mul_of_nonneg_right h (Rat.le_of_lt hs)
    rw [cs] at hm
    apply Rat.le_of_mul_le_mul_right (c := s*k) _ hsk
    rw [er, cy]
    exact hm

/-- Squared population correlation; no square roots or real analysis required. -/
def rhoSquared {n} (y x : Vec n) :=
  covariance y x*covariance y x/(variance x*variance y)

theorem optimal_variance_fraction {n} (y x : Vec n)
    (hx : variance x ≠ 0) (hy : variance y ≠ 0) :
    variance (linearResidual y x (covariance y x/variance x)) =
      variance y*(1-rhoSquared y x) := by
  rw [optimal_control_variance y x hx]
  unfold rhoSquared
  have hp : variance x*variance y ≠ 0 := by
    intro h; rcases Rat.mul_eq_zero.mp h with h | h
    · exact hx h
    · exact hy h
  have hc := Rat.div_mul_cancel (a := covariance y x*covariance y x) hp
  have hd := Rat.div_mul_cancel (a := covariance y x*covariance y x) hx
  grind

/-- Necessary even for the best fixed linear coefficient and free cheap work. -/
theorem rho_necessary {n} (y x : Vec n) (hx : 0 < variance x)
    (hy : 0 < variance y) (s : Rat) (hs : 0 < s)
    (h : variance (linearResidual y x (covariance y x/variance x)) ≤ variance y/s) :
    1-1/s ≤ rhoSquared y x := by
  rw [optimal_variance_fraction y x (Rat.ne_of_gt hx) (Rat.ne_of_gt hy)] at h
  have hc := Rat.div_mul_cancel (a := (1 : Rat)) (Rat.ne_of_gt hs)
  have he : variance y/s = variance y*(1/s) := by grind [Rat.div_def]
  rw [he] at h
  have hb := Rat.le_of_mul_le_mul_left h hy
  grind

theorem any_fixed_control_rho_necessary {n} (y x : Vec n)
    (hx : 0 < variance x) (hy : 0 < variance y) (beta s : Rat) (hs : 0 < s)
    (h : variance (linearResidual y x beta) ≤ variance y/s) :
    1-1/s ≤ rhoSquared y x :=
  rho_necessary y x hx hy s hs (Rat.le_trans (fixed_control_optimal y x hx beta) h)

theorem rho_100 {n} (y x : Vec n) (hx : 0 < variance x) (hy : 0 < variance y)
    (h : variance (linearResidual y x (covariance y x/variance x)) ≤ variance y/100) :
    (99/100 : Rat) ≤ rhoSquared y x := by
  have := rho_necessary y x hx hy 100 (by decide) h
  grind

theorem rho_1000 {n} (y x : Vec n) (hx : 0 < variance x) (hy : 0 < variance y)
    (h : variance (linearResidual y x (covariance y x/variance x)) ≤ variance y/1000) :
    (999/1000 : Rat) ≤ rhoSquared y x := by
  have := rho_necessary y x hx hy 1000 (by decide) h
  grind

/-- Baseline: ordinary iid-with-replacement sample mean of the true scores. -/
def ordinaryMean {n k} (y : Vec n) (t : Samples n k) := (1/(k : Rat))*drawSum y t

theorem ordinary_mean_variance {n k} (hn : 0 < n) (hk : 0 < k) (y : Vec n) :
    randomVariance (ordinaryMean (k := k) y) = variance y/(k : Rat) := by
  have he : estimator (k := k) y (fun _ => 0) = ordinaryMean (k := k) y := by
    funext t
    unfold estimator ordinaryMean
    rw [avg_const hn]
    have hr : residual y (fun _ => 0) = y := by funext i; unfold residual; grind
    rw [hr]
    grind
  have hv : residual y (fun _ => 0) = y := by funext i; unfold residual; grind
  have h := estimator_variance hn hk y (fun _ => 0)
  rw [he, hv] at h
  exact h

/-- Instantiates the arithmetic necessity with the proved probability variances.
The ordinary baseline is also with replacement, not a census or sampling without
replacement. s is rational; audit and ordinary draw counts are actual naturals. -/
theorem cohort_sample_reduction_necessary {N k n : Nat} (hN : 0 < N)
    (hk : 0 < k) (hn : 0 < n) (y p : Vec N) (s : Rat) (hs : 0 < s)
    (hbudget : s*(k : Rat) ≤ (n : Rat))
    (hmatch : randomVariance (estimator (k := k) y p) ≤
      randomVariance (ordinaryMean (k := n) y)) :
    variance (residual y p) ≤ variance y/s := by
  rw [estimator_variance hN hk, ordinary_mean_variance hN hn] at hmatch
  exact sample_reduction_necessary _ _ _ _ _ (variance_nonneg hN _)
    (nat_pos hk) hs hbudget hmatch

/-- Direct bridge from actual sample-design variances to the correlation threshold
for ANY fixed linear coefficient, not just an assumed optimal estimator. -/
theorem linear_control_sample_reduction_rho {N k n : Nat} (hN : 0 < N)
    (hk : 0 < k) (hn : 0 < n) (y x : Vec N)
    (hx : 0 < variance x) (hy : 0 < variance y) (beta s : Rat) (hs : 0 < s)
    (hbudget : s*(k : Rat) ≤ (n : Rat))
    (hmatch : randomVariance (estimator (k := k) y (fun i => beta*x i)) ≤
      randomVariance (ordinaryMean (k := n) y)) :
    1-1/s ≤ rhoSquared y x := by
  have h := cohort_sample_reduction_necessary hN hk hn y (fun i => beta*x i)
    s hs hbudget hmatch
  exact any_fixed_control_rho_necessary y x hx hy beta s hs h

/-- All costs use a common compute/wall-time unit. pairCost covers BOTH games.
Calibration and independent confirmation are total costs, not free operations. -/
def auditedCost (N k cheapCost pairCost calibrationCost confirmationCost : Rat) :=
  N*cheapCost + calibrationCost + k*pairCost + confirmationCost

theorem audited_cost_floor (N k cheapCost pairCost calibrationCost confirmationCost : Rat)
    (hN : 0 ≤ N) (hp : 0 ≤ cheapCost) (hc : 0 ≤ calibrationCost)
    (hf : 0 ≤ confirmationCost) :
    k*pairCost ≤ auditedCost N k cheapCost pairCost calibrationCost confirmationCost := by
  have h := Rat.mul_nonneg hN hp
  unfold auditedCost
  grind

/-- Positive overhead makes an exact s-fold reduction in game calls LESS than
an s-fold all-in speedup relative to an n*pairCost game-only budget. This is a
budget obstruction, not a model of all baseline overheads or measured speed. -/
theorem positive_overhead_blocks_game_only_speedup
    (N k cheapCost pairCost calibrationCost confirmationCost s : Rat)
    (hs : 0 < s)
    (hoverhead : 0 < N*cheapCost + calibrationCost + confirmationCost) :
    (s*k)*pairCost < s*auditedCost N k cheapCost pairCost calibrationCost confirmationCost := by
  have h := Rat.mul_pos hs hoverhead
  unfold auditedCost
  have he : s*(N*cheapCost + calibrationCost + confirmationCost) =
      s*(N*cheapCost + calibrationCost + k*pairCost + confirmationCost)-(s*k)*pairCost := by
    grind
  rw [he] at h
  grind

end AuditedTuning

-- Exhaustive transitive kernel-dependency audit, including all helper theorems.
#print axioms AuditedTuning.total_congr
#print axioms AuditedTuning.total_add
#print axioms AuditedTuning.total_scale
#print axioms AuditedTuning.total_const
#print axioms AuditedTuning.total_nonneg
#print axioms AuditedTuning.nat_pos
#print axioms AuditedTuning.avg_congr
#print axioms AuditedTuning.avg_add
#print axioms AuditedTuning.avg_scale
#print axioms AuditedTuning.avg_const
#print axioms AuditedTuning.avg_sub
#print axioms AuditedTuning.avg_nonneg
#print axioms AuditedTuning.expect_congr
#print axioms AuditedTuning.expect_const
#print axioms AuditedTuning.expect_add
#print axioms AuditedTuning.expect_scale
#print axioms AuditedTuning.expect_sub
#print axioms AuditedTuning.expect_nonneg
#print axioms AuditedTuning.actual_distribution_formula
#print axioms AuditedTuning.tuple_mass_positive
#print axioms AuditedTuning.probability_normalized
#print axioms AuditedTuning.residual_mean
#print axioms AuditedTuning.center_mean
#print axioms AuditedTuning.square_nonneg
#print axioms AuditedTuning.variance_nonneg
#print axioms AuditedTuning.drawSum_expect
#print axioms AuditedTuning.estimator_unbiased
#print axioms AuditedTuning.centered_sum_energy
#print axioms AuditedTuning.drawSum_center
#print axioms AuditedTuning.estimator_variance
#print axioms AuditedTuning.one_draw_variance
#print axioms AuditedTuning.selective_audit_counterexample
#print axioms AuditedTuning.counterexample_bounded
#print axioms AuditedTuning.predictor_only_counterexample
#print axioms AuditedTuning.skew_weights_valid
#print axioms AuditedTuning.nonuniform_uncorrected_counterexample
#print axioms AuditedTuning.frozen_cohort_unbiased
#print axioms AuditedTuning.frozen_cohort_variance
#print axioms AuditedTuning.center_linear
#print axioms AuditedTuning.control_variance_quadratic
#print axioms AuditedTuning.control_variance_identity
#print axioms AuditedTuning.optimal_control_variance
#print axioms AuditedTuning.fixed_control_optimal
#print axioms AuditedTuning.sample_reduction_necessary
#print axioms AuditedTuning.exact_sample_ratio
#print axioms AuditedTuning.optimal_variance_fraction
#print axioms AuditedTuning.rho_necessary
#print axioms AuditedTuning.any_fixed_control_rho_necessary
#print axioms AuditedTuning.rho_100
#print axioms AuditedTuning.rho_1000
#print axioms AuditedTuning.ordinary_mean_variance
#print axioms AuditedTuning.cohort_sample_reduction_necessary
#print axioms AuditedTuning.linear_control_sample_reduction_rho
#print axioms AuditedTuning.audited_cost_floor
#print axioms AuditedTuning.positive_overhead_blocks_game_only_speedup
