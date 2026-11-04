import Std

/-!
Exact-rational and unbounded-integer specification, independent of any implementation.
Only Std is imported. All closed computations use kernel-checked `decide`.
No statistical, floating-point, runtime, or playing-strength theorem is asserted.
-/
namespace HyperSPSA

/-! Four-corner algebra. Coordinates of `bilinear` are normalized to [-1,1]. -/
structure Corners where
  ll : Rat
  lh : Rat
  hl : Rat
  hh : Rat
  deriving DecidableEq

def mean4 (c : Corners) := c.ll + c.lh + c.hl + c.hh
def x4 (c : Corners) := c.hl + c.hh - c.ll - c.lh
def y4 (c : Corners) := c.lh + c.hh - c.ll - c.hl
def mixed4 (c : Corners) := c.ll + c.hh - c.lh - c.hl

def bilinear (c : Corners) (x y : Rat) : Rat :=
  mean4 c / 4 + x4 c / 4 * x + y4 c / 4 * y + mixed4 c / 4 * x * y

theorem reconstruct (c : Corners) :
    bilinear c (-1) (-1) = c.ll ∧ bilinear c (-1) 1 = c.lh ∧
    bilinear c 1 (-1) = c.hl ∧ bilinear c 1 1 = c.hh := by
  unfold bilinear mean4 x4 y4 mixed4
  grind

def sample (f : Rat → Rat → Rat) : Corners :=
  ⟨f (-1) (-1), f (-1) 1, f 1 (-1), f 1 1⟩

theorem bilinear_unique (a b c d x y : Rat) :
    bilinear (sample (fun u v => a + b*u + c*v + d*u*v)) x y =
      a + b*x + c*y + d*x*y := by
  unfold bilinear sample mean4 x4 y4 mixed4
  grind

def lerp (a b t : Rat) := (1-t)*a + t*b

theorem lerp_upper (a b t M : Rat) (h0 : 0 ≤ t) (h1 : t ≤ 1)
    (ha : a ≤ M) (hb : b ≤ M) : lerp a b t ≤ M := by
  have h : 0 ≤ 1-t := by grind
  have p := Rat.mul_le_mul_of_nonneg_left ha h
  have q := Rat.mul_le_mul_of_nonneg_left hb h0
  unfold lerp
  grind

theorem tensor_interpolation (c : Corners) (t s : Rat) :
    bilinear c (2*t-1) (2*s-1) =
      lerp (lerp c.ll c.lh s) (lerp c.hl c.hh s) t := by
  unfold bilinear mean4 x4 y4 mixed4 lerp
  grind

theorem bilinear_upper (c : Corners) (x y M : Rat)
    (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1)
    (hc : c.ll ≤ M ∧ c.lh ≤ M ∧ c.hl ≤ M ∧ c.hh ≤ M) :
    bilinear c x y ≤ M := by
  have tx : 0 ≤ (x+1)/2 ∧ (x+1)/2 ≤ 1 := by grind
  have ty : 0 ≤ (y+1)/2 ∧ (y+1)/2 ≤ 1 := by grind
  have lo := lerp_upper c.ll c.lh ((y+1)/2) M ty.1 ty.2 hc.1 hc.2.1
  have hi := lerp_upper c.hl c.hh ((y+1)/2) M ty.1 ty.2 hc.2.2.1 hc.2.2.2
  have h := lerp_upper _ _ ((x+1)/2) M tx.1 tx.2 lo hi
  have eq := tensor_interpolation c ((x+1)/2) ((y+1)/2)
  have ex : 2*((x+1)/2)-1 = x := by grind
  have ey : 2*((y+1)/2)-1 = y := by grind
  rw [ex, ey] at eq
  rw [eq]
  exact h

def greater (a b : Rat) := if a < b then b else a
def cornerMax (c : Corners) := greater (greater c.ll c.lh) (greater c.hl c.hh)

theorem cornerMax_spec (c : Corners) :
    (cornerMax c = c.ll ∨ cornerMax c = c.lh ∨
      cornerMax c = c.hl ∨ cornerMax c = c.hh) ∧
    c.ll ≤ cornerMax c ∧ c.lh ≤ cornerMax c ∧
      c.hl ≤ cornerMax c ∧ c.hh ≤ cornerMax c := by
  unfold cornerMax greater
  grind

theorem maximum_at_corner (c : Corners) :
    (∃ x y : Rat, (x = -1 ∨ x = 1) ∧ (y = -1 ∨ y = 1) ∧
      bilinear c x y = cornerMax c) ∧
    ∀ x y : Rat, -1 ≤ x ∧ x ≤ 1 → -1 ≤ y ∧ y ≤ 1 →
      bilinear c x y ≤ cornerMax c := by
  have h := cornerMax_spec c
  have r := reconstruct c
  constructor
  · rcases h.1 with h | h | h | h
    · exact ⟨-1, -1, Or.inl rfl, Or.inl rfl, r.1.trans h.symm⟩
    · exact ⟨-1, 1, Or.inl rfl, Or.inr rfl, r.2.1.trans h.symm⟩
    · exact ⟨1, -1, Or.inr rfl, Or.inl rfl, r.2.2.1.trans h.symm⟩
    · exact ⟨1, 1, Or.inr rfl, Or.inr rfl, r.2.2.2.trans h.symm⟩
  · intro x y hx hy
    exact bilinear_upper c x y (cornerMax c) hx hy h.2

/-- Pullback to an arbitrary nondegenerate physical rectangle. -/
def rectangle (c : Corners) (lx hx ly hy x y : Rat) :=
  bilinear c (2*((x-lx)/(hx-lx))-1) (2*((y-ly)/(hy-ly))-1)

theorem unit_fraction_bounds (a b x : Rat) (hab : a < b) (hx : a ≤ x ∧ x ≤ b) :
    0 ≤ (x-a)/(b-a) ∧ (x-a)/(b-a) ≤ 1 := by
  have hp : 0 < b-a := by grind
  have hi : 0 ≤ (b-a)⁻¹ := by have := Rat.inv_pos.mpr hp; grind
  have hn : 0 ≤ x-a := by grind
  have hl := Rat.mul_nonneg hn hi
  have hu := Rat.mul_le_mul_of_nonneg_right (show x-a ≤ b-a by grind) hi
  have hz : b-a ≠ 0 := by grind
  have he := Rat.mul_inv_cancel (b-a) hz
  change 0 ≤ (x-a)*(b-a)⁻¹ ∧ (x-a)*(b-a)⁻¹ ≤ 1
  grind

theorem rectangle_maximum (c : Corners) (lx hx ly hy : Rat)
    (hwidthX : lx < hx) (hwidthY : ly < hy) :
    (rectangle c lx hx ly hy lx ly = c.ll ∧
     rectangle c lx hx ly hy lx hy = c.lh ∧
     rectangle c lx hx ly hy hx ly = c.hl ∧
     rectangle c lx hx ly hy hx hy = c.hh) ∧
    ∀ x y, lx ≤ x ∧ x ≤ hx → ly ≤ y ∧ y ≤ hy →
      rectangle c lx hx ly hy x y ≤ cornerMax c := by
  have nx : hx-lx ≠ 0 := by grind
  have ny : hy-ly ≠ 0 := by grind
  constructor
  · have r := reconstruct c
    have ex : (hx-lx)/(hx-lx) = 1 := by grind
    have ey : (hy-ly)/(hy-ly) = 1 := by grind
    unfold rectangle
    simp only [ex, ey]
    grind
  · intro x y bx byy
    have tx := unit_fraction_bounds lx hx x hwidthX bx
    have ty := unit_fraction_bounds ly hy y hwidthY byy
    apply bilinear_upper c _ _ (cornerMax c) ?_ ?_ (cornerMax_spec c).2 <;> grind

/-! General centered quadratic recovery, including both diagonal terms. -/
def quadratic (c gx gy dxx dxy dyy x y : Rat) :=
  c + gx*x + gy*y + dxx*x*x + dxy*x*y + dyy*y*y

def centeredSamples (c gx gy dxx dxy dyy a b h k : Rat) : Corners :=
  sample (fun u v => quadratic c gx gy dxx dxy dyy (a+h*u) (b+k*v))

theorem quadratic_recovery (c gx gy dxx dxy dyy a b h k : Rat) :
    mean4 (centeredSamples c gx gy dxx dxy dyy a b h k) / 4 =
      quadratic c gx gy dxx dxy dyy a b + dxx*h*h + dyy*k*k ∧
    x4 (centeredSamples c gx gy dxx dxy dyy a b h k) / 4 =
      h*(gx + 2*dxx*a + dxy*b) ∧
    y4 (centeredSamples c gx gy dxx dxy dyy a b h k) / 4 =
      k*(gy + dxy*a + 2*dyy*b) ∧
    mixed4 (centeredSamples c gx gy dxx dxy dyy a b h k) / 4 = h*k*dxy := by
  unfold centeredSamples sample quadratic mean4 x4 y4 mixed4
  grind

/-! Arbitrary-dimensional quadratic restriction. The matrix need not be symmetric. -/
abbrev Vec (n : Nat) := Fin n → Rat

def total : {n : Nat} → Vec n → Rat
  | 0, _ => 0
  | n+1, f => f 0 + total (fun i : Fin n => f i.succ)

theorem total_congr {n} {f g : Vec n} (h : ∀ i, f i = g i) : total f = total g := by
  rw [funext h]

theorem total_zero (n) : total (fun _ : Fin n => (0 : Rat)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [total, ih]; grind

theorem total_add {n} (f g : Vec n) :
    total (fun i => f i + g i) = total f + total g := by
  induction n with
  | zero => simp only [total]; grind
  | succ n ih => simp only [total, ih]; grind

theorem total_scale {n} (a : Rat) (f : Vec n) :
    total (fun i => a*f i) = a*total f := by
  induction n with
  | zero => simp only [total]; grind
  | succ n ih => simp only [total, ih]; grind

def dot {n} (x y : Vec n) := total (fun i => x i*y i)
def form {n} (Q : Fin n → Vec n) (x y : Vec n) :=
  total (fun i => total (fun j => Q i j*x i*y j))
def affine2 {n} (b u v : Vec n) (s t : Rat) : Vec n :=
  fun i => b i+s*u i+t*v i

theorem dot_affine2 {n} (l b u v : Vec n) (s t : Rat) :
    dot l (affine2 b u v s t) = dot l b + s*dot l u + t*dot l v := by
  unfold dot affine2
  calc
    total (fun i => l i*(b i+s*u i+t*v i)) =
      total (fun i => l i*b i+s*(l i*u i)+t*(l i*v i)) := by
        apply total_congr; intro i; grind
    _ = _ := by simp only [total_add, total_scale]

theorem form_affine2_left {n} (Q : Fin n → Vec n) (b u v w : Vec n) (s t : Rat) :
    form Q (affine2 b u v s t) w = form Q b w + s*form Q u w + t*form Q v w := by
  unfold form affine2
  calc
    total (fun i => total (fun j => Q i j*(b i+s*u i+t*v i)*w j)) =
      total (fun i => total (fun j => Q i j*b i*w j+
        s*(Q i j*u i*w j)+t*(Q i j*v i*w j))) := by
          apply total_congr; intro i; apply total_congr; intro j; grind
    _ = _ := by simp only [total_add, total_scale]

theorem form_affine2_right {n} (Q : Fin n → Vec n) (w b u v : Vec n) (s t : Rat) :
    form Q w (affine2 b u v s t) = form Q w b + s*form Q w u + t*form Q w v := by
  unfold form affine2
  calc
    total (fun i => total (fun j => Q i j*w i*(b j+s*u j+t*v j))) =
      total (fun i => total (fun j => Q i j*w i*b j+
        s*(Q i j*w i*u j)+t*(Q i j*w i*v j))) := by
          apply total_congr; intro i; apply total_congr; intro j; grind
    _ = _ := by simp only [total_add, total_scale]

def objective {n} (c : Rat) (l : Vec n) (Q : Fin n → Vec n) (x : Vec n) :=
  c + dot l x + form Q x x

def directional {n} (l : Vec n) (Q : Fin n → Vec n) (b u : Vec n) :=
  dot l u + form Q b u + form Q u b

theorem quadratic_plane_restriction {n} (c : Rat) (l b u v : Vec n)
    (Q : Fin n → Vec n) (s t : Rat) :
    objective c l Q (affine2 b u v s t) =
      quadratic (objective c l Q b) (directional l Q b u) (directional l Q b v)
        (form Q u u) (form Q u v + form Q v u) (form Q v v) s t := by
  unfold objective
  rw [dot_affine2, form_affine2_left]
  simp only [form_affine2_right]
  unfold quadratic directional
  grind

/-- Basis directions make every coordinate except i,j literally unchanged. -/
def basis {n} (i : Fin n) : Vec n := fun k => if k = i then 1 else 0

theorem pair_fixed_coordinates {n} (b : Vec n) (i j k : Fin n) (s t : Rat)
    (hi : k ≠ i) (hj : k ≠ j) :
    affine2 b (basis i) (basis j) s t k = b k := by
  simp only [affine2, basis, ite_eq_right hi, ite_eq_right hj]
  grind

theorem pair_moving_coordinates {n} (b : Vec n) (i j : Fin n) (s t : Rat)
    (hij : i ≠ j) :
    affine2 b (basis i) (basis j) s t i = b i+s ∧
    affine2 b (basis i) (basis j) s t j = b j+t := by
  unfold affine2 basis
  grind

/-- Recovery at an arbitrary incumbent in arbitrary dimension; other coordinates
are held fixed by basis directions, not perturbed simultaneously. -/
theorem full_quadratic_pair_recovery {n} (c : Rat) (l b : Vec n)
    (Q : Fin n → Vec n) (i j : Fin n) (h k : Rat) :
    let u := basis i
    let v := basis j
    let face := sample (fun s t => objective c l Q (affine2 b u v (h*s) (k*t)))
    mean4 face / 4 = objective c l Q b + form Q u u*h*h + form Q v v*k*k ∧
    x4 face / 4 = h*directional l Q b u ∧
    y4 face / 4 = k*directional l Q b v ∧
    mixed4 face / 4 = h*k*(form Q u v + form Q v u) := by
  dsimp only
  have eq : (fun s t => objective c l Q (affine2 b (basis i) (basis j) (h*s) (k*t))) =
    (fun s t => quadratic (objective c l Q b) (directional l Q b (basis i))
      (directional l Q b (basis j)) (form Q (basis i) (basis i))
      (form Q (basis i) (basis j)+form Q (basis j) (basis i))
      (form Q (basis j) (basis j)) (0+h*s) (0+k*t)) := by
        funext s t
        have e := quadratic_plane_restriction c l b (basis i) (basis j) Q (h*s) (k*t)
        simpa only [Rat.zero_add] using e
  rw [eq]
  have r := quadratic_recovery (objective c l Q b) (directional l Q b (basis i))
    (directional l Q b (basis j)) (form Q (basis i) (basis i))
    (form Q (basis i) (basis j)+form Q (basis j) (basis i))
    (form Q (basis j) (basis j)) 0 0 h k
  unfold centeredSamples at r
  simpa only [quadratic, Rat.mul_zero, Rat.add_zero, Rat.zero_mul] using r

/-- A term depending only on two untouched coordinates is constant on a face. -/
theorem untouched_product_no_alias {n} (b : Vec n) (i j p q : Fin n) (a h k : Rat)
    (hpi : p ≠ i) (hpj : p ≠ j) (hqi : q ≠ i) (hqj : q ≠ j) :
    mixed4 (sample (fun s t =>
      a * affine2 b (basis i) (basis j) (h*s) (k*t) p *
          affine2 b (basis i) (basis j) (h*s) (k*t) q)) = 0 := by
  have eq : (fun s t =>
      a * affine2 b (basis i) (basis j) (h*s) (k*t) p *
          affine2 b (basis i) (basis j) (h*s) (k*t) q) =
      (fun _ _ : Rat => a*b p*b q) := by
    funext s t
    rw [pair_fixed_coordinates b i j p _ _ hpi hpj,
        pair_fixed_coordinates b i j q _ _ hqi hqj]
  rw [eq]
  unfold mixed4 sample
  grind

/-! Literal normalized legacy estimator: 16 rows, simultaneous positive/negative probes. -/
def bitSign (r c k : Nat) : Rat := if r.testBit k && c.testBit k then -1 else 1
def hadamard16 (r c : Nat) : Rat :=
  bitSign r c 0 * bitSign r c 1 * bitSign r c 2 * bitSign r c 3
/-- Entrywise form of the frozen doubling construction [H H; H -H]. -/
def sylvester : Nat → Nat → Nat → Rat
  | 0, _, _ => 1
  | n+1, r, c =>
      (if 2^n ≤ r ∧ 2^n ≤ c then -1 else 1) * sylvester n (r % 2^n) (c % 2^n)

theorem hadamard_matches_doubling : ∀ r c : Fin 16,
    hadamard16 r.val c.val = sylvester 4 r.val c.val := by decide +kernel

def columns : List Nat := [1, 2, 4, 8, 5, 10, 6, 11]
def legacyPoint (r : Fin 16) : Vec 8 := fun i => hadamard16 r.val columns[i.val]!
def negative {n} (x : Vec n) : Vec n := fun i => -x i

def oldInteraction01 (f : Vec 8 → Rat) : Rat :=
  total (fun r : Fin 16 =>
    (f (legacyPoint r) + f (negative (legacyPoint r)) - 2*f (fun _ => 0)) *
      legacyPoint r 0 * legacyPoint r 1 / 32)

theorem column_collision : Nat.xor 1 2 = 3 ∧ Nat.xor 8 11 = 3 := by decide

theorem row_alias : ∀ r : Fin 16,
    legacyPoint r 0 * legacyPoint r 1 = legacyPoint r 3 * legacyPoint r 7 := by
  decide +kernel

theorem row_sign_square : ∀ r : Fin 16, ∀ i : Fin 8,
    legacyPoint r i * legacyPoint r i = 1 := by decide +kernel

theorem oldInteraction_scale (a : Rat) (f : Vec 8 → Rat) :
    oldInteraction01 (fun x => a*f x) = a*oldInteraction01 f := by
  unfold oldInteraction01
  rw [← total_scale]
  apply total_congr
  intro r
  grind

/-- Kernel evaluates the complete 16-row estimator, not a hand-entered answer. -/
theorem legacy_alias_unit : oldInteraction01 (fun x => x 3*x 7) = 1 := by decide +kernel

theorem legacy_alias_general (a : Rat) :
    oldInteraction01 (fun x => a*(x 3*x 7)) = a := by
  rw [oldInteraction_scale, legacy_alias_unit]
  grind

theorem legacy_alias_seven : oldInteraction01 (fun x => 7*(x 3*x 7)) = 7 ∧
    mixed4 (sample (fun s t =>
      let x := affine2 (fun _ : Fin 8 => (0 : Rat)) (basis 0) (basis 1) s t
      7*x 3*x 7)) = 0 := by
  constructor
  · exact legacy_alias_general 7
  · simpa only [Rat.one_mul] using
      untouched_product_no_alias (n := 8) (fun _ => 0) 0 1 3 7 7 1 1
        (by decide) (by decide) (by decide) (by decide)

def oldPerRow (f : Vec 8 → Rat) (r : Fin 16) : Rat :=
  (f (legacyPoint r) - f (negative (legacyPoint r))) * legacyPoint r 0 / 2
def oldGradient (f : Vec 8 → Rat) := total (fun r : Fin 16 => oldPerRow f r / 16)
def oldSecondMoment (f : Vec 8 → Rat) :=
  total (fun r : Fin 16 => oldPerRow f r * oldPerRow f r / 16)
def noiselessLinear (x : Vec 8) := x 0 + 100*x 1

theorem false_SE_moments : oldGradient noiselessLinear = 1 ∧
    oldSecondMoment noiselessLinear = 10001 ∧ oldInteraction01 noiselessLinear = 0 := by
  decide +kernel

/-- SE=25 is characterized as the unique nonnegative square root, avoiding
any unverified real sqrt implementation. The legacy 1e-9 floor is inactive. -/
theorem false_SE_exact :
    (oldSecondMoment noiselessLinear - oldGradient noiselessLinear *
      oldGradient noiselessLinear) / 16 = 25*25 ∧
    (∀ se : Rat, 0 ≤ se → se*se =
      (oldSecondMoment noiselessLinear - oldGradient noiselessLinear *
        oldGradient noiselessLinear) / 16 → se = 25) ∧
    (if (oldGradient noiselessLinear)*2 ≥ 3*(25 : Rat)
      then oldGradient noiselessLinear else 0) = 0 := by
  have h := false_SE_moments
  constructor
  · rw [h.1, h.2.1]; grind
  constructor
  · intro se hs he
    rw [h.1, h.2.1] at he
    have hp : (se-25)*(se+25) = 0 := by grind
    have hn : se+25 ≠ 0 := by grind
    grind
  · rw [h.1]; grind

/-! Bounds: unbounded Int specification of widened arithmetic and truncating half-step. -/
def clip (lo hi z : Int) := if z < lo then lo else if hi < z then hi else z

theorem clip_bounds (lo hi z : Int) (h : lo ≤ hi) :
    lo ≤ clip lo hi z ∧ clip lo hi z ≤ hi := by
  unfold clip
  omega

theorem endpoints_bounded (lo hi b step : Int) (hb : lo ≤ b ∧ b ≤ hi)
    (hs : 0 < step) :
    lo ≤ clip lo hi (b-step) ∧ clip lo hi (b-step) ≤ b ∧
    b ≤ clip lo hi (b+step) ∧ clip lo hi (b+step) ≤ hi := by
  unfold clip
  omega

theorem fixed_endpoints (b step : Int) : clip b b (b-step) = b ∧ clip b b (b+step) = b := by
  unfold clip
  omega

def halfstep (a b : Int) := a + (b-a).tdiv 2

theorem halfstep_between (a b : Int) :
    min a b ≤ halfstep a b ∧ halfstep a b ≤ max a b := by
  unfold halfstep
  by_cases h : a ≤ b
  · have hp : 0 ≤ b-a := by omega
    have h0 := Int.tdiv_nonneg hp (show (0 : Int) ≤ 2 by decide)
    have h1 := Int.tdiv_le_self 2 hp
    omega
  · have hp : 0 ≤ a-b := by omega
    have h0 := Int.tdiv_nonneg hp (show (0 : Int) ≤ 2 by decide)
    have h1 := Int.tdiv_le_self 2 hp
    have he : (b-a).tdiv 2 = -(a-b).tdiv 2 := by
      rw [show b-a = -(a-b) by omega, Int.neg_tdiv]
    omega

theorem halfstep_bounded (lo hi a b : Int) (ha : lo ≤ a ∧ a ≤ hi)
    (hb : lo ≤ b ∧ b ≤ hi) : lo ≤ halfstep a b ∧ halfstep a b ≤ hi := by
  have h := halfstep_between a b
  omega

/-- Includes opposite i64 extremes: b-a and b+step fit i128 before clipping.
No assertion about unchecked Rust arithmetic is made. -/
theorem widened_i64_arithmetic (a b step : Int)
    (ha : -9223372036854775808 ≤ a ∧ a ≤ 9223372036854775807)
    (hb : -9223372036854775808 ≤ b ∧ b ≤ 9223372036854775807)
    (hs : 0 < step ∧ step ≤ 9223372036854775807) :
    -170141183460469231731687303715884105728 ≤ b-a ∧
    b-a ≤ 170141183460469231731687303715884105727 ∧
    -170141183460469231731687303715884105728 ≤ b-step ∧
    b+step ≤ 170141183460469231731687303715884105727 := by omega

/-- Coordinatewise merges preserve bounds, but this is NOT a utility guarantee. -/
theorem merge_bounded {n p} (lo hi : Fin n → Int) (cand : Fin p → Fin n → Int)
    (which : Fin n → Fin p)
    (h : ∀ q i, lo i ≤ cand q i ∧ cand q i ≤ hi i) :
    ∀ i, lo i ≤ cand (which i) i ∧ cand (which i) i ≤ hi i := by
  intro i
  exact h (which i) i

/-! Deterministic maximum: incumbent first, strict replacement, stable ties. -/
def best {α : Type} (score : α → Int) : α → List α → α
  | b, [] => b
  | b, x::xs => best score (if score b < score x then x else b) xs

theorem best_mem {α} (score : α → Int) (b : α) (xs : List α) :
    best score b xs = b ∨ best score b xs ∈ xs := by
  induction xs generalizing b with
  | nil => exact Or.inl rfl
  | cons x xs ih =>
    simp only [best]
    split
    · have h := ih x
      simp only [List.mem_cons]
      grind
    · have h := ih b
      simp only [List.mem_cons]
      grind

theorem best_monotone {α} (score : α → Int) (b : α) (xs : List α) :
    score b ≤ score (best score b xs) := by
  induction xs generalizing b with
  | nil => simp only [best]; omega
  | cons x xs ih =>
    simp only [best]
    split
    · have h := ih x; omega
    · exact ih b

theorem best_dominates {α} (score : α → Int) (b : α) (xs : List α) :
    ∀ x ∈ xs, score x ≤ score (best score b xs) := by
  induction xs generalizing b with
  | nil => simp
  | cons x xs ih =>
    intro y hy
    simp only [List.mem_cons] at hy
    simp only [best]
    split
    · have hm := best_monotone score x xs
      have ht := ih x
      rcases hy with rfl | hy
      · exact hm
      · exact ht y hy
    · have hm := best_monotone score b xs
      have ht := ih b
      rcases hy with rfl | hy
      · omega
      · exact ht y hy

theorem best_holds_on_ties {α} (score : α → Int) (b : α) (xs : List α)
    (h : ∀ x ∈ xs, score x ≤ score b) : best score b xs = b := by
  induction xs generalizing b with
  | nil => rfl
  | cons x xs ih =>
    have hx := h x (by simp)
    have ht : ∀ y ∈ xs, score y ≤ score b := by
      intro y hy
      exact h y (by simp [hy])
    simp only [best, ite_eq_right (show ¬score b < score x by omega)]
    exact ih b ht

theorem best_changed_strict {α} (score : α → Int) (b : α) (xs : List α)
    (h : best score b xs ≠ b) : score b < score (best score b xs) := by
  induction xs generalizing b with
  | nil => exact False.elim (h rfl)
  | cons x xs ih =>
    by_cases hh : score b < score x
    · simp only [best, ite_eq_left hh] at h ⊢
      have hm := best_monotone score x xs
      omega
    · simp only [best, ite_eq_right hh] at h ⊢
      exact ih b h

/-- Protected metrics are oriented 'higher is better'. The first point is the
reference. Production uses a FIXED anchor here, not the changing incumbent. -/
def Guard {α} {m : Nat} (metric : α → Fin m → Int) (tol : Fin m → Int)
    (b x : α) : Prop := ∀ k, metric b k ≤ metric x k + tol k

instance {α} {m : Nat} (metric : α → Fin m → Int) (tol : Fin m → Int) (b x : α) :
    Decidable (Guard metric tol b x) := inferInstanceAs (Decidable (∀ k, metric b k ≤ metric x k+tol k))

/-- HISTORICAL relative-incumbent selector, NOT the production gate.
Its original one-update theorems remain valid; iteration can spend tolerance again.
See `fixedSelect` for the current fixed-anchor contract. -/
def select {α} {m : Nat} (score : α → Int) (metric : α → Fin m → Int)
    (tol : Fin m → Int) (b : α) (predeclared : List α) : α :=
  best score b (predeclared.filter (fun x => decide (Guard metric tol b x)))

theorem acceptance_contract {α} {m : Nat} (score : α → Int) (metric : α → Fin m → Int)
    (tol : Fin m → Int) (b : α) (xs : List α) (ht : ∀ k, 0 ≤ tol k) :
    let out := select score metric tol b xs
    Guard metric tol b out ∧
    (out = b ∨ (out ∈ xs ∧ score b < score out)) := by
  dsimp only
  have hm := best_mem score b (xs.filter (fun x => decide (Guard metric tol b x)))
  by_cases he : select score metric tol b xs = b
  · rw [he]
    constructor
    · intro k; have hk := ht k; omega
    · exact Or.inl rfl
  · have hs := best_changed_strict score b _ he
    have hf : select score metric tol b xs ∈
        xs.filter (fun x => decide (Guard metric tol b x)) := by
      unfold select at he ⊢
      exact hm.resolve_left he
    have hp := List.mem_filter.mp hf
    exact ⟨of_decide_eq_true hp.2, Or.inr ⟨hp.1, hs⟩⟩

theorem selection_dominates_eligible {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (b x : α) (xs : List α)
    (hx : x ∈ xs) (hg : Guard metric tol b x) :
    score x ≤ score (select score metric tol b xs) := by
  apply best_dominates score b _ x
  exact List.mem_filter.mpr ⟨hx, decide_eq_true hg⟩

theorem best_preserves_invariant {α} (score : α → Int) (P : α → Prop)
    (b : α) (xs : List α) (hb : P b) (hx : ∀ x ∈ xs, P x) : P (best score b xs) := by
  rcases best_mem score b xs with h | h
  · rw [h]; exact hb
  · exact hx _ h

/-! Query-count contract, including duplicate/cache exclusion. -/
def fresh {α : Type} [DecidableEq α] : List α → List α → List α
  | _, [] => []
  | cached, x::xs => if x ∈ cached then fresh cached xs else x :: fresh (x::cached) xs

theorem fresh_length {α} [DecidableEq α] (cached xs : List α) :
    (fresh cached xs).length ≤ xs.length := by
  induction xs generalizing cached with
  | nil => simp [fresh]
  | cons x xs ih =>
    simp only [fresh]
    split
    · have h := ih cached; simp only [List.length_cons]; omega
    · have h := ih (x::cached); simp only [List.length_cons]; omega

theorem fresh_mem {α} [DecidableEq α] (cached xs : List α) :
    ∀ x ∈ fresh cached xs, x ∈ xs ∧ x ∉ cached := by
  induction xs generalizing cached with
  | nil => simp [fresh]
  | cons y ys ih =>
    intro x hx
    simp only [fresh] at hx
    split at hx
    · have hp := ih cached x hx
      exact ⟨by simp [hp.1], hp.2⟩
    · simp only [List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact ⟨by simp, by assumption⟩
      · have hp := ih (y::cached) x hx
        exact ⟨by simp [hp.1], by intro hc; exact hp.2 (by simp [hc])⟩

theorem fresh_nodup {α} [DecidableEq α] (cached xs : List α) :
    (fresh cached xs).Nodup := by
  induction xs generalizing cached with
  | nil => simp [fresh]
  | cons x xs ih =>
    simp only [fresh]
    split
    · exact ih cached
    · apply List.nodup_cons.mpr
      constructor
      · intro hx
        have h := (fresh_mem (x::cached) xs x hx).2
        exact h (by simp)
      · exact ih (x::cached)

/-- Four named faces, each with exactly four attempted corners. They can coincide. -/
def blockAttempts {α} (face : Fin 4 → Fin 4 → α) : List α :=
  [face 0 0, face 0 1, face 0 2, face 0 3,
   face 1 0, face 1 1, face 1 2, face 1 3,
   face 2 0, face 2 1, face 2 2, face 2 3,
   face 3 0, face 3 1, face 3 2, face 3 3]

theorem face_query_bound {α} [DecidableEq α] (cached : List α)
    (face : Fin 4 → Fin 4 → α) :
    (blockAttempts face).length = 16 ∧ (fresh cached (blockAttempts face)).length ≤ 16 := by
  constructor
  · rfl
  · exact fresh_length cached (blockAttempts face)

/-- General 16-slot query map; cached incumbent and duplicate faces need no call. -/
theorem block_query_bound {α} [DecidableEq α] (cached : List α) (point : Nat → α) :
    (fresh cached ((List.range 16).map point)).length ≤ 16 := by
  have h := fresh_length cached ((List.range 16).map point)
  simpa only [List.length_map, List.length_range] using h

theorem nominal_query_cost : 4*4 = (16 : Nat) ∧ 16*2 = (32 : Nat) ∧
    2*(4*4) = (16*2 : Nat) ∧ 16 < (32 : Nat) := by decide

/-! The specified circle schedule, computed rather than hand-entered pair sets. -/
def initialOrder : List Nat := [0, 2, 4, 6, 7, 5, 3, 1]
def rotateTail : List Nat → List Nat
  | [] => []
  | h::t => h :: (t.getLast?.toList ++ t.dropLast)
def roundOrder : Nat → List Nat
  | 0 => initialOrder
  | n+1 => rotateTail (roundOrder n)
def roundPairs (r : Nat) : List (Nat × Nat) :=
  (List.range 4).map (fun p => ((roundOrder r)[p]!, (roundOrder r)[7-p]!))
def canonical (p : Nat × Nat) := (min p.1 p.2, max p.1 p.2)
def schedule : List (Nat × Nat) :=
  (List.range 7).flatMap (fun r => (roundPairs r).map canonical)

theorem schedule_starts_and_cycles :
    roundPairs 0 = [(0,1),(2,3),(4,5),(6,7)] ∧ roundOrder 7 = initialOrder := by decide

theorem every_round_partitions : ∀ r : Fin 7,
    (roundPairs r.val).length = 4 ∧
    (∀ i : Fin 8, ((roundPairs r.val).flatMap (fun p => [p.1,p.2])).count i.val = 1) ∧
    (roundPairs r.val).all (fun p => decide (p.1 < 8 ∧ p.2 < 8 ∧ p.1 ≠ p.2)) = true := by decide

theorem all_28_pairs_once : schedule.length = 28 ∧ schedule.Nodup ∧
    (∀ i j : Fin 8, i.val < j.val → schedule.count (i.val,j.val) = 1) := by decide

/-! Counterexample: actual selected pair corners improve; their merge loses. -/
def crossPairScore (x : Fin 8 → Int) :=
  x 0+x 2+x 4+x 6 - 2*(x 0*x 2+x 0*x 4+x 0*x 6+x 2*x 4+x 2*x 6+x 4*x 6)
def pairPoint (p : Fin 4) (s t : Int) : Fin 8 → Int :=
  fun i => if i.val = 2*p.val then s else if i.val = 2*p.val+1 then t else 0
def pairCornerList (p : Fin 4) : List (Fin 8 → Int) :=
  [pairPoint p (-1) (-1), pairPoint p (-1) 1, pairPoint p 1 (-1), pairPoint p 1 1]
def mergedWinner : Fin 8 → Int := fun i => if i.val % 2 = 0 then 1 else -1

theorem independently_selected_winners : ∀ p : Fin 4,
    (∀ i : Fin 8, best crossPairScore (fun _ => 0) (pairCornerList p) i = pairPoint p 1 (-1) i) ∧
    crossPairScore (pairPoint p 1 (-1)) = 1 := by decide +kernel

theorem merge_is_of_selected_pairs : ∀ p : Fin 4, ∀ i : Fin 8,
    i.val / 2 = p.val → mergedWinner i = pairPoint p 1 (-1) i := by decide

theorem merge_requires_validation :
    crossPairScore (fun _ => 0) = 0 ∧ crossPairScore mergedWinner = -8 ∧
    crossPairScore mergedWinner < crossPairScore (fun _ => 0) := by decide

/-! Honest variance algebra, not a probability or confidence-interval theorem. -/
theorem squared_weight_sums :
    4*((1 : Rat)/4)*((1 : Rat)/4) = 1/4 ∧
    32*((1 : Rat)/32)*((1 : Rat)/32) = 1/32 := by grind

theorem independent_noise_tradeoff (v : Rat) (hv : 0 < v) :
    v/4 = 8*(v/32) ∧ v/32 < v/4 := by grind

/-! No finite-query black-box optimality or finite-panel generalization guarantee. -/
def spike {α} [DecidableEq α] (z : α) (height : Rat) (x : α) := if x = z then height else 0

theorem unqueried_indistinguishable {α} [DecidableEq α] (queried : List α) (z : α)
    (hz : z ∉ queried) (height : Rat) :
    (∀ x ∈ queried, spike z height x = 0) ∧ spike z height z = height := by
  constructor
  · intro x hx
    have hn : x ≠ z := by intro he; rw [he] at hx; exact hz hx
    simp only [spike, ite_eq_right hn]
  · simp [spike]

theorem no_unconditional_optimality {α} [DecidableEq α] (queried : List α) (out z : α)
    (hz : z ∉ queried) (ho : out ≠ z) (height : Rat) (hp : 0 < height) :
    (∀ x ∈ queried, spike z height x = (fun _ : α => (0 : Rat)) x) ∧
    spike z height out < spike z height z := by
  have hs := unqueried_indistinguishable queried z hz height
  constructor
  · exact hs.1
  · rw [hs.2]
    simp only [spike, ite_eq_right ho]
    exact hp

/-- Even a diagonal quadratic is not its bilinear interpolant in the interior. -/
theorem diagonal_interior_counterexample :
    sample (fun x _ : Rat => 1-x*x) = sample (fun _ _ : Rat => 0) ∧
    (1-(0 : Rat)*0) > 0 := by decide +kernel

/-- Panel consists only of input 0; the equally weighted two-input population
has negative mean improvement. This does not assume sampling coverage. -/
def panelDelta (input : Nat) : Rat := if input = 0 then 1 else -3

theorem finite_panel_not_population : panelDelta 0 > 0 ∧
    (panelDelta 0 + panelDelta 1)/2 < 0 := by decide +kernel

/-! Degenerate faces and the integer-score bridge. -/
theorem degenerate_x_face (c : Corners) (hlo : c.ll = c.hl) (hhi : c.lh = c.hh)
    (x y : Rat) :
    bilinear c x y = (c.ll+c.lh)/2 + (c.lh-c.ll)/2*y ∧ x4 c = 0 ∧ mixed4 c = 0 := by
  unfold bilinear mean4 x4 y4 mixed4
  grind

theorem degenerate_y_face (c : Corners) (hlo : c.ll = c.lh) (hhi : c.hl = c.hh)
    (x y : Rat) :
    bilinear c x y = (c.ll+c.hl)/2 + (c.hl-c.ll)/2*x ∧ y4 c = 0 ∧ mixed4 c = 0 := by
  unfold bilinear mean4 x4 y4 mixed4
  grind

/-- Clip an incumbent 0 face of x^2 to [0,2]. The contrast is not the gradient at 0. -/
theorem asymmetric_face_not_incumbent_gradient :
    x4 (sample (fun u _ : Rat => (1+u)*(1+u))) / 4 = 2 ∧
    (2 : Rat)*(0 : Rat) = 0 := by decide +kernel

/-- Exact integer numerators embedded into Rat; there is no truncating /4 here. -/
theorem integer_numerator_bridge (ll lh hl hh : Int) :
    let c : Corners := ⟨(ll : Rat), (lh : Rat), (hl : Rat), (hh : Rat)⟩
    ((ll+lh+hl+hh : Int) : Rat) = mean4 c ∧
    ((hl+hh-ll-lh : Int) : Rat) = x4 c ∧
    ((lh+hh-ll-hl : Int) : Rat) = y4 c ∧
    ((ll+hh-lh-hl : Int) : Rat) = mixed4 c := by
  simp [Rat.intCast_add, Rat.intCast_sub, mean4, x4, y4, mixed4]

/-- Four signed i64 scores have numerator magnitude at most 2^65 in unbounded arithmetic. -/
theorem numerator_widening (ll lh hl hh : Int)
    (a : -9223372036854775808 ≤ ll ∧ ll ≤ 9223372036854775807)
    (b : -9223372036854775808 ≤ lh ∧ lh ≤ 9223372036854775807)
    (c : -9223372036854775808 ≤ hl ∧ hl ≤ 9223372036854775807)
    (d : -9223372036854775808 ≤ hh ∧ hh ≤ 9223372036854775807) :
    (-36893488147419103232 ≤ ll+lh+hl+hh ∧ ll+lh+hl+hh ≤ 36893488147419103232) ∧
    (-36893488147419103232 ≤ hl+hh-ll-lh ∧ hl+hh-ll-lh ≤ 36893488147419103232) ∧
    (-36893488147419103232 ≤ lh+hh-ll-hl ∧ lh+hh-ll-hl ≤ 36893488147419103232) ∧
    (-36893488147419103232 ≤ ll+hh-lh-hl ∧ ll+hh-lh-hl ≤ 36893488147419103232) := by omega

/-! Current production gate: fixed metric anchor, changing primary incumbent.
These pure functions specify the successful, well-formed, same-panel path.
Panel/shape checks and the Rust Result/index representation remain caller obligations. -/
def fixedSelect {α} {m : Nat} (score : α → Int) (metric : α → Fin m → Int)
    (tol : Fin m → Int) (anchor b : α) (predeclared : List α) : α :=
  best score b (predeclared.filter (fun x => decide (Guard metric tol anchor x)))

theorem fixed_anchor_contract {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor b : α) (xs : List α)
    (hb : Guard metric tol anchor b) :
    let out := fixedSelect score metric tol anchor b xs
    Guard metric tol anchor out ∧ score b ≤ score out ∧
      (out = b ∨ (out ∈ xs ∧ score b < score out)) := by
  dsimp only
  have hm := best_mem score b (xs.filter (fun x => decide (Guard metric tol anchor x)))
  have hn := best_monotone score b (xs.filter (fun x => decide (Guard metric tol anchor x)))
  by_cases he : fixedSelect score metric tol anchor b xs = b
  · rw [he]; exact ⟨hb, Int.le_refl _, Or.inl rfl⟩
  · have hs := best_changed_strict score b _ he
    have hf : fixedSelect score metric tol anchor b xs ∈
        xs.filter (fun x => decide (Guard metric tol anchor x)) := hm.resolve_left he
    have hp := List.mem_filter.mp hf
    exact ⟨of_decide_eq_true hp.2, hn, Or.inr ⟨hp.1, hs⟩⟩

theorem fixed_anchor_dominates {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor b x : α) (xs : List α)
    (hx : x ∈ xs) (hg : Guard metric tol anchor x) :
    score x ≤ score (fixedSelect score metric tol anchor b xs) := by
  apply best_dominates score b _ x
  exact List.mem_filter.mpr ⟨hx, decide_eq_true hg⟩

theorem fixed_anchor_ties {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor b : α) (xs : List α)
    (h : ∀ x ∈ xs, Guard metric tol anchor x → score x ≤ score b) :
    fixedSelect score metric tol anchor b xs = b := by
  apply best_holds_on_ties
  intro x hx
  have hp := List.mem_filter.mp hx
  exact h x hp.1 (of_decide_eq_true hp.2)

/-- Only anchor metrics matter: there is no comparison with anchor primary utility. -/
theorem fixed_anchor_metric_ext {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (a a' b : α) (xs : List α)
    (h : ∀ k, metric a k = metric a' k) :
    fixedSelect score metric tol a b xs = fixedSelect score metric tol a' b xs := by
  have hg : ∀ x, Guard metric tol a x = Guard metric tol a' x := by
    intro x
    unfold Guard
    simp only [h]
  unfold fixedSelect
  simp only [hg]

/-- Rejected incumbent is an error, not permission to retain an infeasible point. -/
def checkedFixedSelect {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor b : α) (xs : List α) : Option α :=
  if (∀ k, 0 ≤ tol k) ∧ Guard metric tol anchor b
  then some (fixedSelect score metric tol anchor b xs) else none

theorem checked_fixed_success {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor b out : α) (xs : List α)
    (h : checkedFixedSelect score metric tol anchor b xs = some out) :
    (∀ k, 0 ≤ tol k) ∧ Guard metric tol anchor b ∧ Guard metric tol anchor out ∧
      (out = b ∨ (out ∈ xs ∧ score b < score out)) := by
  unfold checkedFixedSelect at h
  split at h
  · rename_i hp
    have he := Option.some.inj h
    have hc := fixed_anchor_contract score metric tol anchor b xs hp.2
    rw [he] at hc
    exact ⟨hp.1, hp.2, hc.1, hc.2.2⟩
  · contradiction

/-- Panels, metric orientation, anchor and tolerances are fixed for this run.
The next predeclared batch may depend on the current point and round. -/
def fixedRun {α} {m : Nat} (score : α → Int) (metric : α → Fin m → Int)
    (tol : Fin m → Int) (anchor initial : α) (batch : Nat → α → List α) : Nat → α
  | 0 => initial
  | r+1 => let b := fixedRun score metric tol anchor initial batch r
           fixedSelect score metric tol anchor b (batch r b)

theorem fixed_run_invariant {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor initial : α)
    (batch : Nat → α → List α) (hb : Guard metric tol anchor initial) (r : Nat) :
    Guard metric tol anchor (fixedRun score metric tol anchor initial batch r) ∧
    score initial ≤ score (fixedRun score metric tol anchor initial batch r) := by
  induction r with
  | zero => exact ⟨hb, Int.le_refl _⟩
  | succ r ih =>
    have hc := fixed_anchor_contract score metric tol anchor
      (fixedRun score metric tol anchor initial batch r)
      (batch r (fixedRun score metric tol anchor initial batch r)) ih.1
    exact ⟨hc.1, Int.le_trans ih.2 hc.2.1⟩

theorem fixed_run_step_strict {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor initial : α)
    (batch : Nat → α → List α) (r : Nat)
    (h : fixedRun score metric tol anchor initial batch (r+1) ≠
      fixedRun score metric tol anchor initial batch r) :
    score (fixedRun score metric tol anchor initial batch r) <
      score (fixedRun score metric tol anchor initial batch (r+1)) := by
  exact best_changed_strict score _ _ h

/-- The invariant is an absolute floor, never r times the tolerance. -/
theorem fixed_run_no_drift {α} {m : Nat} (score : α → Int)
    (metric : α → Fin m → Int) (tol : Fin m → Int) (anchor initial : α)
    (batch : Nat → α → List α) (hb : Guard metric tol anchor initial) (r : Nat) (k : Fin m) :
    metric anchor k - tol k ≤ metric (fixedRun score metric tol anchor initial batch r) k := by
  have h := (fixed_run_invariant score metric tol anchor initial batch hb r).1 k
  omega

/-- Same protected scores: a renewed tolerance admits 92; fixed 100-5 rejects it. -/
theorem fixed_anchor_drift_example :
    select (fun x : Int => -x) (fun x (_ : Fin 1) => x) (fun _ => 5) 96 [92] = 92 ∧
    fixedSelect (fun x : Int => -x) (fun x (_ : Fin 1) => x) (fun _ => 5) 100 96 [92] = 96 := by
  decide +kernel

/-! Unsigned absolute difference: exact mathematical semantics of i128::abs_diff.
The subtraction below is UNBOUNDED specification arithmetic, not a signed machine subtraction. -/
def unsignedAbsDiff (base value : Int) : Nat := (base-value).natAbs

theorem unsigned_guard_iff (base value drop : Int) (hd : 0 ≤ drop) :
    (value ≥ base ∨ unsignedAbsDiff base value ≤ drop.toNat) ↔ base ≤ value+drop := by
  unfold unsignedAbsDiff
  by_cases h : base ≤ value
  · omega
  · have ha := Int.natAbs_of_nonneg (show 0 ≤ base-value by omega)
    omega

theorem unsigned_guard_all_iff {m : Nat} (base value drop : Fin m → Int)
    (hd : ∀ k, 0 ≤ drop k) :
    (∀ k, value k ≥ base k ∨ unsignedAbsDiff (base k) (value k) ≤ (drop k).toNat) ↔
      ∀ k, base k ≤ value k + drop k := by
  constructor <;> intro h k
  · exact (unsigned_guard_iff _ _ _ (hd k)).mp (h k)
  · exact (unsigned_guard_iff _ _ _ (hd k)).mpr (h k)

theorem unsigned_difference_fits (base value drop : Int)
    (hb : -170141183460469231731687303715884105728 ≤ base ∧
      base ≤ 170141183460469231731687303715884105727)
    (hv : -170141183460469231731687303715884105728 ≤ value ∧
      value ≤ 170141183460469231731687303715884105727)
    (hd : 0 ≤ drop ∧ drop ≤ 170141183460469231731687303715884105727) :
    unsignedAbsDiff base value ≤ 340282366920938463463374607431768211455 ∧
    (drop.toNat : Int) = drop ∧ drop.toNat ≤ 340282366920938463463374607431768211455 := by
  unfold unsignedAbsDiff
  by_cases h : 0 ≤ base-value
  · have ha := Int.natAbs_of_nonneg h; omega
  · have ha := Int.ofNat_natAbs_of_nonpos (show base-value ≤ 0 by omega); omega

/-! Actual normalized integer proposal: signed endpoint target, masking BEFORE maximum,
floor fraction in [0,1024], then signed truncation toward the base. -/
def signedTarget (base low high contrast : Int) : Int :=
  if 0 < contrast then high else if contrast < 0 then low else base

def activeMagnitude (base target contrast : Int) : Int :=
  if target = base then 0 else (contrast.natAbs : Int)

def largestMagnitude : {n : Nat} → (Fin n → Int) → Int
  | 0, _ => 0
  | n+1, v => max (v 0) (largestMagnitude (fun i : Fin n => v i.succ))

def fraction1024 (m largest : Int) : Int := (m*1024) / largest

def directedCoord (base target m largest : Int) : Int :=
  if largest = 0 then base else base + ((target-base)*fraction1024 m largest).tdiv 1024

def directed {n : Nat} (base low high contrast : Fin n → Int) : Fin n → Int :=
  let target := fun i => signedTarget (base i) (low i) (high i) (contrast i)
  let mag := fun i => activeMagnitude (base i) (target i) (contrast i)
  fun i => directedCoord (base i) (target i) (mag i) (largestMagnitude mag)

theorem signed_target_bounds (base low high contrast : Int) (h : low ≤ base ∧ base ≤ high) :
    low ≤ signedTarget base low high contrast ∧ signedTarget base low high contrast ≤ high := by
  unfold signedTarget; split <;> (try split) <;> omega

theorem active_magnitude_nonneg (base target contrast : Int) :
    0 ≤ activeMagnitude base target contrast := by
  unfold activeMagnitude; split <;> omega

theorem active_magnitude_mask (base target contrast : Int) (h : target = base) :
    activeMagnitude base target contrast = 0 := by simp [activeMagnitude, h]

theorem active_magnitude_positive (base target contrast : Int)
    (h : 0 < activeMagnitude base target contrast) : target ≠ base ∧ contrast ≠ 0 := by
  unfold activeMagnitude at h
  split at h
  · omega
  · constructor
    · assumption
    · intro he; simp [he] at h

theorem largest_magnitude_spec {n : Nat} (v : Fin n → Int) (hv : ∀ i, 0 ≤ v i) :
    0 ≤ largestMagnitude v ∧ (∀ i, v i ≤ largestMagnitude v) ∧
    (0 < largestMagnitude v → ∃ i, v i = largestMagnitude v) := by
  induction n with
  | zero => simp [largestMagnitude]
  | succ n ih =>
    have ht := ih (fun i : Fin n => v i.succ) (fun i => hv i.succ)
    constructor
    · simp only [largestMagnitude]; omega
    constructor
    · intro i
      refine Fin.cases ?_ (fun j => ?_) i
      · simp only [largestMagnitude]; omega
      · have hj := ht.2.1 j
        simp only [largestMagnitude]; omega
    · intro hp
      by_cases hh : largestMagnitude (fun i : Fin n => v i.succ) ≤ v 0
      · exact ⟨0, by simp only [largestMagnitude]; omega⟩
      · have hpos : 0 < largestMagnitude (fun i : Fin n => v i.succ) := by
          simp only [largestMagnitude] at hp; omega
        obtain ⟨i, hi⟩ := ht.2.2 hpos
        exact ⟨i.succ, by simp only [largestMagnitude]; omega⟩

theorem fraction_bounds (m largest : Int) (hm : 0 ≤ m ∧ m ≤ largest) (hl : 0 < largest) :
    0 ≤ fraction1024 m largest ∧ fraction1024 m largest ≤ 1024 ∧
    (m*1024).tdiv largest = fraction1024 m largest := by
  have hn : 0 ≤ m*1024 := by omega
  have hu : m*1024 ≤ 1024*largest := by omega
  have h0 := Int.ediv_nonneg hn (show 0 ≤ largest by omega)
  have h1 := Int.ediv_le_of_le_mul hl hu
  exact ⟨h0, h1, Int.tdiv_eq_ediv_of_nonneg hn⟩

theorem fraction_maximum (largest : Int) (hl : 0 < largest) :
    fraction1024 largest largest = 1024 := by
  exact Int.mul_ediv_cancel_left 1024 (by omega)

/-- No rational interpolation is substituted for Rust's signed truncation. -/
theorem scaled_trunc_between (delta fraction : Int) (hf : 0 ≤ fraction ∧ fraction ≤ 1024) :
    min 0 delta ≤ (delta*fraction).tdiv 1024 ∧
      (delta*fraction).tdiv 1024 ≤ max 0 delta := by
  by_cases h : 0 ≤ delta
  · have hp := Int.mul_nonneg h hf.1
    have hu := Int.mul_le_mul_of_nonneg_left hf.2 h
    have h0 := Int.tdiv_nonneg hp (show (0 : Int) ≤ 1024 by decide)
    have h1 := Int.tdiv_le_tdiv (show (0 : Int) < 1024 by decide) hu
    rw [Int.mul_tdiv_cancel _ (by decide)] at h1
    omega
  · have hn : 0 ≤ -delta := by omega
    have hp := Int.mul_nonneg hn hf.1
    have hu := Int.mul_le_mul_of_nonneg_left hf.2 hn
    have h0 := Int.tdiv_nonneg hp (show (0 : Int) ≤ 1024 by decide)
    have h1 := Int.tdiv_le_tdiv (show (0 : Int) < 1024 by decide) hu
    rw [Int.mul_tdiv_cancel _ (by decide)] at h1
    have he : (delta*fraction).tdiv 1024 = -((-delta*fraction).tdiv 1024) := by
      rw [Int.neg_mul, Int.neg_tdiv, Int.neg_neg]
    omega

theorem directed_coord_between (base target m largest : Int)
    (hm : 0 ≤ m ∧ m ≤ largest) (hl : 0 ≤ largest) :
    min base target ≤ directedCoord base target m largest ∧
      directedCoord base target m largest ≤ max base target := by
  unfold directedCoord
  split
  · omega
  · have hf := fraction_bounds m largest hm (by omega)
    have hd := scaled_trunc_between (target-base) (fraction1024 m largest) ⟨hf.1, hf.2.1⟩
    omega

theorem directed_coord_endpoint (base target largest : Int) (hl : 0 < largest) :
    directedCoord base target largest largest = target := by
  unfold directedCoord
  rw [ite_eq_right (by omega), fraction_maximum largest hl, Int.mul_tdiv_cancel _ (by decide)]
  omega

theorem directed_bounds {n : Nat} (base low high contrast : Fin n → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i) :
    ∀ i, low i ≤ directed base low high contrast i ∧ directed base low high contrast i ≤ high i := by
  intro i
  let target := fun i => signedTarget (base i) (low i) (high i) (contrast i)
  let mag := fun i => activeMagnitude (base i) (target i) (contrast i)
  have ht := signed_target_bounds (base i) (low i) (high i) (contrast i) (hb i)
  have hm := largest_magnitude_spec mag (fun i => active_magnitude_nonneg _ _ _)
  have hd := directed_coord_between (base i) (target i) (mag i) (largestMagnitude mag)
    ⟨active_magnitude_nonneg _ _ _, hm.2.1 i⟩ hm.1
  change low i ≤ directedCoord (base i) (target i) (mag i) (largestMagnitude mag) ∧
    directedCoord (base i) (target i) (mag i) (largestMagnitude mag) ≤ high i
  change low i ≤ target i ∧ target i ≤ high i at ht
  have hi := hb i
  omega

theorem directed_max_active_endpoint {n : Nat} (base low high contrast : Fin n → Int)
    (hp : ∃ i, 0 < activeMagnitude (base i)
      (signedTarget (base i) (low i) (high i) (contrast i)) (contrast i)) :
    ∃ i, directed base low high contrast i = signedTarget (base i) (low i) (high i) (contrast i) ∧
      directed base low high contrast i ≠ base i := by
  let target := fun i => signedTarget (base i) (low i) (high i) (contrast i)
  let mag := fun i => activeMagnitude (base i) (target i) (contrast i)
  have hm := largest_magnitude_spec mag (fun i => active_magnitude_nonneg _ _ _)
  obtain ⟨j, hj⟩ := hp
  have hpos : 0 < largestMagnitude mag := Int.lt_of_lt_of_le hj (hm.2.1 j)
  obtain ⟨i, hi⟩ := hm.2.2 hpos
  have ha : 0 < mag i := by omega
  have hn := (active_magnitude_positive _ _ _ ha).1
  have he : directed base low high contrast i = target i := by
    change directedCoord (base i) (target i) (mag i) (largestMagnitude mag) = target i
    rw [hi]
    exact directed_coord_endpoint _ _ _ hpos
  exact ⟨i, he, by rw [he]; exact hn⟩

/-- With no active magnitude the complete vector is unchanged. -/
theorem directed_no_active {n : Nat} (base low high contrast : Fin n → Int)
    (h : ∀ i, activeMagnitude (base i)
      (signedTarget (base i) (low i) (high i) (contrast i)) (contrast i) = 0) :
    directed base low high contrast = base := by
  have hz : ∀ n, largestMagnitude (fun _ : Fin n => (0 : Int)) = 0 := by
    intro n; induction n with
    | zero => rfl
    | succ n ih => simp [largestMagnitude, ih]
  funext i
  simp only [directed, h, hz, directedCoord, ite_true]

/-- Two-stage products fit i128, including opposite i64 endpoints.
The tighter internal bounds are 2^75 for magnitude*1024 and <2^74 for distance*fraction. -/
theorem normalized_product_widening (base target contrast fraction : Int)
    (hb : -9223372036854775808 ≤ base ∧ base ≤ 9223372036854775807)
    (ht : -9223372036854775808 ≤ target ∧ target ≤ 9223372036854775807)
    (hc : -36893488147419103232 ≤ contrast ∧ contrast ≤ 36893488147419103232)
    (hf : 0 ≤ fraction ∧ fraction ≤ 1024) :
    let m := activeMagnitude base target contrast
    0 ≤ m ∧ m ≤ 36893488147419103232 ∧
    0 ≤ m*1024 ∧ m*1024 ≤ 37778931862957161709568 ∧
    -18889465931478580853760 ≤ (target-base)*fraction ∧
      (target-base)*fraction ≤ 18889465931478580853760 ∧
    -170141183460469231731687303715884105728 ≤ (target-base)*fraction ∧
      (target-base)*fraction ≤ 170141183460469231731687303715884105727 ∧
    m*1024 ≤ 170141183460469231731687303715884105727 := by
  dsimp only
  have hm : 0 ≤ activeMagnitude base target contrast ∧
      activeMagnitude base target contrast ≤ 36893488147419103232 := by
    unfold activeMagnitude
    split
    · omega
    · by_cases h : 0 ≤ contrast
      · have ha := Int.natAbs_of_nonneg h; omega
      · have ha := Int.ofNat_natAbs_of_nonpos (show contrast ≤ 0 by omega); omega
  have hd : -18446744073709551615 ≤ target-base ∧
      target-base ≤ 18446744073709551615 := by omega
  have lo := Int.mul_le_mul_of_nonneg_right hd.1 hf.1
  have hi := Int.mul_le_mul_of_nonneg_right hd.2 hf.1
  omega

/-- Narrowing the widened final sum back to i64 is safe; no wraparound needed. -/
theorem directed_narrowing (base target m largest : Int)
    (hb : -9223372036854775808 ≤ base ∧ base ≤ 9223372036854775807)
    (ht : -9223372036854775808 ≤ target ∧ target ≤ 9223372036854775807)
    (hm : 0 ≤ m ∧ m ≤ largest) (hl : 0 ≤ largest) :
    -9223372036854775808 ≤ directedCoord base target m largest ∧
      directedCoord base target m largest ≤ 9223372036854775807 := by
  have h := directed_coord_between base target m largest hm hl
  omega

/-- The eight local proposal slots, before the two global exploration slots.
Pair winners and their merge are RAW, not projected. New points need their own
validation scores. `completePool` below specifies the complete ten-slot pool. -/
def proposalSlots {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) : List (Fin n → Int) :=
  [directedPoint, (fun i => halfstep (base i) (directedPoint i)),
   joint, (fun i => halfstep (base i) (joint i)),
   pairWinner 0, pairWinner 1, pairWinner 2, pairWinner 3]

/-- Extensional equality on finite integer points, with an executable decision. -/
instance {n : Nat} : DecidableEq (Fin n → Int) := fun f g =>
  if h : ∀ i, f i = g i then isTrue (funext h)
  else isFalse (fun he => h (congrFun he))

def proposalPool {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) : List (Fin n → Int) :=
  fresh [base] (proposalSlots base directedPoint joint pairWinner)

theorem proposal_pool_contract {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) :
    (proposalPool base directedPoint joint pairWinner).length ≤ 8 ∧
    (proposalPool base directedPoint joint pairWinner).Nodup ∧
    ∀ x ∈ proposalPool base directedPoint joint pairWinner,
      x ∈ proposalSlots base directedPoint joint pairWinner ∧ x ≠ base := by
  constructor
  · exact fresh_length [base] (proposalSlots base directedPoint joint pairWinner)
  constructor
  · exact fresh_nodup _ _
  · intro x hx
    have hp := fresh_mem [base] (proposalSlots base directedPoint joint pairWinner) x hx
    exact ⟨hp.1, by simpa only [List.mem_singleton] using hp.2⟩

theorem proposal_pool_bounds {n : Nat} (low high base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i)
    (hd : ∀ i, low i ≤ directedPoint i ∧ directedPoint i ≤ high i)
    (hj : ∀ i, low i ≤ joint i ∧ joint i ≤ high i)
    (hp : ∀ p i, low i ≤ pairWinner p i ∧ pairWinner p i ≤ high i) :
    ∀ x ∈ proposalPool base directedPoint joint pairWinner,
      ∀ i, low i ≤ x i ∧ x i ≤ high i := by
  intro x hx i
  have hm := (proposal_pool_contract base directedPoint joint pairWinner).2.2 x hx
  simp only [proposalSlots, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm.1 with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact hd i
  · exact halfstep_bounded _ _ _ _ (hb i) (hd i)
  · exact hj i
  · exact halfstep_bounded _ _ _ _ (hb i) (hj i)
  · exact hp 0 i
  · exact hp 1 i
  · exact hp 2 i
  · exact hp 3 i

/-- The first slot survives dedup exactly when it differs from the base. -/
theorem directed_in_pool {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (h : directedPoint ≠ base) :
    directedPoint ∈ proposalPool base directedPoint joint pairWinner := by
  change directedPoint ∈ fresh [base] (directedPoint :: _)
  rw [fresh, ite_eq_right (show directedPoint ∉ [base] by simpa using h)]
  exact List.mem_cons_self

/-! Bounded affine progress, not a nonlinear or global theorem.
Scores must be the exact SAME noiseless affine objective on all faces AND at
new proposals. Independent coordinate intervals; no coupling constraints here. -/
def affineFaceScore (c ax ay x y : Int) := c + ax*x + ay*y

theorem affine_face_numerators (c ax ay lx hx ly hy : Int) :
    (affineFaceScore c ax ay hx ly + affineFaceScore c ax ay hx hy -
      affineFaceScore c ax ay lx ly - affineFaceScore c ax ay lx hy) = 2*ax*(hx-lx) ∧
    (affineFaceScore c ax ay lx hy + affineFaceScore c ax ay hx hy -
      affineFaceScore c ax ay lx ly - affineFaceScore c ax ay hx ly) = 2*ay*(hy-ly) := by
  unfold affineFaceScore
  grind

/-- For an exact affine face, moving toward the sign-selected physical endpoint
cannot lose utility, and that endpoint strictly gains whenever it differs. -/
theorem affine_segment_gain (slope base low high contrast x : Int)
    (hb : low ≤ base ∧ base ≤ high) (hc : contrast = 2*slope*(high-low))
    (hx : min base (signedTarget base low high contrast) ≤ x ∧
      x ≤ max base (signedTarget base low high contrast)) :
    slope*base ≤ slope*x ∧
    (signedTarget base low high contrast ≠ base →
      slope*base < slope*signedTarget base low high contrast) := by
  by_cases hw : low < high
  · have hd : 0 < high-low := by omega
    by_cases hp : 0 < slope
    · have h2 : 0 < 2*slope := by omega
      have hpos : 0 < contrast := by rw [hc]; exact Int.mul_pos h2 hd
      have ht : signedTarget base low high contrast = high := by simp [signedTarget, hpos]
      rw [ht] at hx ⊢
      have hxb : base ≤ x := by omega
      constructor
      · exact Int.mul_le_mul_of_nonneg_left hxb (by omega)
      · intro hn
        exact Int.mul_lt_mul_of_pos_left (by omega) hp
    · by_cases hn : slope < 0
      · have hpos : 0 < (-2*slope)*(high-low) :=
          Int.mul_pos (by omega) hd
        have hneg : contrast < 0 := by rw [hc]; grind
        have ht : signedTarget base low high contrast = low := by simp [signedTarget, hneg, show ¬0 < contrast by omega]
        rw [ht] at hx ⊢
        have hxb : x ≤ base := by omega
        have hs : 0 < -slope := by omega
        have hle := Int.mul_le_mul_of_nonneg_left hxb (show 0 ≤ -slope by omega)
        constructor
        · grind
        · intro hne
          have hlt := Int.mul_lt_mul_of_pos_left (show low < base by omega) hs
          grind
      · have hs : slope = 0 := by omega
        have hz : contrast = 0 := by rw [hc, hs]; simp
        simp [signedTarget, hz, hs]
  · have he : low = high := by omega
    have ht : signedTarget base low high contrast = base := by unfold signedTarget; grind
    rw [ht] at hx ⊢
    have heq : x = base := by omega
    simp [heq]

def intTotal : {n : Nat} → (Fin n → Int) → Int
  | 0, _ => 0
  | n+1, f => f 0 + intTotal (fun i : Fin n => f i.succ)

def affineUtility {n : Nat} (constant : Int) (slope point : Fin n → Int) : Int :=
  constant + intTotal (fun i => slope i * point i)

theorem int_total_monotone {n : Nat} (f g : Fin n → Int) (h : ∀ i, f i ≤ g i) :
    intTotal f ≤ intTotal g := by
  induction n with
  | zero => exact Int.le_refl _
  | succ n ih =>
    have h0 := h 0
    have ht := ih (fun i => f i.succ) (fun i => g i.succ) (fun i => h i.succ)
    simp only [intTotal]; omega

theorem int_total_strict {n : Nat} (f g : Fin n → Int)
    (h : ∀ i, f i ≤ g i) (hp : ∃ i, f i < g i) : intTotal f < intTotal g := by
  induction n with
  | zero => obtain ⟨i, _⟩ := hp; exact Fin.elim0 i
  | succ n ih =>
    obtain ⟨i, hi⟩ := hp
    have h0 := h 0
    have ht := int_total_monotone (fun j : Fin n => f j.succ) (fun j => g j.succ)
      (fun j => h j.succ)
    refine Fin.cases ?_ (fun j hj => ?_) i hi
    · simp only [intTotal]; omega
    · have hs := ih (fun j => f j.succ) (fun j => g j.succ)
        (fun j => h j.succ) ⟨j, hj⟩
      simp only [intTotal]; omega

/-- Existence of ANY active affine direction implies strict gain of the integer
normalized proposal, despite floor fractions and signed truncation on other axes. -/
theorem affine_directed_progress {n : Nat} (constant : Int)
    (slope base low high contrast : Fin n → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i)
    (hc : ∀ i, contrast i = 2*slope i*(high i-low i))
    (hp : ∃ i, 0 < activeMagnitude (base i)
      (signedTarget (base i) (low i) (high i) (contrast i)) (contrast i)) :
    affineUtility constant slope base < affineUtility constant slope (directed base low high contrast) := by
  let target := fun i => signedTarget (base i) (low i) (high i) (contrast i)
  let mag := fun i => activeMagnitude (base i) (target i) (contrast i)
  have hm := largest_magnitude_spec mag (fun i => active_magnitude_nonneg _ _ _)
  have hseg : ∀ i, min (base i) (target i) ≤ directed base low high contrast i ∧
      directed base low high contrast i ≤ max (base i) (target i) := by
    intro i
    exact directed_coord_between _ _ _ _ ⟨active_magnitude_nonneg _ _ _, hm.2.1 i⟩ hm.1
  have hg := fun i => affine_segment_gain (slope i) (base i) (low i) (high i)
    (contrast i) (directed base low high contrast i) (hb i) (hc i) (hseg i)
  obtain ⟨i, hi, hn⟩ := directed_max_active_endpoint base low high contrast hp
  have hstrict : slope i*base i < slope i*directed base low high contrast i := by
    rw [hi] at hn ⊢
    exact (hg i).2 hn
  have ht := int_total_strict _ _ (fun i => (hg i).1) ⟨i, hstrict⟩
  unfold affineUtility
  omega

/-- No conflicting protected constraints: the actual directed point must be
eligible under the FIXED anchor. Utility is evaluated at that point, not copied
from a corner. Raw pair winners need not themselves satisfy that assumption. -/
theorem affine_pool_selection_progress {n m : Nat} (constant : Int)
    (slope base low high contrast joint anchor : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int)
    (metric : (Fin n → Int) → Fin m → Int) (tol : Fin m → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i)
    (hc : ∀ i, contrast i = 2*slope i*(high i-low i))
    (hp : ∃ i, 0 < activeMagnitude (base i)
      (signedTarget (base i) (low i) (high i) (contrast i)) (contrast i))
    (hbase : Guard metric tol anchor base)
    (hfree : Guard metric tol anchor (directed base low high contrast)) :
    let out := fixedSelect (affineUtility constant slope) metric tol anchor base
      (proposalPool base (directed base low high contrast) joint pairWinner)
    Guard metric tol anchor out ∧ affineUtility constant slope base < affineUtility constant slope out := by
  dsimp only
  have hg := affine_directed_progress constant slope base low high contrast hb hc hp
  have hn : directed base low high contrast ≠ base := by intro he; rw [he] at hg; omega
  have hm := directed_in_pool base (directed base low high contrast) joint pairWinner hn
  have hd := fixed_anchor_dominates (affineUtility constant slope) metric tol anchor base
    (directed base low high contrast) _ hm hfree
  have hf := fixed_anchor_contract (affineUtility constant slope) metric tol anchor base
    (proposalPool base (directed base low high contrast) joint pairWinner) hbase
  exact ⟨hf.1, Int.lt_of_lt_of_le hg hd⟩

/-- An available nonzero affine slope is an active direction; this condition
is stated in physical coordinates, not assumed candidate improvement. -/
theorem affine_movable_active (slope base low high contrast : Int)
    (hb : low ≤ base ∧ base ≤ high) (hc : contrast = 2*slope*(high-low))
    (hm : (0 < slope ∧ base < high) ∨ (slope < 0 ∧ low < base)) :
    0 < activeMagnitude base (signedTarget base low high contrast) contrast := by
  have hw : 0 < high-low := by omega
  rcases hm with hp | hn
  · have hpos : 0 < contrast := by
      rw [hc]; exact Int.mul_pos (by omega) hw
    have ha := Int.natAbs_of_nonneg (show 0 ≤ contrast by omega)
    simp only [signedTarget, ite_eq_left hpos, activeMagnitude,
      ite_eq_right (show high ≠ base by omega)]
    omega
  · have hpos : 0 < (-2*slope)*(high-low) := Int.mul_pos (by omega) hw
    have hneg : contrast < 0 := by rw [hc]; grind
    have ha := Int.ofNat_natAbs_of_nonpos (show contrast ≤ 0 by omega)
    simp only [signedTarget, ite_eq_right (show ¬0 < contrast by omega),
      ite_eq_left hneg, activeMagnitude, ite_eq_right (show low ≠ base by omega)]
    omega

theorem affine_available_progress {n : Nat} (constant : Int)
    (slope base low high contrast : Fin n → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i)
    (hc : ∀ i, contrast i = 2*slope i*(high i-low i))
    (hm : ∃ i, (0 < slope i ∧ base i < high i) ∨ (slope i < 0 ∧ low i < base i)) :
    affineUtility constant slope base < affineUtility constant slope (directed base low high contrast) := by
  obtain ⟨i, hi⟩ := hm
  apply affine_directed_progress constant slope base low high contrast hb hc
  exact ⟨i, affine_movable_active _ _ _ _ _ (hb i) (hc i) hi⟩

/-! Closed checks of the current integer construction, not nonlinear guarantees. -/
def curvedScore (x : Fin 8 → Int) := x 0 + 100*x 1 - 1000*x 0*x 0

def curvedContrast (i : Fin 8) : Int :=
  let p : Fin 4 := ⟨i.val / 2, by omega⟩
  let ll := curvedScore (pairPoint p (-1) (-1))
  let lh := curvedScore (pairPoint p (-1) 1)
  let hl := curvedScore (pairPoint p 1 (-1))
  let hh := curvedScore (pairPoint p 1 1)
  if i.val % 2 = 0 then hl+hh-ll-lh else lh+hh-ll-hl

def curvedDirected : Fin 8 → Int := directed (fun _ => 0) (fun _ => -1) (fun _ => 1) curvedContrast

theorem curved_face_only_stalls :
    (∀ p : Fin 4, ∀ i : Fin 8,
      best curvedScore (fun _ => 0) (pairCornerList p) i = 0) ∧
    (∀ x ∈ pairCornerList 0, curvedScore x < curvedScore (fun _ => 0)) := by
  decide +kernel

theorem curved_actual_contrasts : ∀ i : Fin 8,
    curvedContrast i = if i.val = 0 then 4 else if i.val = 1 then 400 else 0 := by
  decide +kernel

theorem curved_directed_wins :
    (∀ i : Fin 8, curvedDirected i = if i.val = 1 then 1 else 0) ∧
    curvedScore curvedDirected = 100 ∧ curvedScore (fun _ => 0) = 0 ∧
    curvedScore (fun i => halfstep 0 (curvedDirected i)) = 0 := by
  decide +kernel

/-- Outward contrast at a lower bound is masked BEFORE setting the maximum. -/
theorem masked_outward_example :
    directed (n := 2) (fun _ => 0) (fun _ => 0) (fun _ => 1)
      (fun i => if i.val = 0 then -36893488147419103232 else 1) 1 = 1 := by
  decide +kernel

theorem normalized_truncation_example :
    fraction1024 4 400 = 10 ∧ directedCoord 0 1 4 400 = 0 ∧
    directedCoord 0 (-1) 4 400 = 0 ∧ directedCoord 0 (-1) 400 400 = -1 := by
  decide +kernel

/-! Exhaustive dependency audit: every theorem in this file appears below. -/
#print axioms reconstruct
#print axioms bilinear_unique
#print axioms lerp_upper
#print axioms tensor_interpolation
#print axioms bilinear_upper
#print axioms cornerMax_spec
#print axioms maximum_at_corner
#print axioms unit_fraction_bounds
#print axioms rectangle_maximum
#print axioms quadratic_recovery
#print axioms total_congr
#print axioms total_zero
#print axioms total_add
#print axioms total_scale
#print axioms dot_affine2
#print axioms form_affine2_left
#print axioms form_affine2_right
#print axioms quadratic_plane_restriction
#print axioms pair_fixed_coordinates
#print axioms pair_moving_coordinates
#print axioms full_quadratic_pair_recovery
#print axioms untouched_product_no_alias
#print axioms hadamard_matches_doubling
#print axioms column_collision
#print axioms row_alias
#print axioms row_sign_square
#print axioms oldInteraction_scale
#print axioms legacy_alias_unit
#print axioms legacy_alias_general
#print axioms legacy_alias_seven
#print axioms false_SE_moments
#print axioms false_SE_exact
#print axioms clip_bounds
#print axioms endpoints_bounded
#print axioms fixed_endpoints
#print axioms halfstep_between
#print axioms halfstep_bounded
#print axioms widened_i64_arithmetic
#print axioms merge_bounded
#print axioms best_mem
#print axioms best_monotone
#print axioms best_dominates
#print axioms best_holds_on_ties
#print axioms best_changed_strict
#print axioms acceptance_contract
#print axioms selection_dominates_eligible
#print axioms best_preserves_invariant
#print axioms fresh_length
#print axioms fresh_mem
#print axioms fresh_nodup
#print axioms face_query_bound
#print axioms block_query_bound
#print axioms nominal_query_cost
#print axioms schedule_starts_and_cycles
#print axioms every_round_partitions
#print axioms all_28_pairs_once
#print axioms independently_selected_winners
#print axioms merge_is_of_selected_pairs
#print axioms merge_requires_validation
#print axioms squared_weight_sums
#print axioms independent_noise_tradeoff
#print axioms unqueried_indistinguishable
#print axioms no_unconditional_optimality
#print axioms diagonal_interior_counterexample
#print axioms finite_panel_not_population
#print axioms degenerate_x_face
#print axioms degenerate_y_face
#print axioms asymmetric_face_not_incumbent_gradient
#print axioms integer_numerator_bridge
#print axioms numerator_widening
#print axioms fixed_anchor_contract
#print axioms fixed_anchor_dominates
#print axioms fixed_anchor_ties
#print axioms fixed_anchor_metric_ext
#print axioms checked_fixed_success
#print axioms fixed_run_invariant
#print axioms fixed_run_step_strict
#print axioms fixed_run_no_drift
#print axioms fixed_anchor_drift_example
#print axioms unsigned_guard_iff
#print axioms unsigned_guard_all_iff
#print axioms unsigned_difference_fits
#print axioms signed_target_bounds
#print axioms active_magnitude_nonneg
#print axioms active_magnitude_mask
#print axioms active_magnitude_positive
#print axioms largest_magnitude_spec
#print axioms fraction_bounds
#print axioms fraction_maximum
#print axioms scaled_trunc_between
#print axioms directed_coord_between
#print axioms directed_coord_endpoint
#print axioms directed_bounds
#print axioms directed_max_active_endpoint
#print axioms directed_no_active
#print axioms normalized_product_widening
#print axioms directed_narrowing
#print axioms proposal_pool_contract
#print axioms proposal_pool_bounds
#print axioms directed_in_pool
#print axioms affine_face_numerators
#print axioms affine_segment_gain
#print axioms int_total_monotone
#print axioms int_total_strict
#print axioms affine_directed_progress
#print axioms affine_pool_selection_progress
#print axioms affine_movable_active
#print axioms affine_available_progress
#print axioms curved_face_only_stalls
#print axioms curved_actual_contrasts
#print axioms curved_directed_wins
#print axioms masked_outward_example
#print axioms normalized_truncation_example

/-! Complete production proposal pool: add two predeclared global endpoint trials.
They cost validation calls; their scores are NOT inferred from the local faces.
These lemmas extend the local-pool contracts, not a global-convergence claim. -/
def endpointTrial {n : Nat} (low high : Fin n → Int) (upper : Fin n → Bool) : Fin n → Int :=
  fun i => if upper i then high i else low i

theorem endpoint_trial_bounds {n : Nat} (low high : Fin n → Int) (upper : Fin n → Bool)
    (h : ∀ i, low i ≤ high i) :
    ∀ i, low i ≤ endpointTrial low high upper i ∧ endpointTrial low high upper i ≤ high i := by
  intro i
  have hi := h i
  unfold endpointTrial
  split <;> omega

def completeSlots {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (exploration : Fin 2 → Fin n → Int) : List (Fin n → Int) :=
  proposalSlots base directedPoint joint pairWinner ++ [exploration 0, exploration 1]

def completePool {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (exploration : Fin 2 → Fin n → Int) : List (Fin n → Int) :=
  fresh [base] (completeSlots base directedPoint joint pairWinner exploration)

theorem complete_pool_contract {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (exploration : Fin 2 → Fin n → Int) :
    (completePool base directedPoint joint pairWinner exploration).length ≤ 10 ∧
    (completePool base directedPoint joint pairWinner exploration).Nodup ∧
    ∀ x ∈ completePool base directedPoint joint pairWinner exploration,
      x ∈ completeSlots base directedPoint joint pairWinner exploration ∧ x ≠ base := by
  constructor
  · simpa [completePool, completeSlots, proposalSlots] using
      fresh_length [base] (completeSlots base directedPoint joint pairWinner exploration)
  constructor
  · exact fresh_nodup _ _
  · intro x hx
    have hp := fresh_mem [base] (completeSlots base directedPoint joint pairWinner exploration) x hx
    exact ⟨hp.1, by simpa only [List.mem_singleton] using hp.2⟩

theorem complete_pool_bounds {n : Nat} (low high base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (exploration : Fin 2 → Fin n → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i)
    (hd : ∀ i, low i ≤ directedPoint i ∧ directedPoint i ≤ high i)
    (hj : ∀ i, low i ≤ joint i ∧ joint i ≤ high i)
    (hp : ∀ p i, low i ≤ pairWinner p i ∧ pairWinner p i ≤ high i)
    (he : ∀ p i, low i ≤ exploration p i ∧ exploration p i ≤ high i) :
    ∀ x ∈ completePool base directedPoint joint pairWinner exploration,
      ∀ i, low i ≤ x i ∧ x i ≤ high i := by
  intro x hx i
  have hm := (complete_pool_contract base directedPoint joint pairWinner exploration).2.2 x hx
  simp only [completeSlots, proposalSlots, List.cons_append, List.nil_append,
    List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm.1 with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact hd i
  · exact halfstep_bounded _ _ _ _ (hb i) (hd i)
  · exact hj i
  · exact halfstep_bounded _ _ _ _ (hb i) (hj i)
  · exact hp 0 i
  · exact hp 1 i
  · exact hp 2 i
  · exact hp 3 i
  · exact he 0 i
  · exact he 1 i

theorem directed_in_complete_pool {n : Nat} (base directedPoint joint : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (exploration : Fin 2 → Fin n → Int)
    (h : directedPoint ≠ base) :
    directedPoint ∈ completePool base directedPoint joint pairWinner exploration := by
  change directedPoint ∈ fresh [base] (directedPoint :: _)
  rw [fresh, ite_eq_right (show directedPoint ∉ [base] by simpa using h)]
  exact List.mem_cons_self

/-- Baselines and label construction are separate. Training and validation
panels can have different per-call costs, so this is not a wall-time theorem. -/
theorem complete_round_query_bound (probes validation : Nat) (hp : probes ≤ 16) (hv : validation ≤ 10) :
    probes + validation ≤ 26 ∧ probes + validation < 32 := by omega

/-- Adding exploratory candidates cannot remove a feasible affine improvement. -/
theorem affine_complete_selection_progress {n m : Nat} (constant : Int)
    (slope base low high contrast joint anchor : Fin n → Int)
    (pairWinner : Fin 4 → Fin n → Int) (exploration : Fin 2 → Fin n → Int)
    (metric : (Fin n → Int) → Fin m → Int) (tol : Fin m → Int)
    (hb : ∀ i, low i ≤ base i ∧ base i ≤ high i)
    (hc : ∀ i, contrast i = 2*slope i*(high i-low i))
    (hp : ∃ i, 0 < activeMagnitude (base i)
      (signedTarget (base i) (low i) (high i) (contrast i)) (contrast i))
    (hbase : Guard metric tol anchor base)
    (hfree : Guard metric tol anchor (directed base low high contrast)) :
    let out := fixedSelect (affineUtility constant slope) metric tol anchor base
      (completePool base (directed base low high contrast) joint pairWinner exploration)
    Guard metric tol anchor out ∧ affineUtility constant slope base < affineUtility constant slope out := by
  dsimp only
  have hg := affine_directed_progress constant slope base low high contrast hb hc hp
  have hn : directed base low high contrast ≠ base := by intro he; rw [he] at hg; omega
  have hm := directed_in_complete_pool base (directed base low high contrast) joint pairWinner exploration hn
  have hd := fixed_anchor_dominates (affineUtility constant slope) metric tol anchor base
    (directed base low high contrast) _ hm hfree
  have hf := fixed_anchor_contract (affineUtility constant slope) metric tol anchor base
    (completePool base (directed base low high contrast) joint pairWinner exploration) hbase
  exact ⟨hf.1, Int.lt_of_lt_of_le hg hd⟩

#print axioms endpoint_trial_bounds
#print axioms complete_pool_contract
#print axioms complete_pool_bounds
#print axioms directed_in_complete_pool
#print axioms complete_round_query_bound
#print axioms affine_complete_selection_progress

end HyperSPSA
