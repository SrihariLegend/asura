import Std

/-!
Small arithmetic contracts for the search correctness regressions.
Run with: lean proofs/search_invariants.lean

These statements do not verify the Rust program, the network, chess strategy,
or playing strength. The executable regressions exercise the real code paths.
-/
namespace Asura.SearchInvariants

/-- Replacing learned PSQT by a different material proxy changes compensation. -/
theorem compensation_baseline (total psqt proxy : Int) :
    total - psqt = total - proxy ↔ psqt = proxy := by omega

/-- Rust signed division truncates towards zero, preserving perspective symmetry. -/
theorem psqt_negation (difference : Int) :
    (-difference).tdiv 32 = -(difference.tdiv 32) := Int.neg_tdiv difference 32

example : (-511 : Int).tdiv 32 = -15 := by decide
example : (511 : Int).tdiv 32 = 15 := by decide

/-- Applying correction history twice adds an extra correction, not a raw eval. -/
theorem duplicate_correction (raw correction : Int) :
    (raw + correction + correction) - (raw + correction) = correction := by omega

/-- Capacities suffice given these feature-family bounds for legal positions.
The geometry supplying the individual bounds is not formalized here. -/
theorem ft_capacity (psq threats pairs passers : Nat)
    (hpsq : psq ≤ 32) (hthreats : threats ≤ 32 * 8)
    (hpairs : pairs ≤ 120) (hpassers : passers ≤ 16) :
    psq + threats + pairs + passers ≤ 512 := by omega

theorem pk_capacity (pawns kings holes indicators : Nat)
    (hpawns : pawns ≤ 16) (hkings : kings ≤ 2)
    (hholes : holes ≤ 96) (hindicators : indicators ≤ 6) :
    pawns + kings + holes + indicators ≤ 128 := by omega

/-- Vertical rank flip (sq ↦ (7 - r) * 8 + f) inverts chessboard square parity (r + f) % 2.
Therefore, a perspective-relative pawn square-complex feature under rank flip must invert
the light/dark mask for Black to preserve color-swap symmetry. -/
theorem rank_flip_inverts_square_color (r f : Nat) (hr : r < 8) :
    ((7 - r) + f) % 2 ≠ (r + f) % 2 := by omega

/-- Clamping TT score to (beta + margin) in Singular Extensions can hide a genuinely
singular move: even when tt_score exceeds the best alternative (verify) by an arbitrarily
large gap (here 215 cp), the clamp pulls s_beta below verify. -/
def sBetaClamped (ttScore beta margin depth : Int) : Int :=
  min ttScore (beta + margin) - depth

def sBetaUnclamped (ttScore depth : Int) : Int :=
  ttScore - depth

theorem se_beta_clamp_hides_singularity :
    let ttScore : Int := 250
    let beta : Int := 20
    let margin : Int := 26
    let depth : Int := 16
    let verify : Int := 35
    ttScore - verify ≥ 200 ∧
    verify < sBetaUnclamped ttScore depth ∧
    verify ≥ sBetaClamped ttScore beta margin depth := by
  decide

/-- Stacking Exact-bound (+982) and child-cutoff (+1050) LMR penalties on top of a 2D
logarithmic table (1807 at depth 8, move 8) with weak PV bias (-560) and non-improving
penalty (+1120) produces a 4-ply reduction at depth 8 (leaving only 3 plies vs 7). -/
def pvRawReductionStacked (table pvBias nonImp exactPen cutoffPen : Int) : Int :=
  table + pvBias + nonImp + exactPen + cutoffPen

theorem pv_stacked_lmr_over_reduces :
    let table : Int := 1807
    let pvBias : Int := -560
    let nonImp : Int := 1120
    let exactPen : Int := 982
    let cutoffPen : Int := 1050
    (pvRawReductionStacked table pvBias nonImp exactPen cutoffPen).tdiv 1000 = 4 := by
  decide

/-- In the shipped asura_atn1.nnue checkpoint, uniform positive PK hidden biases (mean ≥ 36)
combined with row-0 weight sum (+746) and row-14 weight sum (-679) produce shifted gate values
of +839 and -763, exceeding the [-127, 127] clamp rails by more than 6x. -/
theorem pk_positive_bias_saturates_both_rails :
    let actFloor : Int := 36
    let row0WeightSum : Int := 746
    let row14WeightSum : Int := -679
    (actFloor * row0WeightSum).tdiv 32 > 6 * 127 ∧
    (actFloor * row14WeightSum).tdiv 32 < -6 * 127 := by
  decide

/-- When pk_gamma saturates at -127 on skip channels 14..15, the FiLM update
h + gamma * (h / 128) collapses h = 128 * q + r to q + r (attenuating the quotient q by 128x),
whereas an unsaturated identity gate (gamma = 0) preserves 128 * q + r. -/
def filmChannel (h q gamma : Int) : Int :=
  h + gamma * q

theorem saturated_minus_127_gate_attenuates_skip_by_128x (q r : Int) :
    let h := 128 * q + r
    filmChannel h q (-127) = q + r ∧
    filmChannel h q 0 = 128 * q + r := by
  dsimp [filmChannel]
  constructor <;> omega

/-- Pairwise accumulator multiplication (L = q1 + q2, R = k1 + k2) expands to the exact
all-pairs bilinear interaction sum Σ_i Σ_j q_i * k_j while requiring only O(1) additive
accumulator updates when a feature changes. -/
theorem pairwise_accumulator_product_expands_all_pairs (q1 q2 k1 k2 : Int) :
    (q1 + q2) * (k1 + k2) = q1 * k1 + q1 * k2 + q2 * k1 + q2 * k2 := by
  exact Int.add_mul q1 q2 (k1 + k2) |>.trans (by rw [Int.mul_add, Int.mul_add]; omega)

/-- 2-adic lower-triangular property of multiplication: for any modulus M = 2^k,
((x_lo + M * x_hi) * c) % M depends only on x_lo, erasing all bits at or above bit k. -/
theorem low_bit_multiply_mask_erases_high_bits (xLo xHi c m : Nat) :
    ((xLo + m * xHi) * c) % m = (xLo * c) % m := by
  have h : (xLo + m * xHi) * c = xLo * c + m * (xHi * c) := by
    rw [Nat.add_mul, Nat.mul_assoc]
  rw [h, Nat.add_mul_mod_self_left]

/-- Under SMP, when main thread reaches a 1024-node cadence boundary (1024 % 1024 == 0)
while helpers have searched 317 nodes, the aggregate count (1341) fails (total % 1024 == 0). -/
theorem smp_aggregate_mask_misses_polling_boundary :
    let mainNodes : Nat := 1024
    let helperNodes : Nat := 317
    mainNodes % 1024 = 0 ∧ (mainNodes + helperNodes) % 1024 ≠ 0 := by
  decide

/-- Shift-before-multiply FiLM (h + gamma * (h / 128)) has a downward cliff of -63 units
when h steps from 127 to 128 at gamma = -64, whereas multiply-before-shift is monotonic. -/
def filmShiftFirst (h gamma : Int) : Int :=
  h + gamma * (h.tdiv 128)

def filmMulFirst (h gamma : Int) : Int :=
  h + (gamma * h).tdiv 128

theorem film_shift_first_cliff_vs_mul_first_monotonic :
    filmShiftFirst 127 (-64) = 127 ∧
    filmShiftFirst 128 (-64) = 64 ∧
    filmMulFirst 127 (-64) ≤ filmMulFirst 128 (-64) := by
  decide

end Asura.SearchInvariants
