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

end Asura.SearchInvariants
