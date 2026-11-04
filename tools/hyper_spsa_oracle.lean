import hyper_spsa

/-!
Executable adapter to the theorem-backed definitions in `hyper_spsa` (Std only).
This IO program is evidence, never a proof oracle. It adds no axioms or theorems.

Strict, space-separated decimal protocol (one response per record):
 C ll lh hl hh                         -- i64 corner scores; four /4 numerators
 B lo hi base step direction           -- i64; direction in {-1,0,1}; round zero
 S anchor incumbentUtility incumbentMetric tolerance n (utility metric)*n
                                      -- i128 scores, 0 <= n <= 64
 Q round constant (lo hi base step)*8 linear*8 matrix*64
                                      -- row-major, possibly nonsymmetric matrix
Q has round <= 895, coordinates in [-32,32], step in [1,64], coefficients in
[-1000,1000]. These bounds keep the Rust fixture objective exact in i64.

B uses clip, signedTarget, halfstep and fresh, not a copied arithmetic oracle.
S uses checkedFixedSelect, hence fixedSelect/Guard/best. The IDs preserve stable
index/tie semantics even for equal measurements. Q assembles the public probe
report using roundPairs, objective, contrasts, best, directed, endpointTrial,
completePool (including proposalSlots/halfstep/fresh), and fresh observations.
Only the global bit-mask schedule is adapter assembly; bounds and the complete
ten-slot pool use the installed theorem-backed definitions. No global-convergence
or universal software-correctness theorem is asserted here.
Malformed/empty/overlong inputs fail, never disappear from the comparison.
-/

open HyperSPSA

namespace Bridge

def i64min : Int := -9223372036854775808
def i64max : Int := 9223372036854775807
def i128min : Int := -170141183460469231731687303715884105728
def i128max : Int := 170141183460469231731687303715884105727

def require (p : Bool) (message : String) : Except String Unit :=
  if p then .ok () else .error message

def decimal (s : String) : Except String Int := do
  let chars := s.toList
  let digits := if chars.head? == some '-' then chars.drop 1 else chars
  require (!digits.isEmpty && digits.all (fun c => '0' ≤ c && c ≤ '9')) "invalid decimal"
  match s.toInt? with
  | some v => return v
  | none => throw "invalid integer"

def bounded (xs : List Int) (lo hi : Int) : Bool := xs.all (fun x => lo ≤ x && x ≤ hi)
def words (xs : List Int) : String := String.intercalate " " (xs.map toString)
def natInt (n : Nat) : Int := n

def integer (x : Rat) : Int := x.num
-- All uses of integer below have integer-valued inputs and polynomial operations.
-- C additionally checks the denominator; Q's polynomial has no division.
def numerators (c : Corners) : List Rat := [mean4 c, x4 c, y4 c, mixed4 c]

def contrastRecord (xs : List Int) : Except String String := do
  require (xs.length == 4 && bounded xs i64min i64max) "C shape/range"
  let c : Corners := ⟨xs[0]!, xs[1]!, xs[2]!, xs[3]!⟩
  let ns := numerators c
  require (ns.all (fun q => q.den == 1)) "C noninteger numerator"
  return "C " ++ words (ns.map integer)

def boundRecord (xs : List Int) : Except String String := do
  require (xs.length == 5 && bounded xs i64min i64max) "B shape/range"
  let lo := xs[0]!
  let hi := xs[1]!
  let base := xs[2]!
  let step := xs[3]!
  let direction := xs[4]!
  require (-1 ≤ direction && direction ≤ 1) "B direction"
  if !(lo ≤ base && base ≤ hi && 0 < step) then return "B ERR"
  let low := clip lo hi (base-step)
  let high := clip lo hi (base+step)
  let target := signedTarget base low high direction
  let calls := (fresh [base] [low, high]).length
  let trials := fresh [base] [target, halfstep base target, high, low]
  return "B " ++ words ([low, high, natInt calls, natInt trials.length] ++ trials)

structure Measured where
  id : Int
  utility : Int
  metric : Int

def selectorRecord (xs : List Int) : Except String String := do
  require (xs.length ≥ 5 && bounded xs i128min i128max) "S shape/range"
  let n := xs[4]!
  require (0 ≤ n && n ≤ 64 && xs.length == 5 + 2*n.toNat) "S candidate count"
  let anchor : Measured := ⟨-2, 0, xs[0]!⟩
  let incumbent : Measured := ⟨-1, xs[1]!, xs[2]!⟩
  let candidates := (List.range n.toNat).map fun i =>
    (⟨natInt i, xs[5+2*i]!, xs[6+2*i]!⟩ : Measured)
  let selected := checkedFixedSelect Measured.utility
    (fun x (_ : Fin 1) => x.metric) (fun _ => xs[3]!) anchor incumbent candidates
  return match selected with
    | none => "S ERR"
    | some x => "S " ++ toString x.id

abbrev Point := Fin 8 → Int

def pointList (p : Point) : List Int := (List.finRange 8).map p

def setPair (b : Point) (x y : Nat) (u v : Int) : Point :=
  fun i => if i.val == x then u else if i.val == y then v else b i

structure Face where
  x : Nat
  y : Nat
  low : List Int
  high : List Int
  scores : List Int
  contrast : Corners
  winner : Point
  score : Int

instance : Inhabited Face := ⟨⟨0, 0, [], [], [], ⟨0, 0, 0, 0⟩, fun _ => 0, 0⟩⟩

def faceWords (f : Face) : List Int :=
  [natInt f.x, natInt f.y] ++ f.low ++ f.high ++ f.scores ++
    (numerators f.contrast).map integer ++ pointList f.winner ++ [f.score]

def quadraticRecord (xs : List Int) : Except String String := do
  require (xs.length == 106) "Q shape"
  let round := xs[0]!
  require (0 ≤ round && round ≤ 895) "Q round"
  require (bounded [xs[1]!] (-1000) 1000 && bounded (xs.drop 34) (-1000) 1000) "Q coefficients"
  let base : Point := fun i => xs[4+4*i.val]!
  let lo : Point := fun i => xs[2+4*i.val]!
  let hi : Point := fun i => xs[3+4*i.val]!
  let step : Point := fun i => xs[5+4*i.val]!
  require ((List.finRange 8).all (fun i =>
    -32 ≤ lo i && lo i ≤ base i && base i ≤ hi i && hi i ≤ 32 &&
      1 ≤ step i && step i ≤ 64)) "Q axes"
  let low : Point := fun i => clip (lo i) (hi i) (base i-step i)
  let high : Point := fun i => clip (lo i) (hi i) (base i+step i)
  let linear : Vec 8 := fun i => xs[34+i.val]!
  let matrix : Fin 8 → Vec 8 := fun i j => xs[42+8*i.val+j.val]!
  let score := fun p : Point => integer (objective (xs[1]! : Rat) linear matrix (fun i => p i))
  let pairs := roundPairs (round.toNat % 7)
  let face := fun pair : Nat × Nat => Id.run do
    let x := pair.1
    let y := pair.2
    let l := pointList low
    let h := pointList high
    let corners := [setPair base x y l[x]! l[y]!, setPair base x y l[x]! h[y]!,
      setPair base x y h[x]! l[y]!, setPair base x y h[x]! h[y]!]
    let scores := corners.map score
    let winner := best score base corners
    return (corners, (⟨x, y, [l[x]!, l[y]!], [h[x]!, h[y]!], scores,
      ⟨scores[0]!, scores[1]!, scores[2]!, scores[3]!⟩, winner, score winner⟩ : Face))
  let faces := pairs.map face
  let reports := faces.map Prod.snd
  let attempts := faces.flatMap Prod.fst
  let observed := base :: fresh [base] attempts
  let contrast : Point := fun i =>
    reports.foldl (fun a f => if i.val == f.x then integer (x4 f.contrast)
      else if i.val == f.y then integer (y4 f.contrast) else a) 0
  let joint : Point := fun i =>
    reports.foldl (fun a f => if i.val == f.x || i.val == f.y then f.winner i else a) (base i)
  let directedPoint := directed base low high contrast
  let winners : Fin 4 → Point := fun p => (reports[p.val]!).winner
  let mask := ((round.toNat % 128)*73) % 128
  let exploration : Fin 2 → Point := fun side =>
    endpointTrial low high (fun i => mask.testBit i.val == (side.val == 1))
  let trials := completePool base directedPoint joint winners exploration
  let output := [natInt (observed.length-1)] ++ reports.flatMap faceWords ++
    [natInt observed.length] ++ observed.flatMap (fun p => pointList p ++ [score p]) ++
    [natInt trials.length] ++ trials.flatMap pointList
  return "Q " ++ words output

def respond (line : String) : Except String String := do
  require (line.length ≤ 16384) "line too long"
  let tokens := line.splitOn " " |>.filter (fun s => !s.isEmpty)
  match tokens with
  | [] => throw "empty record"
  | tag :: rest =>
    let xs ← rest.mapM decimal
    match tag with
    | "C" => contrastRecord xs
    | "B" => boundRecord xs
    | "S" => selectorRecord xs
    | "Q" => quadraticRecord xs
    | _ => throw "unknown record"

end Bridge

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let mut count := 0
  repeat
    let raw ← stdin.getLine
    if raw.isEmpty then break
    count := count + 1
    if count > 8192 then throw (IO.userError "too many records")
    let line := (raw.trimAscii).toString
    match Bridge.respond line with
    | .ok response => stdout.putStrLn response
    | .error message => throw (IO.userError s!"record {count}: {message}")
  if count == 0 then throw (IO.userError "empty fixture stream")
