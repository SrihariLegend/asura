#!/usr/bin/env python3
"""Audit the finite-cohort Lean proof; optionally evaluate frozen game cohorts.

The optional check compares exact finite enumeration in Python with the compiled
Lean definitions. It does not verify RNGs, game execution, an adaptive tuner or Elo.
No dependency downloads or writes into the repository/build tree.
"""
from collections import Counter
from fractions import Fraction as Q
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from check_hyper_spsa import audit_output, lean_code, need, sha

ROOT = Path(__file__).resolve().parents[1]


def average(values):
    return sum(values, Q(0)) / len(values)


def variance(values):
    center = average(values)
    return average([(v - center) ** 2 for v in values])


def literal(q):
    need(abs(q.numerator) < 2**256 and q.denominator < 2**256, "rational fixture too large")
    return f"(({q.numerator} : Rat) / {q.denominator})"


def vector(name, values):
    arms = " ".join(f"| {i} => {literal(v)}" for i, v in enumerate(values))
    return f"  let {name} : Vec {len(values)} := fun i => match i.val with {arms} | _ => 0\n"


def fixtures(analysis):
    result = json.loads(analysis.read_text())
    cases = []
    expected = {}
    for predictor in ("scorecard", "cheap_game"):
        groups = result["predictors"][predictor]["frozen_cohorts"]
        need(1 <= len(groups) <= 8, "cohort count outside bound")
        for group in groups:
            name = predictor + "/" + group["profile"]
            need(re.fullmatch(r"[a-z_/]+", name) is not None, "invalid cohort label")
            need(name not in {c[0] for c in cases}, "duplicate cohort")
            y = [Q(v) for v in group["y"]]
            p = [Q(v) for v in group["prediction"]]
            need(len(y) == len(p) and 1 <= len(y) <= 8, "invalid cohort shape")
            need(all(-1 <= q <= 1 for q in y + p), "game score/prediction outside bounds")
            cases.append((name, y, p))
            for k in (1, 2, 3):
                # Enumerate ACTUAL ordered uniform audit tuples, with replacement.
                outputs, ordinary = [], []
                for indices in itertools.product(range(len(y)), repeat=k):
                    outputs.append(average(p) + average([y[i] - p[i] for i in indices]))
                    ordinary.append(average([y[i] for i in indices]))
                expected[name, k] = [average(y), average(outputs), variance(outputs), variance(ordinary),
                                     variance([a - b for a, b in zip(y, p)]) / k]
    program = '''import audited_tuning
open AuditedTuning

def rat (q : Rat) : String := s!"{q.num}/{q.den}"
def emit (label : String) (n k : Nat) (y p : Vec n) : IO Unit := do
  let e := expect (estimator (k := k) y p)
  let v := randomVariance (estimator (k := k) y p)
  let o := randomVariance (ordinaryMean (k := k) y)
  let r := variance (residual y p) / (k : Rat)
  IO.println s!"CHECK\\t{label}\\t{k}\\t{rat (avg y)}\\t{rat e}\\t{rat v}\\t{rat o}\\t{rat r}"

def main : IO Unit := do
'''
    for i, (name, y, p) in enumerate(cases):
        program += vector(f"y{i}", y) + vector(f"p{i}", p)
        program += f'  for k in [1, 2, 3] do emit "{name}" {len(y)} k y{i} p{i}\n'
    return program, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    lean = shutil.which(args.lean)
    need(lean is not None, "Lean executable unavailable")
    source = ROOT / "proofs/audited_tuning.lean"
    code = lean_code(source.read_text())
    need(re.findall(r"(?m)^import\s+(\S+)\s*$", code) == ["Std"], "unexpected proof import")
    need(not re.search(r"\b(?:axiom|sorry|admit|native_decide|implemented_by|extern|unsafe|opaque)\b", code), "proof escape")
    names = re.findall(r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)", code)
    expected_names = ["AuditedTuning." + n for n in names]
    prints = re.findall(r"#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_'.]*)", code)
    need(names and len(names) == len(set(names)), "missing/duplicate theorems")
    need(Counter(expected_names) == Counter(prints), "incomplete/duplicate source dependency audits")
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="audited-tuning-", dir=args.work_dir))
    shutil.copyfile(source, scratch / source.name)
    env = dict(os.environ, LEAN_PATH=str(scratch))
    start = time.monotonic()
    run = subprocess.run([lean, "-t", "0", "-DwarningAsError=true", "-o", "audited_tuning.olean", source.name],
                         cwd=scratch, env=env, text=True, capture_output=True, timeout=120)
    (scratch / "proof.log").write_text(run.stdout + run.stderr)
    need(run.returncode == 0, f"proof compilation failed; see {scratch / 'proof.log'}")
    axioms = audit_output(run.stdout + run.stderr, expected_names)
    matched = 0
    if args.analysis:
        program, expected = fixtures(args.analysis)
        (scratch / "cohorts.lean").write_text(program)
        run = subprocess.run([lean, "-t", "0", "-DwarningAsError=true", "--run", "cohorts.lean"],
                             cwd=scratch, env=env, text=True, capture_output=True, timeout=120)
        (scratch / "cohorts.log").write_text(run.stdout + run.stderr)
        need(run.returncode == 0, f"cohort evaluation failed; see {scratch / 'cohorts.log'}")
        seen = set()
        for row in run.stdout.splitlines():
            fields = row.split("\t")
            need(len(fields) == 8 and fields[0] == "CHECK", "malformed Lean cohort response")
            key = fields[1], int(fields[2])
            need(key in expected and key not in seen, "unexpected/duplicate cohort response")
            actual = [Q(v) for v in fields[3:]]
            need(actual == expected[key], f"Python/Lean mismatch for {key}")
            need(actual[0] == actual[1] and actual[2] == actual[4], f"mean/variance contract failed for {key}")
            seen.add(key)
        need(seen == set(expected), "missing cohort response")
        matched = len(seen)
    report = {"proof_sha256": sha(source), "theorems": len(names), "audited": len(prints), "axioms": axioms,
              "cohort_rows_matched": matched, "wall_seconds": time.monotonic() - start,
              "analysis_sha256": sha(args.analysis) if args.analysis else None,
              "scope": "Finite exact probability design and optional frozen-cohort arithmetic only; not RNG, game execution, adaptive selection, unseen openings, SPSA campaign efficiency or Elo."}
    (scratch / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scratch": str(scratch), **report}, indent=2))


if __name__ == "__main__":
    main()
