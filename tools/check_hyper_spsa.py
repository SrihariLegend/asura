#!/usr/bin/env python3
"""Bounded Lean-to-compiled-public-Rust optimizer conformance (standard library only).

Run from anywhere; no network/dependencies, source extraction, training, or matches:
  python3 tools/check_hyper_spsa.py --target-dir /tmp/optimizer-bridge/target
  # Explicitly compare an alternate specification (reported separately):
  python3 tools/check_hyper_spsa.py --target-dir /tmp/optimizer-bridge/target --proof PATH

Always builds release --lib offline in an isolated target, never the repository's
main target. --lean selects an installed Lean executable; its matching leanc is
used. Every theorem declaration must have a dependency audit, and the compiler's
actual audit output must contain only Classical.choice/propext/Quot.sound.

C/B/S/Q are documented in hyper_spsa_oracle.lean. Python generates inputs and
compares complete canonical responses; it does NOT compute optimizer answers.
Q compares full eight-axis quadratic probe reports (including private normalized
proposal arithmetic through the public API). The 896 rounds cover the combined
7-pair/128-mask schedule. These are synthetic exact mathematical fixtures, NOT
NNUE/model/chess/IEEE/generalization/strength evidence. Multi-metric/panel-shape
rejection, arbitrary objectives, and native NNUE integration remain separate.

All source hashes, commands, build logs, fixtures, executable hashes and results
are retained in a unique scratch run directory, printed on success or failure.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SEED = 0x4859504552535053
MAX_CASES = 8192
ALLOW_AXIOMS = {"Classical.choice", "propext", "Quot.sound"}
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
I128_MIN, I128_MAX = -(1 << 127), (1 << 127) - 1
OWNED = [ROOT / "tools" / name for name in (
    "hyper_spsa_oracle.lean", "hyper_spsa_contract.rs", "check_hyper_spsa.py")]


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lean_code(text):
    """Remove nested Lean comments and strings, preserving tokens and newlines."""
    out, i, depth = [], 0, 0
    while i < len(text):
        if text.startswith("/-", i):
            depth += 1
            out.append("  ")
            i += 2
        elif depth and text.startswith("-/", i):
            depth -= 1
            out.append("  ")
            i += 2
        elif depth:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
        elif text.startswith("--", i):
            end = text.find("\n", i)
            i = len(text) if end == -1 else end
            out.append(" ")
        elif text[i] == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            need(i < len(text), "unterminated Lean string")
            i += 1
            out.append(" ")
        else:
            out.append(text[i])
            i += 1
    need(depth == 0, "unterminated Lean comment")
    return "".join(out)


def audit_source(proof, oracle):
    code = lean_code(proof)
    adapter = lean_code(oracle)
    forbidden = r"\b(?:axiom|sorry|admit|native_decide|implemented_by|extern|unsafe)\b"
    need(not re.search(forbidden, code + "\n" + adapter), "forbidden Lean proof/adapter escape")
    need(re.findall(r"(?m)^import\s+(\S+)\s*$", code) == ["Std"], "proof must import only Std")
    need(re.findall(r"(?m)^import\s+(\S+)\s*$", adapter) == ["hyper_spsa"], "oracle import changed")
    need(not re.search(r"\b(?:theorem|lemma)\b", adapter), "adapter must not add unaudited theorems")
    declarations = re.findall(r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)", code)
    audits = re.findall(r"#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_'.]*)", code)
    need(declarations and len(declarations) == len(set(declarations)), "missing/duplicate theorem declarations")
    need(Counter(declarations) == Counter(n.removeprefix("HyperSPSA.") for n in audits),
         "every theorem must have exactly one source dependency audit")
    return {"HyperSPSA." + n for n in declarations}


def audit_output(text, expected):
    pattern = r"'([^']+)'\s+(?:depends on axioms:\s*\[([^\]]*)\]|does not depend on any axioms)"
    entries = re.findall(pattern, text)
    need(Counter(n for n, _ in entries) == Counter(expected), "compiler audit missing/extra/duplicate theorem")
    used = set()
    for name, dependencies in entries:
        deps = {x.strip() for x in dependencies.split(",") if x.strip()}
        need(deps <= ALLOW_AXIOMS, f"unapproved theorem dependency: {name}: {deps - ALLOW_AXIOMS}")
        used |= deps
    need("sorry" not in text, "compiler reported sorry")
    return sorted(used)


def fixtures():
    rng = random.Random(SEED)
    rows = []

    def add(tag, xs):
        rows.append(tag + " " + " ".join(map(str, xs)))

    for corners in itertools.product([I64_MIN, -1, 0, 1, I64_MAX], repeat=4):
        add("C", corners)
    for _ in range(512):
        add("C", [rng.randint(I64_MIN, I64_MAX) for _ in range(4)])

    # Exhaustive small asymmetric/fixed bounds, odd negative halfsteps, clipping.
    for lo in range(-3, 4):
        for hi in range(lo, 4):
            for base in range(lo, hi + 1):
                for step in [1, 2, 3, 7]:
                    for direction in [-1, 0, 1]:
                        add("B", [lo, hi, base, step, direction])
    for lo, hi in [(I64_MIN, I64_MAX), (I64_MIN, I64_MIN + 7),
                   (I64_MAX - 7, I64_MAX), (I64_MIN, I64_MIN), (I64_MAX, I64_MAX)]:
        for base in sorted({lo, hi, (lo + hi) // 2}):
            for step in [1, 3, I64_MAX]:
                for direction in [-1, 0, 1]:
                    add("B", [lo, hi, base, step, direction])
    for _ in range(512):
        lo, hi = sorted([rng.randint(I64_MIN, I64_MAX) for _ in range(2)])
        add("B", [lo, hi, rng.randint(lo, hi), rng.randint(1, I64_MAX), rng.choice([-1, 0, 1])])
    for invalid in [(1, 0, 0, 1, 1), (0, 1, -1, 1, 1), (0, 1, 2, 1, -1),
                    (0, 1, 0, 0, 0), (0, 1, 1, -1, 1), (0, 1, 0, I64_MIN, 0)]:
        add("B", invalid)

    def selector(anchor, utility, metric, tolerance, candidates):
        add("S", [anchor, utility, metric, tolerance, len(candidates)] +
            [v for pair in candidates for v in pair])

    edges = [I128_MIN, I128_MIN + 1, -1, 0, 1, I128_MAX]
    for anchor, metric, tolerance in itertools.product(edges, repeat=3):
        selector(anchor, 0, metric, tolerance, list(zip(reversed(edges), edges)))
    for candidates in [[], [(10, 92)], [(1, 95), (1, 96)], [(0, 100)],
                       [(2, 94), (1, 95), (3, 100)], [(1, 95), (9, 92)],
                       [(i % 7, 100 - i % 11) for i in range(64)]]:
        selector(100, 0, 96, 5, candidates)
    # Renewing tolerance would accept 92; the fixed anchor must not.
    selector(100, 1, 95, 5, [(2, 92)])
    for _ in range(512):
        anchor = rng.choice(edges) if rng.randrange(3) == 0 else rng.randint(-100, 100)
        tolerance = rng.choice([-1, 0, 1, 5, I128_MAX])
        metric = rng.choice(edges + [anchor])
        utility = rng.choice(edges)
        candidates = [(rng.choice(edges), rng.choice(edges + [anchor])) for _ in range(rng.randrange(13))]
        selector(anchor, utility, metric, tolerance, candidates)

    def quad(round_number, axes, linear, matrix, constant=0):
        add("Q", [round_number, constant] + [v for axis in axes for v in axis] + linear + matrix)

    # Curvature hides progress from every corner; the directed proposal still moves.
    matrix = [0] * 64
    matrix[0] = -1000
    quad(0, [(-1, 1, 0, 1)] * 8, [1, 100] + [0] * 6, matrix)
    # Immobile/outward magnitude must not suppress a smaller active coordinate.
    quad(0, [(0, 1, 0, 1)] * 8, [-1000, 1] + [0] * 6, [0] * 64)
    # Signed truncation with unequal magnitudes and odd endpoint distances.
    quad(0, [(-9, 7, 0, 9)] * 8, [-1, 100, 0, 3, -10, 0, 99, 1], [0] * 64)
    quad(895, [(3, 3, 3, 1)] * 8, [1000] * 8, [1000] * 64, 1000)
    quad(127, [(-32, 32, 0, 64)] * 8, [0] * 8, [0] * 64)
    for round_number in range(896):
        axes = []
        for _ in range(8):
            lo, hi = sorted([rng.randint(-32, 32) for _ in range(2)])
            base = rng.choice([lo, hi, rng.randint(lo, hi)])
            axes.append((lo, hi, base, rng.randint(1, 64)))
        linear = [rng.randint(-1000, 1000) for _ in range(8)]
        matrix = [rng.randint(-1000, 1000) for _ in range(64)] if round_number % 3 else [0] * 64
        quad(round_number, axes, linear, matrix, rng.randint(-1000, 1000))
    need(0 < len(rows) <= MAX_CASES, "empty/oversized generated fixtures")
    need({row[0] for row in rows} == set("CBSQ"), "missing protocol fixture category")
    need(all(len(row) <= 16384 for row in rows), "oversized generated record")
    return rows


def malformed_fixtures(rows):
    q = next(row for row in rows if row.startswith("Q "))
    invalid = ["", "\n", "UNKNOWN 0\n", "C 1 2 3\n", "C 1 2 3 bad\n", "C 1 2 3 4 5\n",
               "C +1 2 3 4\n", "C 1_000 2 3 4\n", "C 0x1 2 3 4\n", "C 1\t2 3 4\n",
               f"C {I64_MAX + 1} 0 0 0\n", f"C {I64_MIN - 1} 0 0 0\n",
               "B 0 1 0 1\n", "B 0 1 0 1 2\n", "S 0 0 0 0 -1\n", "S 0 0 0 0 65\n",
               "S 0 0 0 0 1\n", "S 0 0 0 0 0 1 2\n", "S 0 0 0 0 1 nope 0\n",
               f"S 0 0 {I128_MIN - 1} 0 0\n", f"S 0 {I128_MAX + 1} 0 0 0\n",
               "Q 0\n", q.replace("Q 0 ", "Q 896 ", 1) + "\n", "C " + "1" * 16384 + "\n",
               "C 0 0 0 0\n" * (MAX_CASES + 1)]
    # A valid prefix must not let a broken later record be silently filtered.
    invalid += ["C 0 0 0 0\n" + bad for bad in ["\n", "C 1 2 bad 4\n", "S 0 0 0 0 1\n"]]
    return invalid


def response_shape(request, response):
    tag = request[0]
    parts = response.split(" ")
    need(parts[0] == tag, "response tag mismatch")
    if parts == [tag, "ERR"]:
        need(tag in {"B", "S"}, "unexpected error response")
        return
    need(all(re.fullmatch(r"-?(?:0|[1-9][0-9]*)", p) for p in parts[1:]), "malformed response integer")
    values = list(map(int, parts[1:]))
    if tag == "C":
        need(len(values) == 4, "C response shape")
    elif tag == "B":
        need(len(values) >= 4 and 0 <= values[2] <= 2 and 0 <= values[3] <= 4
             and len(values) == 4 + values[3], "B response shape")
    elif tag == "S":
        n = int(request.split()[5])
        need(len(values) == 1 and -1 <= values[0] < max(n, 0), "S response index")
    elif tag == "Q":
        need(len(values) >= 95 and 0 <= values[0] <= 16, "Q response prefix")
        index = 1 + 4 * 23  # calls then four complete pair reports
        observations = values[index]
        need(observations == values[0] + 1, "Q observation count")
        index += 1 + observations * 9
        need(index < len(values), "Q missing trial count")
        trials = values[index]
        need(0 <= trials <= 10 and len(values) == index + 1 + trials * 8, "Q trial shape")


class Run:
    def __init__(self, directory, env):
        self.directory, self.env = directory, env
        self.commands = []
        self.deadline = time.monotonic() + 900

    def execute(self, label, command, *, cwd=ROOT, data=None, timeout=180, success=True):
        remaining = self.deadline - time.monotonic()
        need(remaining > 0, "total 900s time budget exceeded")
        timeout = min(timeout, remaining)
        command = list(map(str, command))
        self.commands.append({"label": label, "argv": command, "cwd": str(cwd), "timeout": timeout})
        (self.directory / "commands.json").write_text(json.dumps(self.commands, indent=2) + "\n")
        # A timeout kills the complete compiler/cargo process group, not just its parent.
        with subprocess.Popen(command, cwd=cwd, env=self.env, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(data, timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                (self.directory / f"{label}.stdout").write_bytes(stdout)
                (self.directory / f"{label}.stderr").write_bytes(stderr)
                raise RuntimeError(f"{label}: timed out (process group killed)")
        (self.directory / f"{label}.stdout").write_bytes(stdout)
        (self.directory / f"{label}.stderr").write_bytes(stderr)
        if success:
            need(process.returncode == 0, f"{label}: exit {process.returncode}; see scratch logs")
        return process.returncode, stdout.decode("utf-8", errors="strict"), stderr.decode("utf-8", errors="strict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--lean", default="lean", help="installed Lean executable")
    parser.add_argument("--proof", type=Path, help="explicit alternate proof snapshot (reported as such)")
    parser.add_argument("--work-dir", type=Path, help="scratch log parent; defaults to target-dir/bridge-runs")
    args = parser.parse_args()
    default_target = Path(tempfile.gettempdir()) / "hyper-spsa-bridge" / "target"
    target = (args.target_dir or Path(os.environ.get("CARGO_TARGET_DIR", str(default_target)))).resolve()
    # Also reject ancestors, so choosing the repository itself cannot alter it.
    main_target = (ROOT / "target").resolve()
    need(target != ROOT and ROOT not in target.parents and target not in ROOT.parents,
         "target-dir must be outside the source repository")
    need(target != main_target and main_target not in target.parents and target not in main_target.parents,
         "refusing main target or its ancestors/descendants")
    work = (args.work_dir or target / "bridge-runs").resolve()
    need(work != ROOT and ROOT not in work.parents and work not in ROOT.parents, "work-dir must be scratch")
    # Respect the caller's affinity; Linux callers may pin with taskset.
    available = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
    proof = (args.proof or ROOT / "proofs/hyper_spsa.lean").resolve()
    need(proof.is_file(), "missing proof fixture; install proofs/hyper_spsa.lean or explicitly pass --proof")
    lean = shutil.which(args.lean)
    cargo, rustc = shutil.which("cargo"), shutil.which("rustc")
    need(lean and cargo and rustc, "missing Lean/Cargo/rustc tool")
    work.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=work))
    print(f"scratch={directory}", flush=True)
    env = os.environ.copy()
    for name in ["CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "LEAN_PATH"]:
        env.pop(name, None)
    env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", CARGO_NET_OFFLINE="true",
               RUSTFLAGS="-C target-cpu=native -C panic=abort", RUSTC=rustc, LEAN_NUM_THREADS="4")
    run = Run(directory, env)
    summary = []
    try:
        # Hash sources before AND after building/running: fail on a concurrent edit.
        paths = set(OWNED + [proof, ROOT / "Cargo.toml", ROOT / "Cargo.lock"])
        paths.update((ROOT / "src").rglob("*.rs"))
        paths.update(p for p in (ROOT / "build").rglob("*") if p.is_file())
        paths.update(p for p in (ROOT / ".cargo").rglob("*") if p.is_file())
        sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in sorted(paths)}
        (directory / "sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        declarations = audit_source(proof.read_text(), OWNED[0].read_text())
        shutil.copyfile(proof, directory / "hyper_spsa.lean")
        need(sha(directory / "hyper_spsa.lean") == sha(proof), "proof snapshot changed while copying")
        _, lean_version, _ = run.execute("lean-version", [lean, "--version"], timeout=20)
        _, rust_version, _ = run.execute("rust-version", [rustc, "--version"], timeout=20)
        _, prefix, _ = run.execute("lean-prefix", [lean, "--print-prefix"], timeout=20)
        # Pin the actual toolchain before changing cwd to scratch (elan otherwise
        # could resolve a different project/default toolchain there).
        lean = Path(prefix.strip()) / "bin/lean"
        leanc = Path(prefix.strip()) / "bin/leanc"
        need(lean.is_file() and leanc.is_file(), "missing matching Lean/leanc")
        run.env["LEAN_PATH"] = str(directory)
        _, proof_output, proof_errors = run.execute("lean-proof", [lean, "-t", "0", "-DwarningAsError=true", "-o", "hyper_spsa.olean", "-c", "hyper_spsa.c",
                                                                   "hyper_spsa.lean"], cwd=directory, timeout=240)
        used = audit_output(proof_output + proof_errors, declarations)
        run.execute("lean-adapter", [lean, "-t", "0", "-DwarningAsError=true", "-o", directory / "oracle.olean", "-c", directory / "oracle.c", OWNED[0]])
        oracle = directory / "oracle"
        run.execute("lean-link", [leanc, "-O1", "-o", oracle, directory / "oracle.c", directory / "hyper_spsa.c"])
        _, cargo_output, _ = run.execute("cargo-lib", [cargo, "build", "--offline", "--locked", "--release", "--lib",
                                                               "--target-dir", target, "--message-format=json-render-diagnostics"], timeout=240)
        metadata = tomllib.loads((ROOT / "Cargo.toml").read_text())
        crate = metadata.get("lib", {}).get("name", metadata["package"]["name"].replace("-", "_"))
        artifacts = []
        for line in cargo_output.splitlines():
            record = json.loads(line)  # Never discard broken build records.
            if record.get("reason") == "compiler-artifact" and record["target"]["name"] == crate and "lib" in record["target"]["kind"]:
                artifacts.extend(Path(f).resolve() for f in record["filenames"] if f.endswith(".rlib"))
        need(len(artifacts) == 1 and artifacts[0].is_file() and target in artifacts[0].parents,
             "missing/ambiguous/non-isolated production library artifact")
        library = artifacts[0]
        contract = directory / "contract"
        run.execute("rust-adapter", [rustc, "--edition=2024", "-C", "target-cpu=native", "-C", "panic=abort", "-C", "opt-level=2",
                                     "--extern", f"production={library}", "-L", f"dependency={library.parent / 'deps'}",
                                     OWNED[1], "-o", contract])
        rows = fixtures()
        data = ("\n".join(rows) + "\n").encode("ascii")
        (directory / "fixtures.txt").write_bytes(data)
        _, expected, lean_stderr = run.execute("lean-responses", [oracle], data=data, timeout=90)
        _, actual, rust_stderr = run.execute("rust-responses", [contract], data=data, timeout=90)
        need(not lean_stderr and not rust_stderr, "unexpected adapter stderr")
        lean_rows, rust_rows = expected.splitlines(), actual.splitlines()
        need(len(lean_rows) == len(rust_rows) == len(rows), "missing/extra responses")
        need(expected.endswith("\n") and actual.endswith("\n"), "unterminated response")
        for number, (request, left, right) in enumerate(zip(rows, lean_rows, rust_rows), 1):
            response_shape(request, left)
            response_shape(request, right)
            need(left == right, f"mismatch at record {number}: {request}\nLean: {left}\nRust: {right}")
        negative = malformed_fixtures(rows)
        for index, bad in enumerate(negative):
            (directory / f"malformed-{index:02}.txt").write_text(bad)
            for label, executable in [("lean", oracle), ("rust", contract)]:
                status, _, errors = run.execute(f"reject-{index:02}-{label}", [executable], data=bad.encode(), timeout=10, success=False)
                need(status > 0 and errors, f"{label} silently accepted/crashed on malformed fixture {index}")
        need(all(sha(p) == sources[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] for p in paths),
             "source changed during conformance run; rerun against a stable snapshot")
        build_paths = [library, oracle, contract, directory / "hyper_spsa.olean", directory / "hyper_spsa.c",
                       directory / "oracle.olean", directory / "oracle.c", directory / "fixtures.txt"]
        builds = {str(p): sha(p) for p in build_paths}
        (directory / "builds.json").write_text(json.dumps(builds, indent=2, sort_keys=True) + "\n")
        counts = Counter(row[0] for row in rows)
        installed = proof == (ROOT / "proofs/hyper_spsa.lean").resolve()
        summary = [f"proof={'installed' if installed else 'explicit-snapshot'} theorems={len(declarations)} audited={len(declarations)} "
                   f"axioms={','.join(used)}",
                   f"seed={SEED} CPU={','.join(map(str, sorted(available)))}",
                   "responses " + " ".join(f"{tag}={counts[tag]}" for tag in "CBSQ") + f" total={len(rows)} matched={len(rows)}",
                   f"malformed={len(negative)} lean_rejected={len(negative)} rust_rejected={len(negative)} rejection_checks={2*len(negative)}",
                   f"proof_sha256={sha(proof)}", f"production_library_sha256={sha(library)}",
                   f"fixtures_sha256={sha(directory / 'fixtures.txt')}",
                   "PASS: bounded executable Lean/public-Rust optimizer conformance",
                   "NOT COVERED: panel-shape/multi-metric errors, arbitrary objectives, native NNUE, IEEE, or playing strength."]
        if not installed:
            summary.append("INCOMPLETE: repository specification not used; rerun without --proof for its contract.")
        manifest = {"summary": summary, "lean_version": lean_version.strip(), "rust_version": rust_version.strip(),
                    "sources": sources, "builds": builds, "seed": SEED, "counts": counts,
                    "malformed": len(negative), "proof_mode": "installed" if installed else "explicit-snapshot"}
        (directory / "result.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (directory / "success.log").write_text("\n".join(summary) + "\n")
        print("\n".join(summary))
    except Exception as error:
        (directory / "failure.log").write_text(str(error) + "\n")
        raise


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
