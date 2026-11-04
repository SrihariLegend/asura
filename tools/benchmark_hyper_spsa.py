#!/usr/bin/env python3
"""Freeze inputs and compare tuner designs. Never adopts parameters or runs training.

This measures old/new probe and proposal logic on the SAME repaired native
adapter and finite panels, not the entire historical executable. Phase timings
exclude label construction and cached incumbent measurements. Both methods use
the same guarded validation policy. Results are not Elo or population evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=Path)
    group.add_argument("--fens", type=Path)
    p.add_argument("--output", type=Path, required=True, help="new artifact directory")
    p.add_argument("--scan", type=int, default=40000)
    p.add_argument("--train", type=int, default=6)
    p.add_argument("--validation", type=int, default=6)
    p.add_argument("--test", type=int, default=6)
    p.add_argument("--nodes", default="500,1500,4500")
    p.add_argument("--label-nodes", type=int, default=6000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--blocks", default="2")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-seconds", type=int, default=300)
    p.add_argument("--synthetic-repeats", type=int, default=32)
    p.add_argument("--no-build", action="store_true", help="use an explicitly prebuilt release executable")
    a = p.parse_args()
    if (not 1 <= a.scan <= 1_000_000 or not 1 <= a.max_seconds <= 86400
            or not 1 <= a.repeats <= 64 or not 0 <= a.synthetic_repeats <= 64):
        p.error("invalid scan/deadline/repetition limit")
    model_source = a.model.resolve()
    source = (a.data or a.fens).resolve()
    if not source.is_file() or not model_source.is_file():
        p.error("model and data must be regular files")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not a.no_build:
        with (out / "build.log").open("w") as log:
            subprocess.run(["cargo", "build", "--release", "--lib", "--bin", "hyper_spsa"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute():
        target = ROOT / target
    binary = out / "hyper_spsa"
    shutil.copy2(target / "release" / "hyper_spsa", binary)
    model = out / "model.nnue"
    with model_source.open("rb") as src:
        payload = src.read(128 * 1024 * 1024 + 1)
    if not payload or len(payload) > 128 * 1024 * 1024:
        raise RuntimeError("empty model or model exceeds128 MiB snapshot limit")
    model.write_bytes(payload)
    del payload
    data = out / ("positions.bin" if a.data else "positions.fens")
    # Freeze exactly the bounded source prefix consumed by the scanner. Do not
    # call a prefix checksum a hash of a multi-gigabyte source dataset.
    if a.data:
        if source.stat().st_size % 32:
            raise RuntimeError("packed source length is not a multiple of32")
        with source.open("rb") as src, data.open("wb") as dst:
            dst.write(src.read(a.scan * 32))
    else:
        with source.open("rb") as src, data.open("wb") as dst:
            for _ in range(a.scan):
                line = src.readline(4097)
                if len(line) > 4096:
                    raise RuntimeError("FEN source line exceeds4096 bytes")
                if not line:
                    break
                dst.write(line)
    inputs = {str(model): digest(model), str(data): digest(data), str(binary): digest(binary)}
    if inputs[str(model)] != digest(model_source):
        raise RuntimeError("model changed while being snapshotted")
    source_files = sorted(list((ROOT / "src").rglob("*.rs")) + list((ROOT / "build").rglob("*.rs"))
                          + [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / ".cargo/config.toml"])
    command = [str(binary), "bench", "--model", str(model), "--data" if a.data else "--fens", str(data)]
    for name in ["scan", "train", "validation", "test", "nodes", "label_nodes", "threads", "blocks", "repeats", "seed", "max_seconds"]:
        command += ["--" + name.replace("_", "-"), str(getattr(a, name))]
    manifest = {
        "command": command, "input_sha256": inputs, "source_model": str(model_source),
        "source_data": str(source), "source_data_bytes": source.stat().st_size,
        "consumed_prefix_bytes": data.stat().st_size, "dataset_hash_scope": "copied bounded prefix only",
        "source_sha256": {str(f.relative_to(ROOT)): digest(f) for f in source_files},
        "driver_sha256": digest(Path(__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "scope": "same adapter/panels; phase times exclude labels/incumbent setup; no parameter adoption or Elo claim",
        "build_skipped": a.no_build,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    with (out / "native.tsv").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=a.max_seconds + 30)
    if result.returncode:
        raise RuntimeError(f"native benchmark failed: exit{result.returncode}; see native.tsv")
    text = (out / "native.tsv").read_text()
    if "native_nnue=true" not in text or "RESULT\tmode=bench\tadopted=false" not in text or "RESTORE\tok=true" not in text:
        raise RuntimeError("missing completion/model/restoration evidence")
    rows = []
    for line in text.splitlines():
        if line.startswith("BENCH\t"):
            rows.append(dict(item.split("=", 1) for item in line.split("\t")[1:]))
    blocks = a.blocks.split(",")
    if len(rows) != len(blocks) * a.repeats * 2:
        raise RuntimeError("incomplete benchmark rows")
    summary = {"wall_seconds": time.monotonic()-start, "native": {}, "scope": manifest["scope"]}
    for block in blocks:
        group_rows = [r for r in rows if r["block"] == block]
        design_rows = {name: [r for r in group_rows if r["design"] == name] for name in ["legacy", "paired"]}
        for name, rr in design_rows.items():
            if len(rr) != a.repeats or len({r["repeat"] for r in rr}) != a.repeats:
                raise RuntimeError("missing or duplicate repeated comparison")
            invalid = (any(int(r["probe_calls"]) != 32 for r in rr) if name == "legacy"
                       else any(int(r["probe_calls"]) > 16 for r in rr))
            if invalid:
                raise RuntimeError("probe query contract failed")
            if name == "paired" and any(int(r["cycle_calls"]) > 26 for r in rr):
                raise RuntimeError("complete-cycle query bound failed")
        stats = {}
        for phase in ["probe", "cycle"]:
            medians = {name: statistics.median(float(r[f"{phase}_ms"]) for r in rr) for name, rr in design_rows.items()}
            stats[phase] = {"median_ms": medians, "speedup_ratio": medians["legacy"] / medians["paired"]}
        summary["native"][block] = stats
    if a.synthetic_repeats:
        with (out / "synthetic.tsv").open("w") as log:
            subprocess.run([str(binary), "math-bench", "--queries", "512", "--repeats", str(a.synthetic_repeats), "--seed", str(a.seed)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
        groups = {}
        for line in (out / "synthetic.tsv").read_text().splitlines():
            if line.startswith("MATH\t"):
                r = dict(x.split("=", 1) for x in line.split("\t")[1:])
                if not 0 < int(r["calls"]) <= 512:
                    raise RuntimeError("synthetic objective-call cap violated")
                groups.setdefault((r["family"], r["design"]), []).append(r)
        if any(len(v) != a.synthetic_repeats for v in groups.values()) or len(groups) != 12:
            raise RuntimeError("incomplete synthetic benchmark")
        summary["synthetic"] = {
            "/".join(k): {
                "mean_regret": statistics.mean(int(r["regret"]) for r in v),
                "mean_calls": statistics.mean(int(r["calls"]) for r in v),
                "zero_regret": sum(int(r["regret"]) == 0 for r in v), "cases": len(v)
            } for k, v in groups.items()
        }
    if any(digest(Path(path)) != h for path, h in inputs.items()) or digest(model_source) != inputs[str(model)]:
        raise RuntimeError("frozen input changed")
    summary["status"] = "complete"
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
