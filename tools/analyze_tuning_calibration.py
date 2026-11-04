#!/usr/bin/env python3
"""Descriptive analysis of a frozen tuning-calibration pilot, not an Elo test.

Uses exact rational arithmetic. Fits on the registered calibration openings only.
The held-out hindsight slope is labeled an oracle diagnostic, never a fitted model
for deployment. Repeated candidates share opening families; rows are not claimed
independent. No significance test or parameter-adoption decision is made.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time


def mean(values):
    if not values:
        raise ValueError("empty mean")
    return sum(values, Q(0)) / len(values)


def energy(values):
    return sum((v * v for v in values), Q(0))


def centered(values):
    m = mean(values)
    return [v - m for v in values]


def fit(rows, key):
    xs = [r[key] for r in rows]
    ys = [r["y"] for r in rows]
    xc, yc = centered(xs), centered(ys)
    vx = energy(xc)
    beta = sum((x * y for x, y in zip(xc, yc)), Q(0)) / vx if vx else Q(0)
    return mean(ys) - beta * mean(xs), beta


def numeric(value):
    return {"exact": str(value), "value": float(value)}


def analyze(root: Path):
    started = time.process_time()
    completed = json.loads((root / "run-complete.json").read_text())
    if completed.get("status") != "complete" or not completed.get("model_unchanged"):
        raise ValueError("require a complete verified run with unchanged model")
    prereg = json.loads((root / "preregister.json").read_text())
    sealed = json.loads((root / "before-games.json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(root / "preregister.json") != sealed["preregister_sha256"]:
        raise ValueError("registration changed after scoring")
    if digest(root / "scorecard.tsv") != sealed["scorecard_sha256"]:
        raise ValueError("scorecard changed after sealing")
    openings = json.loads((root / "openings.json").read_text())
    fen_ids = {" ".join(o["fen"].split()[:4]): o["id"] for o in openings}
    scores = {}
    for line in (root / "scorecard.tsv").read_text().splitlines():
        if line.startswith("SCORE\t"):
            _, profile, fen, utility, _ = line.split("\t")
            key = (profile, fen_ids[" ".join(fen.split()[:4])])
            if key in scores:
                raise ValueError("duplicate scorecard row")
            scores[key] = Q(utility)
    profiles = ("ll", "lh", "hl", "hh")
    expected = {(c, o["id"]) for c in ("base", *profiles) for o in openings}
    if set(scores) != expected:
        raise ValueError("missing/extra scorecard rows")
    cohorts = {}
    for label in ("cheap", "target"):
        data = json.loads((root / f"{label}-verified.json").read_text())
        cohorts[label] = {}
        for profile in profiles:
            match = data["matches"][profile]
            if digest(root / f"{label}-{profile}" / "games.pgn") != match["pgn_sha256"]:
                raise ValueError("verified PGN changed")
            for pair in match["pairs"]:
                key = (profile, pair["opening"])
                if key in cohorts[label] or pair["profile"] != profile:
                    raise ValueError("duplicate/mismatched game pair")
                cohorts[label][key] = Q(pair["signed_pair_score"])
        if set(cohorts[label]) != expected - {( "base", o["id"]) for o in openings}:
            raise ValueError("missing/extra game pairs")
    rows = [{"profile": c, "opening": o["id"], "y": cohorts["target"][c, o["id"]],
             "scorecard": scores[c, o["id"]] - scores["base", o["id"]],
             "cheap_game": cohorts["cheap"][c, o["id"]]}
            for c in profiles for o in openings]
    train_ids = set(prereg["analysis"]["calibration_openings"])
    test_ids = set(prereg["analysis"]["heldout_openings"])
    if train_ids & test_ids or train_ids | test_ids != set(fen_ids.values()):
        raise ValueError("invalid calibration/held-out split")
    train = [r for r in rows if r["opening"] in train_ids]
    test = [r for r in rows if r["opening"] in test_ids]
    output = {"scope": "Small, shared-opening descriptive calibration pilot. Not Elo, promotion, or equal-compute SPSA evidence.",
              "registered_sha256": digest(root / "preregister.json"), "predictors": {}, "candidate_means": {},
              "cost": {"run_wall_seconds": completed["wall_seconds"],
                       "metered_run_child_cpu_seconds": completed["child_cpu_seconds"],
                       "failed_setup_child_cpu_seconds": prereg["failed_setup_child_cpu_seconds"],
                       "excluded_not_free": "Historical opening construction, cached build and interactive development/debug/proof costs are not completely instrumented. No all-in saving is claimed.",
                       "cheap_games_child_cpu_seconds": completed["phases"]["cheap"]["child_cpu_seconds"],
                       "target_games_child_cpu_seconds": completed["phases"]["target"]["child_cpu_seconds"]}}
    for key in ("scorecard", "cheap_game"):
        alpha, beta = fit(train, key)
        predicted = [min(Q(1), max(Q(-1), alpha + beta * r[key])) for r in test]
        within_y, within_x, within_error = [], [], []
        groups = []
        for profile in profiles:
            indices = [i for i, r in enumerate(test) if r["profile"] == profile]
            y = [test[i]["y"] for i in indices]
            x = [test[i][key] for i in indices]
            p = [predicted[i] for i in indices]
            errors = [a - b for a, b in zip(y, p)]
            within_y.extend(centered(y))
            within_x.extend(centered(x))
            within_error.extend(centered(errors))
            groups.append({"profile": profile, "y": [str(v) for v in y], "prediction": [str(v) for v in p],
                           "opening_ids": [test[i]["opening"] for i in indices]})
        vy, vx, ve = energy(within_y), energy(within_x), energy(within_error)
        cov = sum((x * y for x, y in zip(within_x, within_y)), Q(0))
        rho2 = cov * cov / (vx * vy) if vx and vy else None
        ratio = ve / vy if vy else None
        output["predictors"][key] = {
            "calibration_rows": len(train), "heldout_rows": len(test), "heldout_opening_families": len(test_ids),
            "intercept": numeric(alpha), "slope": numeric(beta), "clipped_to_signed_score_range": True,
            "heldout_mse": numeric(mean([(r["y"] - p) ** 2 for r, p in zip(test, predicted)])),
            "heldout_bias_prediction_minus_y": numeric(mean([p - r["y"] for r, p in zip(test, predicted)])),
            "within_candidate_y_ss": numeric(vy), "within_candidate_residual_ss": numeric(ve),
            "within_candidate_residual_variance_ratio": numeric(ratio) if ratio is not None else None,
            "necessary_100x_residual_condition_on_this_cohort": ratio <= Q(1, 100) if ratio is not None else None,
            "necessary_1000x_residual_condition_on_this_cohort": ratio <= Q(1, 1000) if ratio is not None else None,
            "hindsight_within_candidate_linear_rho_squared": numeric(rho2) if rho2 is not None else None,
            "hindsight_free_cheap_linear_sample_saving_ceiling": numeric(1 / (1 - rho2)) if rho2 is not None and rho2 < 1 else None,
            "hindsight_scope": "Best single LINEAR slope after seeing held-out outcomes; optimistic algebraic diagnostic only, not a held-out fit or a limit on richer/nonlinear predictors.",
            "zero_variance": {"y": not bool(vy), "x": not bool(vx)}, "frozen_cohorts": groups}
    for profile in profiles:
        selected = [r for r in test if r["profile"] == profile]
        output["candidate_means"][profile] = {key: numeric(mean([r[key] for r in selected]))
                                               for key in ("y", "cheap_game", "scorecard")}
    output["observations"] = [{k: str(v) if isinstance(v, Q) else v for k, v in r.items()} for r in rows]
    output["analysis_cpu_seconds"] = time.process_time() - started
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("refuse to overwrite analysis output")
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
