//! Experimental driver of the actual native tuner adapter and UCI engine.
//! No production defaults, source or checkpoint are written. This is not a tuner.
#[allow(dead_code)]
#[path = "../src/bin/hyper_spsa/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/bin/hyper_spsa/evaluator.rs"]
mod evaluator;

use asura::{parameters::is_spsa_active, tuning::Point};
use config::{Block, Restore, Snapshot};
use evaluator::{Evaluator, Input, Panel};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

const PROFILES: [&str; 5] = ["base", "ll", "lh", "hl", "hh"];

fn point(block: &Block, profile: &str) -> Result<Point, String> {
    let mut point = block.values()?;
    let signs = match profile {
        "base" | "frozen" => return Ok(point),
        "ll" => [-1, -1],
        "lh" => [-1, 1],
        "hl" => [1, -1],
        "hh" => [1, 1],
        _ => return Err(format!("unknown profile {profile:?}")),
    };
    let axes = block.axes(0)?;
    for i in 0..2 {
        point[i] = (axes[i].value + signs[i] * axes[i].step).clamp(axes[i].min, axes[i].max);
    }
    Ok(point)
}

fn run(args: &[String]) -> Result<(), String> {
    let blocks = config::blocks();
    let block = &blocks[1];
    if args == ["describe"] {
        for profile in PROFILES {
            let values = point(block, profile)?;
            println!("PROFILE\t{profile}\tLMR_TABLE_SCALE={}\tSE_BETA_MARGIN={}", values[0], values[1]);
        }
        return Ok(());
    }
    let [mode, model, input] = args else {
        return Err("usage: describe | play MODEL PROFILE | probe MODEL ROOT_CP_TAB_FENS".into());
    };
    if !["play", "probe"].contains(&mode.as_str()) {
        return Err("mode must be play or probe".into());
    }
    asura::nnue::load_global_model(model).map_err(|e| format!("native model: {e}"))?;
    if !asura::nnue::is_nnue_active() {
        return Err("native NNUE required".into());
    }
    asura::initialize();
    if mode == "play" {
        // Freeze the point before enabling/applying tunables. The control uses
        // the same runtime path as candidates; frozen is the production control.
        let values = point(block, input)?;
        let _restore = if input == "frozen" { None } else { Some(Restore::enter(&blocks)?) };
        block.apply(&values)?;
        return asura::run(Default::default()).map_err(|e| format!("UCI: {e}"));
    }
    let start = Instant::now();
    let samples = evaluator::load_samples(&Input::Fens(PathBuf::from(input)), 100, 8, 20261104)?;
    let fens: Vec<_> = samples.iter().map(|s| s.board.to_fen()).collect();
    let _restore = Restore::enter(&blocks)?;
    let base = Snapshot::capture(&blocks)?;
    let profiles = PROFILES.map(|profile| point(block, profile)).into_iter().collect::<Result<Vec<_>, _>>()?;
    let params = asura::nnue::global_model().ok_or("native model unavailable")?;
    // Drop the evaluator before restoration changes any parameter.
    let mut evaluator = Evaluator::new(params, 4, 1, start + Duration::from_secs(180))?;
    let labels = evaluator.label(samples, 6000)?;
    let panels = labels
        .into_iter()
        .enumerate()
        .map(|(i, label)| Panel::new((i + 1) as u64, vec![label]))
        .collect::<Result<Vec<_>, _>>()?;
    println!("LABELS\tseconds={:.6}\twork={:?}", start.elapsed().as_secs_f64(), evaluator.work());
    for (profile, point) in PROFILES.into_iter().zip(profiles) {
        base.apply();
        block.apply(&point)?;
        for (i, panel) in panels.iter().enumerate() {
            let totals = evaluator.evaluate(panel, [500, 1500, 4500])?;
            println!("SCORE\t{profile}\t{}\t{}\t{:?}", fens[i], totals.utility()?, totals.sums);
        }
    }
    println!("WORK\tseconds={:.6}\twork={:?}", start.elapsed().as_secs_f64(), evaluator.work());
    Ok(())
}

fn main() -> std::process::ExitCode {
    let blocks = config::blocks();
    let before = Snapshot::capture(&blocks);
    let active = is_spsa_active();
    let result = run(&std::env::args().skip(1).collect::<Vec<_>>());
    let restored = before.is_ok() && before == Snapshot::capture(&blocks) && active == is_spsa_active();
    match result {
        Ok(()) if restored => std::process::ExitCode::SUCCESS,
        Ok(()) => {
            eprintln!("ERROR: parameters not restored");
            std::process::ExitCode::from(2)
        }
        Err(error) => {
            eprintln!("ERROR: {error}; restored={restored}");
            std::process::ExitCode::from(2)
        }
    }
}
