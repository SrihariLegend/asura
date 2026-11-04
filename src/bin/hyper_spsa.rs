//! Hyper-SPSA is a candidate generator, not a playing-strength certificate.

#[path = "hyper_spsa/config.rs"]
mod config;
#[path = "hyper_spsa/evaluator.rs"]
mod evaluator;
#[path = "hyper_spsa/legacy.rs"]
mod legacy;
#[path = "hyper_spsa/math_bench.rs"]
mod math_bench;

use asura::{
    parameters::is_spsa_active,
    tuning::{Axis, PanelMeasurement, Point, probe_block, select_on_panel},
};
use config::{Block, Restore, Snapshot};
use evaluator::{Evaluator, Input, Panel, Totals, Work};
use std::{
    collections::HashSet,
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

const HELP: &str = "Hyper-SPSA: bounded proposals and finite-panel checks (not Elo certification)
Usage: hyper_spsa <check|math-bench|verify|bench|tune> [options]
  check                         Exact failure witnesses and restoration check; no model needed
  math-bench --queries N         Equal-query synthetic comparison, including ordinary SPSA
  verify                        Actual NNUE/search reproducibility and probe cleanup
  bench                         Alternating old/new probe and guarded-cycle timings; no adoption
  tune                          Generate candidates; reserve final test panel until the end
Native options (explicit model and exactly one input required):
  --model FILE                  Frozen ATN-1 checkpoint; no material fallback
  --data FILE | --fens FILE      Packed records or root_cp<TAB>full FEN lines
  --train N --validation N --test N  Disjoint position counts (default 12 each)
  --scan N                      Maximum records/physical lines to inspect (default 40000)
  --nodes N,N,N                 Nondecreasing node budgets (default 2000,8000,32000)
  --label-nodes N               Frozen-label child search budget (default twice largest rung)
  --threads N --hash N           Workers and MiB per worker (default <=12, 1)
  --blocks 1,2,...               Parameter blocks, default all six
  --passes N                    Tuning passes, default 2; pair schedule cycles every 7
  --repeats N                   Benchmark repetitions, default 6
  --seed N                      Deterministic panel/benchmark seed (default 1)
  --max-seconds N                Root-boundary deadline, default 300; NOT a hard process timeout
No source, dataset, or checkpoint is overwritten. Output candidates still require held-out matches.
";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Help,
    Check,
    MathBench,
    Verify,
    Bench,
    Tune,
}

struct Options {
    mode: Mode,
    model: Option<PathBuf>,
    input: Option<Input>,
    scan: usize,
    seed: u64,
    train: usize,
    validation: usize,
    test: usize,
    budgets: [u64; 3],
    label_nodes: u64,
    threads: usize,
    hash: usize,
    passes: usize,
    repeats: usize,
    blocks: Vec<usize>,
    seconds: u64,
    queries: usize,
}

fn parse() -> Result<Options, String> {
    let args = std::env::args_os()
        .skip(1)
        .map(|s| s.into_string().map_err(|_| "arguments must be UTF-8".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let mode = match args.first().map(String::as_str) {
        None | Some("--help" | "-h") => Mode::Help,
        Some("check") => Mode::Check,
        Some("math-bench") => Mode::MathBench,
        Some("verify") => Mode::Verify,
        Some("bench") => Mode::Bench,
        Some("tune") => Mode::Tune,
        Some(name) => return Err(format!("unknown command {name:?}; use --help")),
    };
    let mut o = Options {
        mode,
        model: None,
        input: None,
        scan: 40_000,
        seed: 1,
        train: 12,
        validation: 12,
        test: 12,
        budgets: [2_000, 8_000, 32_000],
        label_nodes: 0,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(12)),
        hash: 1,
        passes: 2,
        repeats: 6,
        blocks: (0..6).collect(),
        seconds: 300,
        queries: 512,
    };
    if mode == Mode::Help {
        return Ok(o);
    }
    let mut seen = HashSet::new();
    let mut args = args.into_iter().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            o.mode = Mode::Help;
            return Ok(o);
        }
        if !seen.insert(flag.clone()) {
            return Err(format!("duplicate option {flag}"));
        }
        if mode == Mode::Check
            || (mode == Mode::MathBench && !["--seed", "--queries", "--repeats"].contains(&flag.as_str()))
        {
            return Err(format!("unknown option {flag} for this command"));
        }
        let value = args
            .next()
            .filter(|s| !s.is_empty() && !s.starts_with("--"))
            .ok_or_else(|| format!("missing value for {flag}"))?;
        let number = || value.parse::<u64>().map_err(|_| format!("invalid unsigned integer for {flag}"));
        let size = || usize::try_from(number()?).map_err(|_| format!("value too large for {flag}"));
        match flag.as_str() {
            "--model" => o.model = Some(value.into()),
            "--data" | "--fens" => {
                if o.input.is_some() {
                    return Err("provide exactly one input".into());
                }
                o.input = Some(if flag == "--data" { Input::Packed(value.into()) } else { Input::Fens(value.into()) });
            }
            "--scan" => o.scan = size()?,
            "--seed" => o.seed = number()?,
            "--train" => o.train = size()?,
            "--validation" => o.validation = size()?,
            "--test" => o.test = size()?,
            "--threads" => o.threads = size()?,
            "--hash" => o.hash = size()?,
            "--passes" => o.passes = size()?,
            "--repeats" => o.repeats = size()?,
            "--queries" if mode == Mode::MathBench => o.queries = size()?,
            "--max-seconds" => o.seconds = number()?,
            "--label-nodes" => o.label_nodes = number()?,
            "--nodes" => {
                let values = value
                    .split(',')
                    .map(|s| s.parse::<u64>().map_err(|_| "invalid node list".to_owned()))
                    .collect::<Result<Vec<_>, _>>()?;
                o.budgets = values.try_into().map_err(|_| "provide three node budgets")?;
            }
            "--blocks" => {
                let values = value
                    .split(',')
                    .map(|s| s.parse::<usize>().map_err(|_| "invalid block list".to_owned()))
                    .collect::<Result<Vec<_>, _>>()?;
                let unique: HashSet<_> = values.iter().copied().collect();
                if values.is_empty() || unique.len() != values.len() || values.iter().any(|&i| !(1..=6).contains(&i)) {
                    return Err("blocks must be distinct numbers in 1..=6".into());
                }
                o.blocks = values.into_iter().map(|i| i - 1).collect();
            }
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if mode == Mode::Check {
        return Ok(o);
    }
    if !(1..=64).contains(&o.repeats) || !(64..=100_000).contains(&o.queries) {
        return Err("require repeats 1..=64 and synthetic queries 64..=100000".into());
    }
    if mode == Mode::MathBench {
        return Ok(o);
    }
    let count = o.train.checked_add(o.validation).and_then(|n| n.checked_add(o.test)).ok_or("panel size overflow")?;
    if o.model.is_none() || o.input.is_none() {
        return Err("native commands require --model and --data or --fens".into());
    }
    if [o.train, o.validation, o.test].contains(&0) || count > 4096 || o.scan < count || o.scan > 1_000_000 {
        return Err("require three nonempty panels, total <=4096, total <=scan <=1000000".into());
    }
    if o.budgets[0] == 0 || o.budgets[2] > 50_000_000 || o.budgets.windows(2).any(|w| w[0] > w[1]) {
        return Err("node budgets must be positive, nondecreasing and <=50000000".into());
    }
    if o.label_nodes == 0 {
        if seen.contains("--label-nodes") {
            return Err("label node budget must be positive".into());
        }
        o.label_nodes = 2 * o.budgets[2];
    }
    if o.label_nodes > 100_000_000 || (mode == Mode::Tune && o.label_nodes < o.budgets[2]) {
        return Err("label budget must be <=100000000 and, for tuning, >=largest probe budget".into());
    }
    if !(1..=64).contains(&o.threads)
        || o.hash == 0
        || o.threads.checked_mul(o.hash).is_none_or(|n| n > 4096)
        || !(1..=64).contains(&o.passes)
        || !(1..=86_400).contains(&o.seconds)
    {
        return Err("invalid worker/hash/pass/deadline resource limits".into());
    }
    Ok(o)
}

fn measurement(panel: &Panel, totals: &Totals) -> Result<PanelMeasurement, String> {
    if totals.cases != panel.len() {
        return Err("panel totals have wrong case count".into());
    }
    let weighted = |metric: usize| -> Result<i128, String> {
        [15i128, 30, 55].into_iter().enumerate().try_fold(0i128, |sum, (r, weight)| {
            sum.checked_add(totals.sums[r][metric].checked_mul(weight).ok_or("metric product overflow")?)
                .ok_or_else(|| "metric sum overflow".into())
        })
    };
    let mut protected: Vec<_> = totals.sums.iter().map(|r| r[5]).collect();
    for metric in [0, 1, 3, 4] {
        protected.push(weighted(metric)?);
    }
    Ok(PanelMeasurement {
        panel_id: panel.id,
        cases: panel.len(),
        utility: weighted(5)?,
        protected,
    })
}

fn tolerances(panel: &Panel) -> Vec<i128> {
    // No observed utility loss at ANY rung. Existing proxy tolerances are in
    // original units, then scaled to integer totals. They are not confidence intervals.
    let unit = panel.len() as i128 * 100 * 1000;
    vec![0, 0, 0, 5 * unit, 5 * unit, 3 * unit, 3 * unit]
}

fn evaluate_point(
    e: &mut Evaluator, panel: &Panel, budgets: [u64; 3], block: &Block, point: &Point,
) -> Result<Totals, String> {
    block.apply(point)?;
    e.evaluate(panel, budgets)
}

fn measured_trials(
    e: &mut Evaluator, panel: &Panel, budgets: [u64; 3], block: &Block, trials: &[Point],
) -> Result<Vec<PanelMeasurement>, String> {
    // The complete trial list already exists; validation never creates another trial.
    trials.iter().map(|point| measurement(panel, &evaluate_point(e, panel, budgets, block, point)?)).collect()
}

fn delta(after: Work, before: Work) -> Work {
    Work {
        searches: after.searches - before.searches,
        nodes: after.nodes - before.nodes,
        batches: after.batches - before.batches,
    }
}

fn native_bench(
    o: &Options, blocks: &[Block], e: &mut Evaluator, train: &Panel, validation: &Panel,
) -> Result<(), String> {
    for &index in &o.blocks {
        let block = &blocks[index];
        let axes = block.axes(0)?;
        let base = block.values()?;
        let baseline = e.evaluate(train, o.budgets)?.utility()?;
        let validation_base = measurement(validation, &e.evaluate(validation, o.budgets)?)?;
        let limits = tolerances(validation);
        for repeat in 0..o.repeats {
            for method in if repeat % 2 == 0 { ["legacy", "paired"] } else { ["paired", "legacy"] } {
                block.apply(&base)?;
                let before = e.work();
                let start = Instant::now();
                let (calls, trials) = if method == "legacy" {
                    let report = legacy::probe(&axes, baseline as f64 / 1000.0, |point| {
                        Ok(evaluate_point(e, train, o.budgets, block, point)?.utility()? as f64 / 1000.0)
                    })?;
                    (report.calls, report.trials)
                } else {
                    let report = probe_block(&axes, 0, baseline, |point| {
                        evaluate_point(e, train, o.budgets, block, point)?.utility()
                    })?;
                    (report.calls, report.trials)
                };
                let probe_ms = start.elapsed().as_secs_f64() * 1000.0;
                let probe_work = delta(e.work(), before);
                if probe_work.batches != calls as u64 {
                    return Err("probe accounting mismatch".into());
                }
                block.apply(&base)?;
                let observed = measured_trials(e, validation, o.budgets, block, &trials)?;
                let selected = select_on_panel(&validation_base, &observed, &validation_base, &limits)?;
                let cycle_ms = start.elapsed().as_secs_f64() * 1000.0;
                let cycle_work = delta(e.work(), before);
                let gain = selected.map_or(0, |i| observed[i].utility - validation_base.utility);
                block.apply(&base)?;
                println!(
                    "BENCH\tblock={}\trepeat={repeat}\tdesign={method}\tprobe_calls={calls}\tprobe_ms={probe_ms:.3}\tprobe_nodes={}\tcycle_calls={}\tcycle_ms={cycle_ms:.3}\tcycle_nodes={}\ttrials={}\tpanel_gain={gain}",
                    index + 1,
                    probe_work.nodes,
                    cycle_work.batches,
                    cycle_work.nodes,
                    trials.len()
                );
            }
        }
    }
    println!("RESULT\tmode=bench\tadopted=false\tstrength_validated=false");
    Ok(())
}

fn native_verify(o: &Options, blocks: &[Block], e: &mut Evaluator, train: &Panel) -> Result<(), String> {
    let block = &blocks[o.blocks[0]];
    let base = block.values()?;
    let initial = e.evaluate(train, o.budgets)?;
    let repeated = e.evaluate(train, o.budgets)?;
    if initial.sums != repeated.sums {
        return Err("native objective is not reproducible".into());
    }
    let report = probe_block(&block.axes(0)?, 0, initial.utility()?, |point| {
        evaluate_point(e, train, o.budgets, block, point)?.utility()
    })?;
    block.apply(&base)?;
    let restored = e.evaluate(train, o.budgets)?;
    if initial.sums != restored.sums {
        return Err("native objective changed after probe restoration".into());
    }
    println!(
        "VERIFY\tnative_nnue=true\trepeat_equal=true\tprobe_restore_equal=true\tprobe_calls={}\tutility={}\ttotals={:?}",
        report.calls,
        initial.utility()?,
        initial.sums
    );
    Ok(())
}

fn native_tune(
    o: &Options, blocks: &[Block], e: &mut Evaluator, train: &Panel, validation: &Panel, test: &Panel,
    original: &Snapshot,
) -> Result<(), String> {
    let anchor = measurement(validation, &e.evaluate(validation, o.budgets)?)?;
    let mut current = anchor.clone();
    let mut current_parameters = Snapshot::capture(blocks)?;
    let limits = tolerances(validation);
    let mut accepted = 0;
    for pass in 0..o.passes {
        for &index in &o.blocks {
            if Snapshot::capture(blocks)? != current_parameters {
                return Err("parameters changed outside the selected update".into());
            }
            let block = &blocks[index];
            let base = block.values()?;
            let baseline = e.evaluate(train, o.budgets)?.utility()?;
            let report = probe_block(&block.axes(pass)?, pass, baseline, |point| {
                evaluate_point(e, train, o.budgets, block, point)?.utility()
            })?;
            block.apply(&base)?;
            let observed = measured_trials(e, validation, o.budgets, block, &report.trials)?;
            let selected = select_on_panel(&current, &observed, &anchor, &limits)?;
            if let Some(i) = selected {
                block.apply(&report.trials[i])?;
                current = observed[i].clone();
                current_parameters = Snapshot::capture(blocks)?;
                accepted += 1;
            } else {
                block.apply(&base)?;
            }
            println!(
                "STEP\tpass={}\tblock={}\tname={:?}\tprobe_calls={}\ttrials={}\taccepted={}\tvalidation_total={}",
                pass + 1,
                index + 1,
                block.name,
                report.calls,
                report.trials.len(),
                selected.is_some(),
                current.utility
            );
        }
    }
    let candidate = Snapshot::capture(blocks)?;
    // The test panel is first observed only after the final candidate is fixed.
    // A failed test never causes another tuning attempt in this invocation.
    original.apply();
    let test_base = measurement(test, &e.evaluate(test, o.budgets)?)?;
    candidate.apply();
    let test_candidate = measurement(test, &e.evaluate(test, o.budgets)?)?;
    let test_pass =
        select_on_panel(&test_base, std::slice::from_ref(&test_candidate), &test_base, &tolerances(test))?.is_some();
    println!(
        "RESULT\tmode=tune\taccepted_updates={accepted}\tvalidation_gain={}\ttest_gain={}\ttest_pass={test_pass}\tstrength_validated=false\tadopted=false",
        current.utility - anchor.utility,
        test_candidate.utility - test_base.utility
    );
    if test_pass {
        for &(name, value) in &candidate.0 {
            if original.0.iter().any(|&(key, old)| key == name && old != value) {
                println!("CANDIDATE_PARAM\t{name}\t{value}");
            }
        }
    } else {
        println!("CANDIDATE_WITHHELD\tfinal finite-position test did not pass; no retry on this test panel");
    }
    Ok(())
}

fn native(o: &Options, blocks: &[Block]) -> Result<(), String> {
    asura::initialize();
    let start = Instant::now();
    let deadline = start.checked_add(Duration::from_secs(o.seconds)).ok_or("deadline overflow")?;
    let samples = evaluator::load_samples(
        o.input.as_ref().ok_or("missing input")?,
        o.scan,
        o.train + o.validation + o.test,
        o.seed,
    )?;
    for (index, sample) in samples.iter().enumerate() {
        let panel = if index < o.train {
            "train"
        } else if index < o.train + o.validation {
            "validation"
        } else {
            "test"
        };
        println!("POSITION\tpanel={panel}\troot_cp={}\tfen={}", sample.root_cp, sample.board.to_fen());
    }
    let model = o.model.as_ref().ok_or("missing model")?;
    if !std::fs::metadata(model).map_err(|e| format!("model metadata: {e}"))?.is_file() {
        return Err("model must be a regular file".into());
    }
    asura::nnue::load_global_model(model).map_err(|e| format!("load native model: {e}"))?;
    let params = asura::nnue::global_model().ok_or("native model unavailable")?;
    // Drop order matters: evaluator joins before restore changes the tunables.
    let restore = Restore::enter(blocks)?;
    let mut e = Evaluator::new(params, o.threads, o.hash, deadline)?;
    println!(
        "SETUP\tnative_nnue=true\tmodel={:?}\tthreads={}\tnodes={:?}\tlabel_nodes={}\tunique_parameters={}\tposition_panels_not_independent_games=true",
        model,
        o.threads,
        o.budgets,
        o.label_nodes,
        restore.before.0.len()
    );
    let mut labeled = e.label(samples, o.label_nodes)?;
    let test = Panel::new(3, labeled.split_off(o.train + o.validation))?;
    let validation = Panel::new(2, labeled.split_off(o.train))?;
    let train = Panel::new(1, labeled)?;
    println!(
        "LABELS\tseconds={:.3}\tsearches={}\tnodes={}\tself_nnue_proxy=true",
        start.elapsed().as_secs_f64(),
        e.work().searches,
        e.work().nodes
    );
    match o.mode {
        Mode::Verify => native_verify(o, blocks, &mut e, &train)?,
        Mode::Bench => native_bench(o, blocks, &mut e, &train, &validation)?,
        Mode::Tune => native_tune(o, blocks, &mut e, &train, &validation, &test, &restore.before)?,
        _ => return Err("invalid native mode".into()),
    }
    println!(
        "WORK\tseconds={:.3}\tsearches={}\tnodes={}\tbatches={}",
        start.elapsed().as_secs_f64(),
        e.work().searches,
        e.work().nodes,
        e.work().batches
    );
    Ok(())
}

fn check(blocks: &[Block]) -> Result<(), String> {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let old = legacy::probe(&axes, 0.0, |x| Ok((7 * x[3] * x[7]) as f64))?;
    let repaired = probe_block(&axes, 0, 0, |x| Ok(7 * x[3] * x[7]))?;
    let linear = legacy::probe(&axes, 0.0, |x| Ok((x[0] + 100 * x[1]) as f64))?;
    let fixed_linear = probe_block(&axes, 0, 0, |x| Ok(x[0] + 100 * x[1]))?;
    if (old.interactions[0] - 7.0).abs() > 1e-12
        || repaired.pairs[0].contrasts.interaction4 != 0
        || linear.gradient[0] != 1.0
        || linear.pseudo_se[0] != 25.0
        || fixed_linear.pairs[0].contrasts.x4 != 4
    {
        return Err("estimator contract diagnostic failed".into());
    }
    let before = Snapshot::capture(blocks)?;
    let active = is_spsa_active();
    let deliberate_error = (|| -> Result<(), String> {
        let _restore = Restore::enter(blocks)?;
        let mut point = blocks[0].values()?;
        point[0] += 1;
        blocks[0].apply(&point)?;
        Err("intentional diagnostic failure".into())
    })();
    if deliberate_error.is_ok() || Snapshot::capture(blocks)? != before || is_spsa_active() != active {
        return Err("parameter error restoration failed".into());
    }
    println!(
        "CHECK\tlegacy_alias={}\trepaired_alias=0\tnoiseless_gradient={}\tlegacy_pseudo_se={}\trepaired_gradient=1\tlegacy_queries={}\trepaired_queries={}\terror_restore=true",
        old.interactions[0], linear.gradient[0], linear.pseudo_se[0], old.calls, repaired.calls
    );
    Ok(())
}

fn main() -> ExitCode {
    let blocks = config::blocks();
    let before = Snapshot::capture(&blocks);
    let active = is_spsa_active();
    let result = parse().and_then(|o| match o.mode {
        Mode::Help => {
            print!("{HELP}");
            Ok(())
        }
        Mode::Check => check(&blocks),
        Mode::MathBench => math_bench::run(o.seed, o.queries, o.repeats),
        _ => native(&o, &blocks),
    });
    let restored = before.is_ok() && before == Snapshot::capture(&blocks) && active == is_spsa_active();
    println!("RESTORE\tok={restored}");
    match result {
        Ok(()) if restored => ExitCode::SUCCESS,
        Ok(()) => {
            eprintln!("ERROR: parameter restoration failed");
            ExitCode::from(2)
        }
        Err(error) => {
            eprintln!("ERROR: {error}");
            ExitCode::from(2)
        }
    }
}
