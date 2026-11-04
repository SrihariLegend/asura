use std::{
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

fn command(args: &[&str], seconds: u64) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hyper_spsa"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!("tuner failed to terminate within {seconds}s: {}", String::from_utf8_lossy(&output.stderr));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn unknown_arguments_are_rejected_before_expensive_work() {
    let output = command(&["--definitely-invalid"], 2);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown"));
}

fn native_args(mode: &str, threads: usize) -> Vec<String> {
    let root = env!("CARGO_MANIFEST_DIR");
    [
        mode.to_owned(),
        "--model".into(),
        format!("{root}/asura_atn1.nnue"),
        "--fens".into(),
        format!("{root}/tests/fixtures/hyper_spsa.fens"),
        "--train".into(),
        "2".into(),
        "--validation".into(),
        "2".into(),
        "--test".into(),
        "2".into(),
        "--scan".into(),
        "20".into(),
        "--nodes".into(),
        "32,64,128".into(),
        "--threads".into(),
        threads.to_string(),
        "--blocks".into(),
        "2".into(),
        "--max-seconds".into(),
        "20".into(),
    ]
    .into()
}

fn native(args: &[String]) -> Output {
    command(&args.iter().map(String::as_str).collect::<Vec<_>>(), 30)
}

#[test]
fn an_explicit_zero_label_budget_is_not_silently_replaced_by_a_default() {
    let mut args = native_args("verify", 1);
    args.extend(["--label-nodes".into(), "0".into()]);
    let output = native(&args);
    assert!(!output.status.success(), "zero node budget unexpectedly accepted");
}

#[test]
fn no_arguments_show_help_instead_of_starting_a_tuning_run() {
    let output = command(&[], 2);
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("Usage: hyper_spsa"));
    assert!(!text.contains("POSITION\t"));
}

#[test]
fn command_exercises_the_real_estimator_witnesses_and_error_restoration() {
    let output = command(&["check"], 2);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("legacy_alias=7\trepaired_alias=0"));
    assert!(text.contains("legacy_pseudo_se=25\trepaired_gradient=1"));
    assert!(text.contains("legacy_queries=32\trepaired_queries=16\terror_restore=true"));
    assert!(text.contains("RESTORE\tok=true"));
}

#[test]
fn real_nnue_objective_is_repeatable_across_worker_counts_and_parameter_probes() {
    let mut verification = Vec::new();
    for threads in [1, 3] {
        let output = native(&native_args("verify", threads));
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let text = String::from_utf8(output.stdout).unwrap();
        let line = text.lines().find(|line| line.starts_with("VERIFY\t")).expect("missing native evidence");
        assert!(line.contains("native_nnue=true\trepeat_equal=true\tprobe_restore_equal=true"));
        assert!(text.contains("RESTORE\tok=true"));
        verification.push(line.to_owned());
    }
    assert_eq!(verification[0], verification[1]);
}

#[test]
fn missing_model_and_insufficient_distinct_positions_fail_explicitly() {
    let mut args = native_args("verify", 1);
    let model = args.iter().position(|s| s == "--model").unwrap() + 1;
    args[model] = format!("{}/no-such-model-for-tuner.nnue", env!("CARGO_MANIFEST_DIR"));
    let output = native(&args);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("model"));
    let mut args = native_args("verify", 1);
    let count = args.iter().position(|s| s == "--train").unwrap() + 1;
    args[count] = "3".into();
    let output = native(&args);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("insufficient unique nonterminal"));
}

#[test]
fn native_candidate_pipeline_never_labels_proxy_results_as_strength_validation() {
    let mut args = native_args("tune", 2);
    args.extend(["--passes".into(), "1".into()]);
    let output = native(&args);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("STEP\tpass=1\tblock=2"));
    assert!(text.contains("RESULT\tmode=tune"));
    assert!(text.contains("strength_validated=false\tadopted=false"));
    assert!(text.contains("RESTORE\tok=true"));
}

#[test]
fn malformed_or_duplicate_options_do_not_fall_through_to_defaults() {
    for extra in [vec!["--nodes", "128,64,32"], vec!["--threads", "0"], vec!["--label-nodes", "0"]] {
        let mut args = native_args("verify", 1);
        if let Some(index) = args.iter().position(|s| s == extra[0]) {
            args[index + 1] = extra[1].into();
        } else {
            args.extend(extra.into_iter().map(str::to_owned));
        }
        let output = native(&args);
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("SETUP\t"));
    }
    let mut duplicate = native_args("verify", 1);
    duplicate.extend(["--threads".into(), "2".into()]);
    assert!(!native(&duplicate).status.success());
}
