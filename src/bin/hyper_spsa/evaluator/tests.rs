use super::*;
use std::time::Duration;

fn same_labels(a: &[LabeledPosition], b: &[LabeledPosition]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.board == b.board
                && a.root_cp == b.root_cp
                && a.best_obj_q_cp == b.best_obj_q_cp
                && a.moves.len() == b.moves.len()
                && a.moves.iter().zip(&b.moves).all(|(a, b)| {
                    a.mv == b.mv
                        && a.obj_q_cp == b.obj_q_cp
                        && a.min_style_q_cp == b.min_style_q_cp
                        && a.draw_permille == b.draw_permille
                        && a.def_difficulty_cp == b.def_difficulty_cp
                        && a.terminal == b.terminal
                })
        })
}

/// One test owns the process-global model. Even the injected channel/panic
/// failures have a real native NNUE peer whose completion must be drained.
/// This tests unwind behavior; a panic=abort executable cannot recover a panic.
#[test]
fn actual_nnue_workers_are_repeatable_and_drain_failed_waves() {
    asura::initialize();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nnue::load_global_model(root.join("asura_atn1.nnue")).expect("bundled native NNUE fixture is required");
    let params = nnue::global_model().expect("native NNUE must be active");
    let deadline = Instant::now() + Duration::from_secs(30);
    let samples = load_samples(&Input::Fens(root.join("tests/fixtures/hyper_spsa.fens")), 20, 3, 123).unwrap();
    let mut one = Evaluator::new(params.clone(), 1, 1, deadline).unwrap();
    let mut three = Evaluator::new(params.clone(), 3, 1, deadline).unwrap();
    let labels1 = one.label(samples.clone(), 64).unwrap();
    let mut labels3 = three.label(samples.clone(), 64).unwrap();
    assert!(same_labels(&labels1, &labels3));
    let panel = Panel::new(1, labels1).unwrap();
    let budgets = [48, 96, 192];
    let expected = one.evaluate(&panel, budgets).unwrap();
    assert_eq!(three.evaluate(&panel, budgets).unwrap().sums, expected.sums);
    assert_eq!(three.evaluate(&panel, budgets).unwrap().sums, expected.sums);
    assert!(one.work().nodes > 0 && three.work().nodes > 0);

    // The search still runs with real NNUE. Removing its answer's label must
    // return an error, not substitute another move or partial successful totals.
    labels3[0].moves.clear();
    let damaged = Panel::new(2, labels3).unwrap();
    let error = three.evaluate(&damaged, budgets).err().expect("missing labels must fail");
    assert!(error.contains("missing from frozen labels"), "{error}");
    assert_eq!(three.evaluate(&panel, budgets).unwrap().sums, expected.sums);
    drop(one);
    drop(three);

    // Inject transport failure in one slot. The other slot remains the actual
    // production worker and must complete all three native budget searches.
    let mut disconnected = Evaluator::new(params.clone(), 2, 1, deadline).unwrap();
    disconnected.workers[0].jobs.take();
    disconnected.workers[0].handle.take().unwrap().join().unwrap();
    let (sender, inbox) = mpsc::sync_channel::<Job>(1);
    let handle = thread::spawn(move || {
        drop(inbox.recv().unwrap());
    });
    disconnected.workers[0] = Worker { jobs: Some(sender), handle: Some(handle) };
    let error = disconnected.evaluate(&panel, budgets).err().expect("lost completion must fail");
    assert!(error.contains("completion disconnected"), "{error}");
    assert_eq!(disconnected.work().searches, 3);
    assert!(disconnected.work().nodes > 0);
    drop(disconnected);

    // An inconsistent private job count causes an actual production worker to
    // unwind before its lookup. Its live peer still labels a real NNUE position.
    let mut panicking = Evaluator::new(params.clone(), 2, 1, deadline).unwrap();
    let error = panicking
        .run_batch(Batch::Label(Arc::new(vec![samples[0].clone()]), 48), 2)
        .err()
        .expect("worker panic must be reported");
    assert!(error.contains("native worker panic"), "{error}");
    assert!(panicking.work().searches > 0 && panicking.work().nodes > 0);
    drop(panicking);

    let mut terminal = Evaluator::new(params, 1, 1, deadline).unwrap();
    let sample = Sample {
        board: Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap(),
        root_cp: 1000,
    };
    let labels = terminal.label(vec![sample], 48).unwrap();
    assert!(labels[0].moves.iter().any(|m| m.terminal && m.obj_q_cp == Score::MATE - 1));
}
