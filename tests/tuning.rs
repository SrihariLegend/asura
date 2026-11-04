use asura::tuning::{Axis, PanelMeasurement, probe_block, select_on_panel};

#[test]
fn unrelated_interaction_is_not_attributed_to_the_first_pair() {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let report = probe_block(&axes, 0, 0, |x| Ok(7 * x[3] * x[7])).unwrap();
    assert_eq!(report.pairs[0].contrasts.interaction4, 0);
}

#[test]
fn exact_zero_signal_does_not_force_a_passenger_coordinate_into_the_first_trial() {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let report = probe_block(&axes, 0, 0, |x| Ok(x[0])).unwrap();
    assert_eq!(report.trials[0], [1, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn validation_rejects_the_harmful_combination_of_individually_good_pair_moves() {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let objective = |x: &[i64; 8]| x[0] + x[2] - 10 * x[0] * x[2];
    let report = probe_block(&axes, 0, 0, |x| Ok(objective(x))).unwrap();
    assert_eq!(objective(&report.trials[0]), -8);
    let measure = |utility| PanelMeasurement { panel_id: 1, cases: 1, utility, protected: vec![] };
    let measurements: Vec<_> = report.trials.iter().map(|p| measure(i128::from(objective(p)))).collect();
    let selected = select_on_panel(&measure(0), &measurements, &measure(0), &[]).unwrap().unwrap();
    assert_eq!(objective(&report.trials[selected]), 1);
}

#[test]
fn protected_limits_stay_anchored_instead_of_drifting_after_each_update() {
    let measurement =
        |utility, protected| PanelMeasurement { panel_id: 1, cases: 1, utility, protected: vec![protected] };
    let anchor = measurement(0, 100);
    let incumbent = measurement(1, 96);
    let candidate = measurement(2, 92);
    assert_eq!(select_on_panel(&incumbent, &[candidate], &anchor, &[5]).unwrap(), None);
}

#[test]
fn a_bad_pair_corner_does_not_hide_a_useful_single_axis_direction() {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let report = probe_block(&axes, 0, 0, |x| Ok(x[0] + 100 * x[1] - 1000 * x[0] * x[0])).unwrap();
    // Every sampled (x0,x1) corner is worse than zero, but (0,1) scores +100.
    assert!(report.trials.contains(&[0, 1, 0, 0, 0, 0, 0, 0]));
}

#[test]
fn noiseless_small_gradient_is_retained_without_a_fictitious_standard_error() {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let report = probe_block(&axes, 0, 0, |x| Ok(x[0] + 100 * x[1])).unwrap();
    assert_eq!(report.pairs[0].contrasts.x4, 4);
    assert!(report.trials.contains(&[1, 1, 0, 0, 0, 0, 0, 0]));
}

#[test]
fn quadratic_contrasts_include_steps_but_do_not_alias_diagonal_curvature() {
    let mut axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    axes[0].step = 2;
    axes[1].step = 3;
    let report = probe_block(&axes, 0, 13, |x| {
        Ok(13 + 2 * x[0] - 3 * x[1] + 5 * x[0] * x[0] - 7 * x[1] * x[1] + 11 * x[0] * x[1] + 19 * x[3] * x[7])
    })
    .unwrap();
    assert_eq!(report.pairs[0].contrasts, asura::tuning::Contrasts { mean4: -120, x4: 16, y4: -36, interaction4: 264 });
}

#[test]
fn all_pairs_are_covered_once_in_seven_bounded_query_rounds() {
    let axes = [Axis { value: 0, min: -10, max: 10, step: 1 }; 8];
    let mut seen = std::collections::BTreeSet::new();
    for round in 0..7 {
        let mut calls = 0;
        let report = probe_block(&axes, round, 0, |x| {
            calls += 1;
            Ok(x.iter().sum())
        })
        .unwrap();
        assert_eq!(report.calls, 16);
        assert_eq!(calls, 16); // The public expensive-query budget is a contract.
        let mut coordinates = std::collections::BTreeSet::new();
        for pair in report.pairs {
            let (a, b) = pair.indices;
            assert!(coordinates.insert(a) && coordinates.insert(b));
            assert!(seen.insert((a.min(b), a.max(b))));
        }
        assert_eq!(coordinates.len(), 8);
    }
    assert_eq!(seen.len(), 28);
    assert_eq!(asura::tuning::pairing(0), asura::tuning::pairing(7));
    assert_eq!(asura::tuning::pairing(usize::MAX), asura::tuning::pairing(usize::MAX % 7));
}

#[test]
fn fixed_axes_reuse_observations_and_clipped_faces_stay_in_bounds() {
    let mut axes = [Axis { value: 0, min: 0, max: 0, step: 1 }; 8];
    axes[0] = Axis { value: 1, min: 0, max: 1, step: 1 };
    axes[1] = Axis { value: 0, min: 0, max: 1, step: 1 };
    let report = probe_block(&axes, 0, 100, |x| Ok(100 * x[0] + x[1])).unwrap();
    assert_eq!(report.calls, 3);
    // Outward, large gradient on axis0 must not suppress movable axis1.
    assert_eq!(report.trials[0], [1, 1, 0, 0, 0, 0, 0, 0]);
    for point in &report.trials {
        assert!(point.iter().zip(axes).all(|(&x, a)| a.min <= x && x <= a.max));
    }
    let axes = [Axis { value: 0, min: 0, max: 0, step: 1 }; 8];
    let report = probe_block(&axes, 0, 7, |_| panic!("fixed block must not spend a query")).unwrap();
    assert_eq!(report.calls, 0);
    assert!(report.trials.is_empty());
}

#[test]
fn score_and_coordinate_extremes_do_not_wrap() {
    let contrast = asura::tuning::Contrasts::from_corners([i64::MIN, i64::MAX, i64::MAX, i64::MIN]);
    assert_eq!(contrast.interaction4, -36_893_488_147_419_103_230);
    assert_eq!(contrast.mean4, -2);
    let mut axes = [Axis { value: 0, min: 0, max: 0, step: 1 }; 8];
    axes[0] = Axis { value: 0, min: i64::MIN, max: i64::MAX, step: i64::MAX };
    let report = probe_block(&axes, 0, 0, |x| Ok(x[0])).unwrap();
    assert_eq!(report.trials[0][0], i64::MAX);
    assert_eq!(report.trials[1][0], i64::MAX / 2);
    assert_eq!(report.calls, 2);
}

#[test]
fn invalid_axes_and_failed_objectives_never_return_a_partial_proposal() {
    let mut axes = [Axis { value: 0, min: -1, max: 1, step: 1 }; 8];
    axes[4].step = 0;
    assert!(probe_block(&axes, 0, 0, |_| panic!("validate before querying")).is_err());
    axes[4].step = 1;
    let original = axes;
    let mut calls = 0;
    let error = probe_block(&axes, 0, 0, |_| {
        calls += 1;
        if calls == 3 { Err("objective failed".into()) } else { Ok(1) }
    })
    .unwrap_err();
    assert_eq!(error, "objective failed");
    assert_eq!(calls, 3);
    assert_eq!(axes, original);
}

#[test]
fn panel_guard_rejects_incomplete_evidence_and_keeps_stable_ties() {
    let base = PanelMeasurement { panel_id: 11, cases: 4, utility: 10, protected: vec![100] };
    let tie = base.clone();
    assert_eq!(select_on_panel(&base, &[tie], &base, &[0]).unwrap(), None);
    let mut candidate = base.clone();
    candidate.utility = 11;
    assert_eq!(select_on_panel(&base, &[candidate.clone(), candidate.clone()], &base, &[0]).unwrap(), Some(0));
    candidate.panel_id = 12;
    assert!(select_on_panel(&base, &[candidate.clone()], &base, &[0]).is_err());
    candidate.panel_id = 11;
    candidate.cases = 3;
    assert!(select_on_panel(&base, &[candidate.clone()], &base, &[0]).is_err());
    candidate.cases = 4;
    candidate.protected.clear();
    assert!(select_on_panel(&base, &[candidate], &base, &[0]).is_err());
    assert!(select_on_panel(&base, &[], &base, &[-1]).is_err());
    let empty = PanelMeasurement { cases: 0, ..base.clone() };
    assert!(select_on_panel(&empty, &[], &empty, &[0]).is_err());
}

#[test]
fn exact_panel_comparisons_do_not_overflow_at_i128_limits() {
    let base = PanelMeasurement {
        panel_id: 1,
        cases: 1,
        utility: i128::MIN,
        protected: vec![i128::MAX],
    };
    let mut candidate = PanelMeasurement { utility: i128::MAX, ..base.clone() };
    assert_eq!(select_on_panel(&base, &[candidate.clone()], &base, &[0]).unwrap(), Some(0));
    candidate.protected[0] = i128::MIN;
    assert_eq!(select_on_panel(&base, &[candidate], &base, &[i128::MAX]).unwrap(), None);
}

#[test]
fn predeclared_global_trials_can_escape_a_three_coordinate_barrier() {
    let axes = [Axis { value: 0, min: -4, max: 4, step: 1 }; 8];
    let objective = |x: &[i64; 8]| {
        let positive = x[..3].iter().filter(|&&v| v > 0).count();
        let cost: i64 = x.iter().enumerate().map(|(i, &v)| (v - if i < 3 { 2 } else { 0 }).pow(2)).sum();
        -cost - if positive == 1 || positive == 2 { 100 } else { 0 }
    };
    let report = probe_block(&axes, 0, -12, |x| Ok(objective(x))).unwrap();
    assert!(report.trials.iter().any(|p| objective(p) > -12));
    assert_eq!(report.calls, 16); // Exploration trials are unscored until validation.
}
