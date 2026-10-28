macro_rules! define {
    {$($type:ty, $name:ident: $value:expr; )*} => {
        $(pub const $name: $type = $value;)*
    };
}

define! {
    // Centipawn-scale (100 = 1 pawn) aspiration constants.
    u8,  ASPIRATION_MIN_DEPTH: 2;
    i32, ASPIRATION_BASE_DELTA: 10;
    i32, ASPIRATION_STABILITY_MAX: 5;
    i32, ASPIRATION_VOLATILITY_NUM: 25;
    i32, ASPIRATION_QUADRATIC_DIV: 10_000;
    i32, ASPIRATION_FAIL_LOW_WIDEN_NUM: 25;
    i32, ASPIRATION_FAIL_HIGH_WIDEN_NUM: 50;
    i32, ASPIRATION_RATIO_DEN: 100;
    i32, ASPIRATION_INFINITE_BOUND: 1_000;
    i32, EVAL_STABILITY_THRESHOLD: 10;
    i32, SCOUT_HISTORY_BONUS: 50;

    // LMR constants in 1_000 = 1 ply fixed-point units.
    i32, LMR_SCALE: 1_000;
    i32, LMR_GOOD_NOISY_BASE: 200;
    i32, LMR_BAD_NOISY_BASE: 2_500;
    i32, LMR_PRIORITY_QUIET_BONUS: 1_000;
    i32, LMR_CHECK_BONUS: 750;
    i32, LMR_THREAT_BONUS: 750;
    i32, LMR_EXACT_BOUND_PENALTY: 1_000;
    i32, LMR_CHILD_CUTOFF_PENALTY: 1_000;

    // Null Move Pruning constants on the 100 = 1 pawn scale.
    u8,  NMP_MIN_DEPTH: 3;
    u8,  NMP_SHALLOW_DEPTH: 10;
    i32, NMP_SHALLOW_MARGIN: 0;
    i32, NMP_DEEP_FLOOR: -100;
    u8,  NMP_TWO_PIECE_DEPTH: 12;
    i32, NMP_TT_SEE_MARGIN: 200;
    i32, NMP_BASE_REDUCTION: 3;
    i32, NMP_EXTRA_DIV: 100;
    i32, NMP_EXTRA_CAP: 6;
    u8,  NMP_VERIFY_MIN_DEPTH: 14;

    // Reverse Futility Pruning constants on the 100 = 1 pawn / 10_000 = MAX_HISTORY scale.
    u8,  RFP_MAX_DEPTH: 8;
    i32, RFP_DEPTH_MARGIN: 85;
    i32, RFP_IMPROVING_BONUS: 40;
    i32, RFP_NO_THREATS_BONUS: 35;
    i32, RFP_STAT_DIVISOR: 200;
    i32, RFP_MIN_MARGIN: 30;

    // ProbCut constants on the 100 = 1 pawn scale.
    u8,  PROBCUT_MIN_DEPTH: 5;
    i32, PROBCUT_BETA_MARGIN: 150;
    i32, PROBCUT_IMPROVING_BONUS: 40;

    // Main-loop SEE pruning constants on the 100 = 1 pawn / 10_000 = MAX_HISTORY scale.
    i32, SEE_QUIET_HIST_DIV: 5_000;
    i32, SEE_QUIET_QUAD_COEFF: -20;
    i32, SEE_NOISY_DEPTH_COEFF: -90;
    i32, SEE_NOISY_HIST_DIV: 100;

    // Main-loop Futility Pruning constants on the 100 = 1 pawn / 10_000 = MAX_HISTORY scale.
    i32, FP_MAX_DEPTH: 6;
    i32, FP_BASE_MARGIN: 80;
    i32, FP_DEPTH_MARGIN: 80;
    i32, FP_HIST_DIV: 200;
    i32, FP_HIST_MIN: -60;
    i32, FP_HIST_MAX: 150;
    i32, FP_MIN_MARGIN: 50;

    // Quiet-History Pruning constants on the 10_000 = MAX_HISTORY scale.
    u8,  HP_MAX_DEPTH: 4;
    i32, HP_DEPTH_SCALE: -400;
    i32, HP_IMPROVING_OFFSET: -200;

    // Internal Iterative Reduction constants.
    u8,  IIR_MIN_DEPTH: 3;
    i32, IIR_TT_DEPTH_GAP: 4;

    // Singular Extension constants on the 100 = 1 pawn scale.
    u8,  SE_MIN_DEPTH: 6;
    i32, SE_BETA_MARGIN: 30;
    i32, SE_DEPTH_MARGIN: 3;
    i32, SE_DOUBLE_GAP: 25;
    i32, SE_TRIPLE_GAP: 100;

    // Razoring constants on the 100 = 1 pawn scale.
    u8,  RAZOR_MAX_DEPTH: 3;
    i32, RAZOR_BASE: 200;
    i32, RAZOR_MARGIN: 250;
    i32, RAZOR_MOVE_MARGIN: 175;

    // Checked TT-capture cutoff constants on the 100 = 1 pawn scale.
    u8,  CHECKED_TT_CAPTURE_MIN_DEPTH: 5;
    i32, CHECKED_TT_CAPTURE_MARGIN: 200;
    i32, CHECKED_TT_CAPTURE_DEPTH_GAP: 4;

    // Post-move late-quiet closure constants on the 100 = 1 pawn scale.
    i32, LATE_QUIET_MAX_DEPTH: 7;
    i32, LATE_QUIET_BASE_MARGIN: 60;
    i32, LATE_QUIET_DEPTH_MARGIN: 75;
}
