macro_rules! define {
    {$($type:ty, $name:ident: $value:expr; )*} => {
        $(pub const $name: $type = $value;)*
    };
}

define! {
    // Centipawn-scale (100 = 1 pawn) aspiration constants.
    u8,  ASPIRATION_MIN_DEPTH: 2;
    i32, ASPIRATION_BASE_DELTA: 12;
    i32, ASPIRATION_STABILITY_MAX: 5;
    i32, ASPIRATION_VOLATILITY_NUM: 25;
    i32, ASPIRATION_QUADRATIC_DIV: 10_000;
    i32, ASPIRATION_FAIL_LOW_WIDEN_NUM: 33;
    i32, ASPIRATION_FAIL_HIGH_WIDEN_NUM: 47;
    i32, ASPIRATION_RATIO_DEN: 100;
    i32, ASPIRATION_INFINITE_BOUND: 1_000;
    i32, EVAL_STABILITY_THRESHOLD: 10;
    i32, SCOUT_HISTORY_BONUS: 50;

    // LMR and node-bias constants in 1_000 = 1 ply fixed-point units.
    i32, LMR_SCALE: 1_000;
    i32, LMR_TABLE_SCALE: 393;
    i32, LMR_HIST_DIV: 10;
    i32, LMR_GOOD_NOISY_BASE: 184;
    i32, LMR_BAD_NOISY_BASE: 2_359;
    i32, LMR_PRIORITY_QUIET_BONUS: 788;
    i32, LMR_CHECK_BONUS: 754;
    i32, LMR_THREAT_BONUS: 565;
    i32, LMR_EXACT_BOUND_PENALTY: 1_031;
    i32, LMR_CHILD_CUTOFF_PENALTY: 1_017;

    i32, LMR_PV_BIAS: -511;
    i32, LMR_CUT_BIAS: 2_184;
    i32, LMR_TT_PV_BONUS: 1_157;
    i32, LMR_IN_CHECK_BONUS: 923;
    i32, LMR_NON_IMPROVING_PENALTY: 701;
    i32, LMR_TT_CAPTURE_PENALTY: 564;
    i32, LMR_PARENT_RED_WEIGHT: 342;
    i32, LMR_CRIT_CHAIN_BONUS: 164;
    i32, LMR_LATERALITY_PENALTY: 37;
    i32, SCOUT_RETRY_BASE_MARGIN: 52;
    i32, SCOUT_RETRY_DEPTH_SCALE: 10;

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
    i32, RFP_DEPTH_MARGIN: 59;
    i32, RFP_IMPROVING_BONUS: 53;
    i32, RFP_NO_THREATS_BONUS: 35;
    i32, RFP_STAT_DIVISOR: 200;
    i32, RFP_MIN_MARGIN: 30;

    // ProbCut constants on the 100 = 1 pawn scale.
    u8,  PROBCUT_MIN_DEPTH: 5;
    i32, PROBCUT_BETA_MARGIN: 114;
    i32, PROBCUT_IMPROVING_BONUS: 40;

    // Main-loop SEE pruning constants on the 100 = 1 pawn / 10_000 = MAX_HISTORY scale.
    i32, SEE_QUIET_HIST_DIV: 5_000;
    i32, SEE_QUIET_QUAD_COEFF: -15;
    i32, SEE_NOISY_DEPTH_COEFF: -102;
    i32, SEE_NOISY_HIST_DIV: 100;

    // Main-loop Futility Pruning constants on the 100 = 1 pawn / 10_000 = MAX_HISTORY scale.
    i32, FP_MAX_DEPTH: 1;
    i32, FP_BASE_MARGIN: 133;
    i32, FP_DEPTH_MARGIN: 88;
    i32, FP_HIST_DIV: 200;
    i32, FP_HIST_MIN: -60;
    i32, FP_HIST_MAX: 150;
    i32, FP_MIN_MARGIN: 50;

    // Quiet-History Pruning constants on the 10_000 = MAX_HISTORY scale.
    u8,  HP_MAX_DEPTH: 4;
    i32, HP_DEPTH_SCALE: -368;
    i32, HP_IMPROVING_OFFSET: -177;

    // Internal Iterative Reduction constants.
    u8,  IIR_MIN_DEPTH: 3;
    i32, IIR_TT_DEPTH_GAP: 4;

    // Singular Extension and Low-Depth Singular Extension (LDSE) constants on the 100 = 1 pawn scale.
    u8,  SE_MIN_DEPTH: 6;
    i32, SE_BETA_MARGIN: 24;
    i32, SE_DEPTH_MARGIN: 3;
    i32, SE_DOUBLE_GAP: 26;
    i32, SE_TRIPLE_GAP: 119;
    i32, SE_HINDSIGHT_MIN_REDUCTION: 3_000;
    u8,  LDSE_MIN_DEPTH: 5;
    u8,  LDSE_MAX_DEPTH: 7;
    i32, LDSE_MARGIN: 25;

    // Razoring constants on the 100 = 1 pawn scale.
    u8,  RAZOR_MAX_DEPTH: 3;
    i32, RAZOR_BASE: 227;
    i32, RAZOR_MARGIN: 232;
    i32, RAZOR_MOVE_MARGIN: 175;

    // Checked TT-capture cutoff constants on the 100 = 1 pawn scale.
    u8,  CHECKED_TT_CAPTURE_MIN_DEPTH: 5;
    i32, CHECKED_TT_CAPTURE_MARGIN: 200;
    i32, CHECKED_TT_CAPTURE_DEPTH_GAP: 4;

    // Post-move late-quiet closure constants on the 100 = 1 pawn scale.
    i32, LATE_QUIET_MAX_DEPTH: 7;
    i32, LATE_QUIET_BASE_MARGIN: 27;
    i32, LATE_QUIET_DEPTH_MARGIN: 77;

    // Quiescence quiet-history pruning threshold on the 10_000 = MAX_HISTORY scale.
    i32, QS_QUIET_HIST_THRESHOLD: -350;

    // Quiescence capture-futility margins on the 100 = 1 pawn scale.
    i32, QS_FUTILITY_MARGIN: 217;
    i32, QS_SEE_FUTILITY_MARGIN: 300;

    // Quiescence late-capture limit (maximum legal move index for unrelated
    // non-promotion, non-checking captures when not in check).
    usize, QS_LATE_CAPTURE_LIMIT: 1;

    // Quiescence history-backed early fail-high margin on the 100 = 1 pawn scale.
    i32, QS_EARLY_CUTOFF_MARGIN: 228;

    // Quiescence SEE pruning constants on the 100 = 1 pawn and 10_000 = MAX_HISTORY scales.
    i32, QS_SEE_QUIET_THRESHOLD: 0;
    i32, QS_SEE_NOISY_FLOOR: -10;
    i32, QS_SEE_BASE: 72;
    i32, QS_SEE_EVAL_DIV: 8;
    i32, QS_SEE_HIST_DIV: 50;

    // Hindsight depth adjustment constants (1_000 = 1 ply, 100 = 1 pawn).
    i32, HINDSIGHT_EXT_MIN_REDUCTION: 2_000;
    i32, HINDSIGHT_REDUCTION_MARGIN: 94;

    // Correction history constants on the 100 = 1 pawn scale.
    i32, CORRHIST_MAX: 1_024;
    i32, CORRHIST_SCALE_DIV: 32;
    i32, CORRHIST_BONUS_DIV: 8;
    i32, CORRHIST_MAX_BONUS: 256;

    // Post-alpha-raise effective depth reduction minimum depth (keeps child_depth >= 1).
    u8,  ALPHA_REDUCE_MIN_DEPTH: 8;

    // Clear-cutoff history depth bonus margin on the 100 = 1 pawn scale.
    i32, CLEAR_CUTOFF_MARGIN: 50;

    // Move-picker and history update constants on the 10_000 = MAX_HISTORY scale.
    i32, DEFER_QUIET_THRESHOLD: -1_000;
    i32, GOOD_NOISY_SEE_HIST_DIV: 100;
    i32, QUIET_HIST_WEIGHT: 1_500;
    i32, PAWN_HIST_WEIGHT: 1_000;
    i32, CONT_HIST_1_WEIGHT: 1_500;
    i32, CONT_HIST_2_WEIGHT: 1_000;
    i32, CONT_HIST_4_WEIGHT: 1_000;
    i32, CONT_HIST_6_WEIGHT: 1_000;
    i32, QUIET_CHECK_SCORE_BONUS: 10_000;
    i32, QUIET_THREATENED_DEST_PENALTY: 8_000;
    i32, QUIET_CREATE_THREAT_BONUS: 3_500;
    i32, QUIET_WALL_PAWN_PENALTY: 4_000;

    i32, NOISY_BONUS_SCALE: 100;
    i32, NOISY_BONUS_CAP: 900;
    i32, NOISY_BONUS_OFFSET: 40;
    i32, NOISY_BONUS_CUT_PENALTY: 80;
    i32, NOISY_MALUS_SCALE: 170;
    i32, NOISY_MALUS_CAP: 1_200;
    i32, NOISY_MALUS_OFFSET: 50;
    i32, NOISY_MALUS_COUNT_STEP: 15;

    i32, QUIET_BONUS_SCALE: 180;
    i32, QUIET_BONUS_CAP: 1_700;
    i32, QUIET_BONUS_OFFSET: 70;
    i32, QUIET_BONUS_CUT_PENALTY: 40;
    i32, QUIET_MALUS_SCALE: 170;
    i32, QUIET_MALUS_CAP: 1_100;
    i32, QUIET_MALUS_OFFSET: 45;
    i32, QUIET_MALUS_COUNT_STEP: 30;

    i32, CONT_BONUS_SCALE: 100;
    i32, CONT_BONUS_CAP: 1_100;
    i32, CONT_BONUS_OFFSET: 70;
    i32, CONT_BONUS_CUT_PENALTY: 45;
    i32, CONT_MALUS_SCALE: 400;
    i32, CONT_MALUS_CAP: 950;
    i32, CONT_MALUS_OFFSET: 50;
    i32, CONT_MALUS_COUNT_STEP: 15;

    // Time-management constants (ms, moves, and 1_000 = 1.0x fixed-point scale).
    u64, DEFAULT_MOVE_OVERHEAD_MS: 15;
    u64, SINGLE_MOVE_SOFT_CAP_MS: 500;
    u64, TIME_CHECK_NODE_MASK: 1_023;
    u64, HORIZON_OPENING_MOVES: 40;
    u64, HORIZON_MIN_MOVES: 22;
    u64, INCREMENT_SHARE_PERMILLE: 667;
    u64, SOFT_CLOCK_CAP_PERMILLE: 145;
    u64, HARD_SOFT_MULT_PERMILLE: 5_163;
    u64, HARD_CLOCK_CAP_PERMILLE: 750;

    i64, TM_SCALE_UNIT: 1_000;
    i64, TM_MIN_SCALE: 400;
    i64, TM_MAX_SCALE: 3_500;
    i64, TM_NODE_BASE: 463;
    i64, TM_NODE_SLOPE: 1_394;
    i64, TM_STABILITY_BASE: 1_219;
    i64, TM_PV_STABILITY_STEP: 64;
    i64, TM_PV_STABILITY_MAX: 6;
    i64, TM_EVAL_STABILITY_STEP: 27;
    i64, TM_EVAL_STABILITY_MAX: 5;
    i64, TM_STABILITY_FLOOR: 800;
    i64, TM_TURNOVER_STEP: 148;
    i64, TM_TURNOVER_MAX: 4;
    i64, TM_TREND_SLOPE_PERMILLE: 4_473;
    i64, TM_TREND_MIN_DIFF: -20;
    i64, TM_TREND_MAX_DIFF: 200;
}
