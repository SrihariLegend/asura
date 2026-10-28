use asura::{
    board::Board,
    movepick::{MainMovePicker, MainStage},
    search::{
        lmr_table_term, search, search_reference, search_with_context, search_with_tt, search_with_zero_draws,
        search_without_checked_tt_cutoff, search_without_fp, search_without_hp, search_without_iir,
        search_without_late_quiet_closure, search_without_lmp, search_without_lmr, search_without_nmp,
        search_without_probcut, search_without_qs_early_cutoff, search_without_qs_futility_pruning,
        search_without_qs_history_pruning, search_without_qs_late_capture_pruning, search_without_qs_see_pruning,
        search_without_razoring, search_without_rfp, search_without_se, search_without_see_pruning,
    },
    transposition::{Bound, TranspositionTable},
    types::{Move, MoveKind, Square},
};

#[test]
fn depth_zero_scores_material_from_the_side_to_move() {
    for (piece, value) in [('P', 100), ('N', 320), ('B', 330), ('R', 500), ('Q', 900)] {
        for (side, sign) in [('w', 1), ('b', -1)] {
            // The extra piece cannot win either pawn in its quiescent tail.
            // The balanced pawn pair also keeps the material-draw policy out.
            let board = Board::from_fen(&format!("7k/7p/8/8/2{piece}5/8/7P/7K {side} - - 0 1")).unwrap();
            let original = board.clone();
            let result = search(&board, 0);
            assert_eq!(result.score, sign * value, "{piece}, {side} to move");
            assert_eq!(result.best_move, None);
            assert!(board == original);
        }
    }
    assert_eq!(search(&Board::starting_position(), 0).score, 0);
}

#[test]
fn terminal_positions_take_precedence_over_material_and_depth() {
    for (fen, score) in [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -32_000),
        ("8/8/8/8/8/6k1/6q1/7K w - - 0 1", -32_000),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", -2),
        ("8/8/8/8/8/6k1/5q2/7K w - - 0 1", -2),
    ] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        for depth in [0, 1, 3] {
            let result = search(&board, depth);
            assert_eq!(result.score, score, "{fen}, depth {depth}");
            assert_eq!(result.best_move, None);
            assert_eq!(result.nodes, 0);
            assert!(board == original);
        }
    }
}

#[test]
fn finds_mate_in_one_for_either_side_without_changing_the_position() {
    for fen in ["7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "8/8/8/8/8/6k1/5q2/7K b - - 0 1"] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        for depth in [1, 2, 3] {
            let result = search(&board, depth);
            let mv = result.best_move.expect("a mating move must be returned");
            assert!(board.is_legal(mv));
            assert_eq!(result.score, 31_999);
            assert!(result.nodes > 1);
            assert!(board == original);

            let mut child = board.clone();
            child.make_move(mv);
            assert!(child.in_check());
            assert!(!child.has_legal_moves(), "selected move is not checkmate: {mv:?}");
        }
    }
}

#[test]
fn search_preserves_existing_history_and_resets_node_counts_each_call() {
    let mut board = Board::starting_position();
    let start = board.clone();
    let mut history = Vec::new();
    for uci in ["e2e4", "d7d5", "e4d5", "d8d5"] {
        let mv = board.generate_all_moves().iter().copied().find(|mv| mv.to_uci(&board) == uci).unwrap();
        board.make_move(mv);
        history.push(mv);
    }

    let original = board.clone();
    for depth in [1, 3, 0, 2] {
        let first = search(&board, depth);
        let second = search(&board, depth);
        assert_eq!(first, second);
        assert!(board == original);
        if depth > 0 {
            assert!(board.is_legal(first.best_move.unwrap()));
        }
    }

    for mv in history.into_iter().rev() {
        board.undo_move(mv);
    }
    assert!(board == start, "search damaged history belonging to its caller");
}

#[test]
fn depth_one_counts_twenty_starting_moves() {
    let result = search(&Board::starting_position(), 1);
    assert_eq!(result.nodes, 20);
    assert_eq!(result.score, 0);
    assert!(result.best_move.is_some());
}

#[test]
fn forced_loss_has_a_negative_ply_adjusted_mate_score() {
    let board = Board::from_fen("7k/8/5K1Q/8/8/8/8/8 b - - 0 1").unwrap();
    let original = board.clone();
    let result = search(&board, 2);
    assert_eq!(result.best_move.unwrap().to_uci(&board), "h8g8");
    assert_eq!(result.score, -31_998);
    assert!(board == original);
}

#[test]
fn zero_draw_mode_keeps_exact_draw_scores_without_changing_other_scores() {
    for (fen, depth, score) in [
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0, 0),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, 0),
        ("7k/7P/8/6K1/8/8/8/8 w - - 0 1", 2, 0),
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", 1, 31_999),
        ("7k/8/5K1Q/8/8/8/8/8 b - - 0 1", 2, -31_998),
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 0, -32_000),
        ("7k/8/8/8/8/8/Q7/7K w - - 0 1", 0, 900),
    ] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        let result = search_with_zero_draws(&board, depth);
        assert_eq!(result.score, score, "{fen}, depth {depth}");
        assert_eq!(result.best_move.is_some(), depth > 0 && board.has_legal_moves());
        if let Some(mv) = result.best_move {
            assert!(board.is_legal(mv));
        }
        assert!(board == original);
    }
}

#[test]
fn draw_jitter_uses_the_search_node_count_and_resets_each_call() {
    // Blocked a-pawns preserve the material balance without adding legal
    // moves, and keep king-only material draws out of this stalemate fixture.
    for (fen, score, nodes, best_move) in
        [("7k/7P/8/p5K1/P7/8/8/8 w - - 0 1", 2, 40, "g5h6"), ("7k/7P/5K2/p7/P7/8/8/8 w - - 0 1", 2, 31, "f6g6")]
    {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        let result = search(&board, 2);

        // Iterative deepening (depths 1..=2) visits these stalemates at deterministic
        // node counts where Black receives -2 and negamax returns +2 to White.
        assert_eq!(result.score, score);
        assert_eq!(result.nodes, nodes);
        let mv = result.best_move.unwrap();
        assert_eq!(mv.to_uci(&board), best_move);
        let mut child = board.clone();
        child.make_move(mv);
        assert!(!child.in_check() && !child.has_legal_moves());

        assert_eq!(search_with_zero_draws(&board, 2).score, 0);
        search(&Board::starting_position(), 1);
        assert_eq!(search(&board, 2), result);
        assert!(board == original);
    }
}

#[test]
fn iterative_deepening_accumulates_iterations_and_resolves_aspiration_swings() {
    let start = Board::starting_position();
    let d1 = search(&start, 1);
    let d2 = search(&start, 2);
    let d3 = search(&start, 3);
    assert_eq!(d1.nodes, 20);
    assert!(d2.nodes > d1.nodes);
    assert!(d3.nodes > d2.nodes);

    // Mate-in-2 (3 plies): depth 1 scores +31_997 via QS quiet checks, and
    // deeper iterations preserve the mate through narrow aspiration windows
    // without reducing depth on a decisive fail-high.
    let mate_in_two = Board::from_fen("7k/8/5K2/8/8/8/3Q4/8 w - - 0 1").unwrap();
    for depth in [1, 2, 3] {
        let result = search_with_zero_draws(&mate_in_two, depth);
        assert_eq!(result.score, 31_997, "depth {depth}");
        assert_eq!(result.best_move.unwrap().to_uci(&mate_in_two), "d2h6");
        assert_eq!(result.pv()[0].to_uci(&mate_in_two), "d2h6");
    }

    // Quiet double-attack across iterations: at depth 1 c3d5 is a quiet
    // non-checking move so Black stands pat in QS at +120; at depth 2 c3d5
    // forks the pawns on b6 and f4, jumping the score by +100 cp to +220
    // and exercising aspiration fail-high widening and re-search.
    let quiet_fork = Board::from_fen("7k/8/1p6/8/5p2/2N5/8/7K w - - 0 1").unwrap();
    let d1_fork = search_with_zero_draws(&quiet_fork, 1);
    let d2_fork = search_with_zero_draws(&quiet_fork, 2);
    assert_eq!(d1_fork.score, 120);
    assert_eq!(d2_fork.score, 220);
    assert_eq!(d2_fork.best_move.unwrap().to_uci(&quiet_fork), "c3d5");
    assert_eq!(d2_fork.pv()[0].to_uci(&quiet_fork), "c3d5");

    // Fail-low across iterations: at depth 1 Black sees only -900 material
    // (since mate requires a quiet move beyond the two-ply QS check budget),
    // then at depth 2 drops to -31_996 out of the [-991, -809] aspiration
    // window when the forced mate enters the horizon.
    let fail_low = Board::from_fen("7k/8/5K2/8/8/8/3Q4/8 b - - 0 1").unwrap();
    let d1_low = search_with_zero_draws(&fail_low, 1);
    let d2_low = search_with_zero_draws(&fail_low, 2);
    assert_eq!(d1_low.score, -900);
    assert_eq!(d2_low.score, -31_996);
    assert!(d2_low.nodes > d1_low.nodes);
}

#[test]
fn lmr_table_and_late_move_reductions_preserve_tactics_while_reducing_tree_size() {
    assert_eq!(lmr_table_term(1, 1), 0);
    assert_eq!(lmr_table_term(8, 8), 1_730);
    assert!(lmr_table_term(12, 16) > lmr_table_term(8, 8));

    // At depths 1 and 2, Black's quiet fork 1...Nd3 (attacking b4 and f2) has
    // not yet resulted in a capture, so a1b1 scores -120 and remains the root
    // work_item (move_count == 1) entering depth 3. At depth 3, a1b1/a1a2/a1b2
    // allow 1...Nd3 and lose a pawn (-220), while the late quiet pawn moves
    // (generated after the king moves at move_count >= 4) are searched under
    // root LMR, beat the failed-low alpha on their reduced scout, and go
    // through full-depth verification + PV re-search to hold -120.
    let lmr_rescue = Board::from_fen("7k/8/8/4n3/1P6/8/5P2/K7 w - - 0 1").unwrap();
    let d1 = search_with_zero_draws(&lmr_rescue, 1);
    let d2 = search_with_zero_draws(&lmr_rescue, 2);
    let d3 = search_with_zero_draws(&lmr_rescue, 3);
    let ref3 = search_reference(&lmr_rescue, 3);
    assert_eq!(d1.best_move.unwrap().to_uci(&lmr_rescue), "a1b1");
    assert_eq!(d2.best_move.unwrap().to_uci(&lmr_rescue), "a1b1");
    assert_eq!(d3.score, -120);
    assert_eq!(d3.score, ref3.score);
    assert!(matches!(d3.best_move.unwrap().to_uci(&lmr_rescue).as_str(), "f2f3" | "f2f4" | "b4b5"));

    // Same selective engine (TT, history, PVS, aspiration, selective QS) with LMR on vs off:
    let start = Board::starting_position();
    let lmr_on = search_with_zero_draws(&start, 5);
    let lmr_off = search_without_lmr(&start, 5);
    assert!(
        lmr_on.nodes < lmr_off.nodes,
        "LMR should reduce nodes in the same selective engine ({} vs {})",
        lmr_on.nodes,
        lmr_off.nodes
    );
}

#[test]
fn null_move_pruning_reduces_nodes_and_skips_pawn_only_zugzwang() {
    // 1. In a middlegame position with non-pawn material on both sides, NMP
    // prunes non-PV subtrees where passing the turn still holds beta.
    let italian = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
    let nmp_on = search_with_zero_draws(&italian, 6);
    let nmp_off = search_without_nmp(&italian, 6);
    assert_eq!(nmp_on.score, nmp_off.score);
    assert!(
        nmp_on.nodes < nmp_off.nodes,
        "NMP should reduce nodes in the same selective engine ({} vs {})",
        nmp_on.nodes,
        nmp_off.nodes
    );

    // 2. In a king-and-pawn zugzwang endgame (0 non-pawn pieces), the
    // non_pawn_count guard disables NMP completely so node counts and moves match.
    let kp_zugzwang = Board::from_fen("8/8/8/2k5/3pP3/5K2/8/8 w - - 0 1").unwrap();
    let z_on = search_with_zero_draws(&kp_zugzwang, 4);
    let z_off = search_without_nmp(&kp_zugzwang, 4);
    assert_eq!(z_on.best_move, z_off.best_move);
    assert_eq!(z_on.score, z_off.score);
    assert_eq!(z_on.nodes, z_off.nodes);

    // 3. A TT collision storing an illegal noisy move (or an illegal quiet move
    // on a cutoff entry) at a child node where NMP runs must not panic in SEE
    // or history updates.
    let tt = TranspositionTable::new(1);
    let illegal_capture = Move::new(Square::A1, Square::H8, MoveKind::Capture);
    let illegal_quiet = Move::new(Square::A4, Square::A5, MoveKind::Normal);
    for &mv in italian.generate_all_moves().iter() {
        let mut child = italian.clone();
        child.make_move(mv);
        tt.write(child.hash(), 1, 200, 200, Bound::Lower, illegal_capture, 1, false, false);
        for &reply in child.generate_all_moves().iter() {
            let mut grandchild = child.clone();
            grandchild.make_move(reply);
            tt.write(grandchild.hash(), 8, 200, 200, Bound::Lower, illegal_quiet, 2, false, false);
        }
    }
    let res = search_with_tt(&italian, 4, &tt);
    assert!(res.best_move.is_some());
}

#[test]
fn reverse_futility_pruning_reduces_nodes_and_skips_pawn_only_endgames() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let rfp_on = search_with_zero_draws(&kiwipete, 5);
    let rfp_off = search_without_rfp(&kiwipete, 5);
    assert_eq!(rfp_on.best_move, rfp_off.best_move);
    assert!(
        rfp_on.nodes < rfp_off.nodes,
        "RFP should reduce nodes in the same selective engine ({} vs {})",
        rfp_on.nodes,
        rfp_off.nodes
    );

    // Pawn-only endgame with a 1-pawn lead: non_pawn_count == 0 disables RFP.
    let kp_lead = Board::from_fen("8/8/8/2k5/3pP3/5K2/6P1/8 w - - 0 1").unwrap();
    let kp_on = search_with_zero_draws(&kp_lead, 4);
    let kp_off = search_without_rfp(&kp_lead, 4);
    assert_eq!(kp_on.best_move, kp_off.best_move);
    assert_eq!(kp_on.score, kp_off.score);
    assert_eq!(kp_on.nodes, kp_off.nodes);
}

#[test]
fn probcut_prunes_tactical_captures_and_validates_collided_tt_moves() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let pc_on = search_with_zero_draws(&kiwipete, 6);
    let pc_off = search_without_probcut(&kiwipete, 6);
    assert_eq!(pc_on.best_move, pc_off.best_move);
    assert!(
        pc_on.nodes < pc_off.nodes,
        "ProbCut should reduce nodes on tactical positions ({} vs {})",
        pc_on.nodes,
        pc_off.nodes
    );

    // Direct MainMovePicker::new_probcut verification:
    // 1. Rejects an illegal noisy TT move without panicking in see_ge.
    // 2. Yields only captures meeting see_threshold (here >= 300, i.e. winning at least a minor piece).
    // 3. Stops at MainStage::Done without yielding quiet or bad-noisy moves.
    let illegal_capture = Move::new(Square::A1, Square::H8, MoveKind::Capture);
    let mut picker = MainMovePicker::new_probcut(illegal_capture, 300);
    let mut yielded = Vec::new();
    while let Some(mv) = picker.next(&kiwipete, None, true, 0) {
        assert!(mv.is_noisy());
        assert!(kiwipete.is_legal(mv));
        assert!(kiwipete.see_ge(mv, 300));
        yielded.push(mv);
    }
    assert!(!yielded.is_empty());
    assert!(picker.stage() == MainStage::Done);

    // Low-SEE legal noisy TT move (e5xf7 loses the knight: SEE < 300) is rejected
    // in HashMove and not re-emitted in GoodNoisy.
    let bad_tt_capture = Move::new(Square::E5, Square::F7, MoveKind::Capture);
    assert!(kiwipete.is_legal(bad_tt_capture));
    assert!(!kiwipete.see_ge(bad_tt_capture, 300));
    let mut picker2 = MainMovePicker::new_probcut(bad_tt_capture, 300);
    while let Some(mv) = picker2.next(&kiwipete, None, true, 0) {
        assert_ne!(mv, bad_tt_capture);
        assert!(kiwipete.see_ge(mv, 300));
    }
}

#[test]
fn main_loop_see_pruning_reduces_nodes_and_preserves_tactics() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let see_on = search_with_zero_draws(&kiwipete, 5);
    let see_off = search_without_see_pruning(&kiwipete, 5);
    assert_eq!(see_on.best_move, see_off.best_move);
    assert!(
        see_on.nodes < see_off.nodes,
        "Main-loop SEE pruning should reduce nodes in the same selective engine ({} vs {})",
        see_on.nodes,
        see_off.nodes
    );
}

#[test]
fn late_move_pruning_closes_quiet_stage_and_reduces_nodes() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let lmp_on = search_with_zero_draws(&kiwipete, 5);
    let lmp_off = search_without_lmp(&kiwipete, 5);
    assert_eq!(lmp_on.best_move, lmp_off.best_move);
    assert!(
        lmp_on.nodes < lmp_off.nodes,
        "LMP should reduce nodes in the same selective engine ({} vs {})",
        lmp_on.nodes,
        lmp_off.nodes
    );
}

#[test]
fn main_loop_futility_pruning_reduces_nodes_and_exempts_direct_checks() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let fp_on = search_with_zero_draws(&kiwipete, 6);
    let fp_off = search_without_fp(&kiwipete, 6);
    assert_eq!(fp_on.best_move, fp_off.best_move);
    assert!(
        fp_on.nodes < fp_off.nodes,
        "Futility pruning should reduce nodes in the same selective engine ({} vs {})",
        fp_on.nodes,
        fp_off.nodes
    );

    // Surgical check-exemption test: in this position White is down 1100 cp
    // and at depths 1-2 selects e6e1 (-1100). At depth 3, the sacrifice
    // 1. Re6xe7+ (BadNoisy, !PV) 1...Kxe7 reaches ply 2 (depth = 1, !PV) with
    // static_eval = -1500 and alpha = -1100 (so static_eval + margin <= alpha
    // activates FP after 2. fxg6 sets best_score = -1400 > -MATE_THRESHOLD).
    // The quiet direct check 2. Nc3-d5+ (clearing c3 for 2...Ke8 3. Ra3-e3#)
    // is saved from FP solely by !board.is_direct_check(mv), finding mate (+31_995).
    let check_mate = Board::from_fen("nr1rkr2/pp1ppp2/3pR1p1/5P2/8/R1N5/8/7K w - - 0 1").unwrap();
    let d2 = search_with_zero_draws(&check_mate, 2);
    let cf_on = search_with_zero_draws(&check_mate, 3);
    let cf_off = search_without_fp(&check_mate, 3);
    assert_eq!(d2.score, -1_100);
    assert_eq!(cf_on.best_move.unwrap().to_uci(&check_mate), "e6e7");
    assert_eq!(cf_on.score, 31_995);
    assert_eq!(cf_on.score, cf_off.score);
}

#[test]
fn quiet_history_pruning_reduces_nodes_and_preserves_tactics() {
    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let hp_on = search_with_zero_draws(&midgame, 5);
    let hp_off = search_without_hp(&midgame, 5);
    assert_eq!(hp_on.best_move, hp_off.best_move);
    assert!(
        hp_on.nodes < hp_off.nodes,
        "Quiet-history pruning should reduce nodes in the same selective engine ({} vs {})",
        hp_on.nodes,
        hp_off.nodes
    );
}

#[test]
fn internal_iterative_reduction_reduces_cut_node_work_without_tt_move() {
    let italian = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
    let iir_on = search_with_zero_draws(&italian, 7);
    let iir_off = search_without_iir(&italian, 7);
    assert_eq!(iir_on.best_move, iir_off.best_move);
    assert!(
        iir_on.nodes < iir_off.nodes,
        "IIR should reduce nodes in the same selective engine ({} vs {})",
        iir_on.nodes,
        iir_off.nodes
    );

    // A collided TT entry storing an illegal move at high depth must have its
    // tt_move sanitized to Move::NULL on probe so cut-node IIR still triggers
    // identically to an entry with Move::NULL.
    let tt_illegal = TranspositionTable::new(1);
    let tt_null = TranspositionTable::new(1);
    let illegal_mv = Move::new(Square::A1, Square::H8, MoveKind::Normal);
    for &mv in italian.generate_all_moves().iter() {
        let mut child = italian.clone();
        child.make_move(mv);
        tt_illegal.write(child.hash(), 10, 0, 0, Bound::Lower, illegal_mv, 1, false, false);
        tt_null.write(child.hash(), 10, 0, 0, Bound::Lower, Move::NULL, 1, false, false);
    }
    let res_illegal = search_with_tt(&italian, 4, &tt_illegal);
    let res_null = search_with_tt(&italian, 4, &tt_null);
    assert_eq!(res_illegal.nodes, res_null.nodes);
    assert_eq!(res_illegal.best_move, res_null.best_move);
}

#[test]
fn singular_extensions_extend_unique_tt_moves_and_multi_cut_alternatives() {
    // 1. On Midgame at depth 8, Multi-Cut and negative extensions reduce nodes
    // while preserving the exact score and best move.
    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let se_on = search_with_zero_draws(&midgame, 8);
    let se_off = search_without_se(&midgame, 8);
    assert_eq!(se_on.best_move, se_off.best_move);
    assert_eq!(se_on.score, se_off.score);
    assert!(
        se_on.nodes < se_off.nodes,
        "SE multi-cut and negative extensions should reduce nodes on Midgame ({} vs {})",
        se_on.nodes,
        se_off.nodes
    );

    // 2. Direct Multi-Cut verification (verify >= beta):
    // After the noisy root move c2xd3 (not reduced by root LMR), at ply 1
    // (depth = 6 at step 7, !PV) Black has two independent queen-winning knight
    // captures (d6xe4 and f6xg4). At step 6 (depth = 5 at ply 1), d6xe4 fails
    // high and stores depth = 4 (5 - 1) in the TT. At step 7 (depth = 6 at
    // ply 1), the excluded-move verification search (excluding d6xe4) finds
    // f6xg4 >= beta and returns a Multi-Cut early without overwriting the TT
    // entry at depth = 5 (6 - 1).
    let mc_pos = Board::from_fen("r1r5/1rp5/1k1n1n2/8/4Q1Q1/1p1p4/P1P5/K7 w - - 0 1").unwrap();
    let d6e4 = Move::new(Square::D6, Square::E4, MoveKind::Capture);
    let c2d3 = Move::new(Square::C2, Square::D3, MoveKind::Capture);
    let mut child_after_c2d3 = mc_pos.clone();
    child_after_c2d3.make_move(c2d3);
    assert!(child_after_c2d3.is_legal(d6e4));

    let tt_mc = TranspositionTable::new(4);
    tt_mc.write(child_after_c2d3.hash(), 5, -200, 500, Bound::Lower, d6e4, 1, false, false);
    let _ = search_with_tt(&mc_pos, 7, &tt_mc);
    let after = tt_mc.read(child_after_c2d3.hash(), 0, 1).unwrap();
    assert_eq!(after.depth, 4, "Multi-Cut at depth 6 must return before overwriting TT at depth 5");

    // 3. On Midgame at depth 8 and Kiwipete at depth 7, extending singular
    // tactical moves and multi-cutting alternatives reduces nodes while
    // preserving the exact best move and score.
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let kw_on = search_with_zero_draws(&kiwipete, 7);
    let kw_off = search_without_se(&kiwipete, 7);
    assert_eq!(kw_on.best_move, kw_off.best_move);
    assert_eq!(kw_on.score, kw_off.score);
    assert!(kw_on.nodes < kw_off.nodes);
}

#[test]
fn razoring_prunes_hopeless_nodes_and_preserves_tactical_refutations() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let rz_on = search_with_zero_draws(&kiwipete, 7);
    let rz_off = search_without_razoring(&kiwipete, 7);
    assert_eq!(rz_on.best_move, rz_off.best_move);
    assert_eq!(rz_on.score, rz_off.score);
    assert!(
        rz_on.nodes < rz_off.nodes,
        "Razoring should reduce nodes on Kiwipete ({} vs {})",
        rz_on.nodes,
        rz_off.nodes
    );

    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let mid_on = search_with_zero_draws(&midgame, 7);
    let mid_off = search_without_razoring(&midgame, 7);
    assert_eq!(mid_on.best_move, mid_off.best_move);
    assert_eq!(mid_on.score, mid_off.score);
    assert!(
        mid_on.nodes < mid_off.nodes,
        "Razoring should reduce nodes on Midgame ({} vs {})",
        mid_on.nodes,
        mid_off.nodes
    );

    // Tactical refutation in quiescence: Black is down 1,180 cp (White's Queen,
    // Rook, and Pawn vs Black's Knight) in static material at ply 1 (depth = 2,
    // !PV) after a quiet White move, triggering the razoring probe, but Black
    // has a knight capture winning White's Queen in quiescence (q_score > alpha),
    // aborting the razor prune.
    let trap = Board::from_fen("1k6/8/3n4/8/4Q3/8/P7/K6R w - - 0 1").unwrap();
    let trap_on = search_with_zero_draws(&trap, 4);
    let trap_off = search_without_razoring(&trap, 4);
    assert_eq!(trap_on.best_move, trap_off.best_move);
    assert_eq!(trap_on.score, trap_off.score);
}

#[test]
fn checked_tt_capture_cutoff_prunes_verified_check_evasion_captures() {
    // White has Move 1: 1. a2xb3 (captures a knight), and a checking move
    // 1. Re1-e8+ where Black's rook on a8 can capture 1... Ra8xe8 (a legal
    // capture evasion when in check!).
    // At step 6 (reduced to depth = 4 at ply 1 by root LMR), 1... Ra8xe8 fails
    // high and stores depth = 3 (4 - 1), Bound::Lower, mv = a8e8 (capture!) in
    // the TT. At step 7 (reduced to depth = 5 at ply 1 by root LMR, !PV,
    // in_check == true), entry.depth (3) < 5 (so early TT cutoff does not
    // fire), but 3 >= 5 - 4 and score >= beta + 200 with tt_move.is_capture(),
    // so checked TT-capture cutoff returns early without overwriting the TT
    // entry at depth = 4 (5 - 1).
    let pos = Board::from_fen("r5k1/5ppp/8/8/8/1n6/P4PPP/4R1K1 w - - 0 1").unwrap();
    let e1e8 = Move::new(Square::E1, Square::E8, MoveKind::Normal);
    let a8e8 = Move::new(Square::A8, Square::E8, MoveKind::Capture);
    let mut child_after_e1e8 = pos.clone();
    child_after_e1e8.make_move(e1e8);
    assert!(child_after_e1e8.in_check());
    assert!(child_after_e1e8.is_legal(a8e8));

    let tt = TranspositionTable::new(4);
    let _ = search_with_tt(&pos, 7, &tt);
    let after = tt.read(child_after_e1e8.hash(), 0, 1).unwrap();
    assert_eq!(after.mv, a8e8);
    assert_eq!(
        after.depth, 3,
        "Checked TT-capture cutoff at depth 5 (step 7 after root LMR) must return before overwriting the depth-3 TT entry"
    );

    let on = search_with_zero_draws(&pos, 7);
    let off = search_without_checked_tt_cutoff(&pos, 7);
    assert_eq!(on.best_move, off.best_move);
    assert_eq!(on.score, off.score);
    assert!(
        on.nodes < off.nodes,
        "Checked TT-capture cutoff should reduce nodes on checked capture evasions ({} vs {})",
        on.nodes,
        off.nodes
    );
}

#[test]
fn post_move_late_quiet_closure_reduces_nodes_and_preserves_tactics() {
    let italian = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
    let on = search_with_zero_draws(&italian, 7);
    let off = search_without_late_quiet_closure(&italian, 7);
    assert_eq!(on.best_move, off.best_move);
    assert_eq!(on.score, off.score);
    assert!(
        on.nodes < off.nodes,
        "Post-move late-quiet closure should reduce nodes on Italian ({} vs {})",
        on.nodes,
        off.nodes
    );

    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let mid_on = search_with_zero_draws(&midgame, 7);
    let mid_off = search_without_late_quiet_closure(&midgame, 7);
    assert_eq!(mid_on.best_move, mid_off.best_move);
    assert_eq!(mid_on.score, mid_off.score);
    assert!(
        mid_on.nodes < mid_off.nodes,
        "Post-move late-quiet closure should reduce nodes on Midgame ({} vs {})",
        mid_on.nodes,
        mid_off.nodes
    );

    // At ply 2 (depth = 1) after 1. Re6xe7+ Kxe7, non-checking quiets are
    // skipped pre-move by FP, so the first searched Quiet-stage move is the
    // non-mating check 2. Ra3-a7+. The !gives_check guard prevents that
    // searched checking quiet from closing the Quiet stage before the mating
    // check 2. Nc3-d5+ is reached.
    let check_mate = Board::from_fen("nr1rkr2/pp1ppp2/3pR1p1/5P2/8/R1N5/8/7K w - - 0 1").unwrap();
    let mate_on = search_with_zero_draws(&check_mate, 3);
    let mate_off = search_without_late_quiet_closure(&check_mate, 3);
    assert_eq!(mate_on.best_move.unwrap().to_uci(&check_mate), "e6e7");
    assert_eq!(mate_on.score, 31_995);
    assert_eq!(mate_on.score, mate_off.score);
}

#[test]
fn quiescence_quiet_history_pruning_skips_negative_continuation_checks() {
    let italian = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
    let on = search_with_zero_draws(&italian, 9);
    let off = search_without_qs_history_pruning(&italian, 9);
    assert_eq!(on.best_move, off.best_move);
    assert_eq!(on.score, off.score);
    assert!(
        on.nodes < off.nodes,
        "Quiescence quiet-history pruning should reduce nodes on Italian ({} vs {})",
        on.nodes,
        off.nodes
    );

    // Direct 1-ply continuation history check in quiescence:
    // After White's root move, Black is at ply 1 (q_depth = 0) with a quiet
    // check 1... Rd8-d1+. Seeding the 1-ply continuation entries for
    // (White root move -> BlackRook to d1) below QS_QUIET_HIST_THRESHOLD (-350)
    // causes quiescence to skip 1... Rd8-d1+, reducing the depth-1 node count
    // compared to an unseeded history table.
    let pos = Board::from_fen("3r2k1/8/8/8/8/8/P7/6KR w - - 0 1").unwrap();
    let tt_clean = TranspositionTable::new(1);
    let mut hist_clean = asura::history::HistoryTables::default();
    let base = search_with_context(&pos, 1, &tt_clean, &mut hist_clean);

    let tt_seeded = TranspositionTable::new(1);
    let mut hist_seeded = asura::history::HistoryTables::default();
    for &mv in pos.generate_all_moves().iter() {
        hist_seeded.continuation_history.remember(0, false, mv, pos.moved_piece(mv));
        hist_seeded.continuation_history.update(1, 1, asura::types::Piece::BlackRook, Square::D1, -2_000);
    }
    let pruned = search_with_context(&pos, 1, &tt_seeded, &mut hist_seeded);
    assert!(
        pruned.nodes < base.nodes,
        "Negative 1-ply continuation history must prune the quiet check in quiescence ({} vs {})",
        pruned.nodes,
        base.nodes
    );
}

#[test]
fn quiescence_capture_futility_pruning_reduces_nodes_and_preserves_tactics() {
    let kiwipete = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let kw_on = search_with_zero_draws(&kiwipete, 6);
    let kw_off = search_without_qs_futility_pruning(&kiwipete, 6);
    assert_eq!(kw_on.best_move, kw_off.best_move);
    assert_eq!(kw_on.score, kw_off.score);
    assert!(
        kw_on.nodes < kw_off.nodes,
        "Quiescence capture-futility pruning should reduce nodes on Kiwipete ({} vs {})",
        kw_on.nodes,
        kw_off.nodes
    );

    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let mid_on = search_with_zero_draws(&midgame, 7);
    let mid_off = search_without_qs_futility_pruning(&midgame, 7);
    assert_eq!(mid_on.best_move, mid_off.best_move);
    assert_eq!(mid_on.score, mid_off.score);
    assert!(
        mid_on.nodes < mid_off.nodes,
        "Quiescence capture-futility pruning should reduce nodes on Midgame ({} vs {})",
        mid_on.nodes,
        mid_off.nodes
    );

    // In this position at depth 1, Move 1 at the root is 1. Rd1xd8 (trades
    // rooks, score = 0), and Move 2 is 1. Qa4xe4 (captures Black's Queen on
    // e4, leaving Black with stand_pat = -900 and margin = -700 at ply 1 while
    // alpha = 0). Black's pawn capture 1... a7xb6 has capture_margin = -600 <=
    // alpha (0) and is pruned by quiescence delta futility.
    let delta_pos = Board::from_fen("3r2k1/p7/1P6/8/Q3q3/8/8/K2R4 w - - 0 1").unwrap();
    let d_on = search_with_zero_draws(&delta_pos, 1);
    let d_off = search_without_qs_futility_pruning(&delta_pos, 1);
    assert_eq!(d_on.best_move, d_off.best_move);
    assert_eq!(d_on.score, d_off.score);
    assert!(
        d_on.nodes < d_off.nodes,
        "Quiescence delta futility must prune 1... a7xb6 after 1. Qa4xe4 ({} vs {})",
        d_on.nodes,
        d_off.nodes
    );

    // Equal-trade SEE futility branch (alpha >= margin && !see_ge(mv, 1)):
    // After 1. Rb7xa7 Ra8xa7 and 1. Rb2xd2 Rd8xd2 (equal rook trades, score = 0)
    // establish alpha at the root, Move 3 (1. Rb2xg2, capturing undefended Ng2)
    // leaves Black at ply 1 with stand_pat = -320 and margin = -120 <= alpha (-1),
    // while capture_margin = +380 > alpha (-1). Black's equal rook trade
    // 1... Ra7xb7 has SEE == 0 (passing see_ge(mv, 0) but failing see_ge(mv, 1)),
    // so it is pruned solely by the second branch (alpha >= margin && !see_ge(mv, 1)).
    let see_pos = Board::from_fen("r2r3k/rR6/2K5/8/8/8/1R1r2n1/8 w - - 0 1").unwrap();
    let s_on = search_with_zero_draws(&see_pos, 1);
    let s_off = search_without_qs_futility_pruning(&see_pos, 1);
    assert_eq!(s_on.best_move, s_off.best_move);
    assert_eq!(s_on.score, s_off.score);
    assert!(
        s_on.nodes < s_off.nodes,
        "Quiescence equal-trade SEE futility must prune equal rook trades after 1. Rb2xg2 ({} vs {})",
        s_on.nodes,
        s_off.nodes
    );
}

#[test]
fn quiescence_late_capture_pruning_skips_late_equal_trades_and_preserves_tactics() {
    let italian = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
    let it_on = search_with_zero_draws(&italian, 7);
    let it_off = search_without_qs_late_capture_pruning(&italian, 7);
    assert_eq!(it_on.best_move, it_off.best_move);
    assert_eq!(it_on.score, it_off.score);
    assert!(
        it_on.nodes < it_off.nodes,
        "Quiescence late-capture pruning should reduce nodes on Italian ({} vs {})",
        it_on.nodes,
        it_off.nodes
    );

    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let mid_on = search_with_zero_draws(&midgame, 7);
    let mid_off = search_without_qs_late_capture_pruning(&midgame, 7);
    assert_eq!(mid_on.best_move, mid_off.best_move);
    assert_eq!(mid_on.score, mid_off.score);
    assert!(
        mid_on.nodes < mid_off.nodes,
        "Quiescence late-capture pruning should reduce nodes on Midgame ({} vs {})",
        mid_on.nodes,
        mid_off.nodes
    );

    // Direct depth-1 quiescence test where alpha < stand_pat + 200 (so capture
    // futility does not fire): Move 1 is searched, while Move 2+ (move_count > 1)
    // is skipped by the quiescence late-capture gate.
    let late_cap = Board::from_fen("1r1r1r1k/1R1R1Rp1/8/8/8/8/1R1R1R2/6K1 w - - 0 1").unwrap();
    let lc_on = search_with_zero_draws(&late_cap, 1);
    let lc_off = search_without_qs_late_capture_pruning(&late_cap, 1);
    assert_eq!(lc_on.best_move, lc_off.best_move);
    assert_eq!(lc_on.score, lc_off.score);
    assert!(
        lc_on.nodes < lc_off.nodes,
        "Quiescence late-capture gate must prune 2nd+ unrelated captures ({} vs {})",
        lc_on.nodes,
        lc_off.nodes
    );
}

#[test]
fn quiescence_early_cutoff_fails_high_on_history_backed_winning_captures() {
    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let on = search_with_zero_draws(&midgame, 7);
    let off = search_without_qs_early_cutoff(&midgame, 7);
    assert_eq!(on.best_move, off.best_move);
    assert_eq!(on.score, off.score);
    assert!(
        on.nodes < off.nodes,
        "Quiescence history-backed early fail-high should reduce nodes on Midgame ({} vs {})",
        on.nodes,
        off.nodes
    );

    // Direct history-gated test: after Move 1 (1. c5xd6 c7xd6) establishes
    // alpha at the root, Move 2 (1. c5xb6) vacates c5 and unveils White's
    // Queen on a3 to Black's Bishop on d6 (1... Bd6xa3, victim_val = 900,
    // attacker_val = 330, victim - attacker - 225 = +345). With positive
    // noisy_history for BlackBishop -> a3 x Queen, quiescence fails high
    // BEFORE calling make_move(1... Bd6xa3), using strictly fewer nodes than
    // with zero noisy_history.
    let pos = Board::from_fen("6k1/2p1p3/1n1b4/2P5/8/Q7/8/6K1 w - - 0 1").unwrap();
    let tt_zero = TranspositionTable::new(1);
    let mut hist_zero = asura::history::HistoryTables::default();
    let res_zero = search_with_context(&pos, 1, &tt_zero, &mut hist_zero);

    let tt_pos = TranspositionTable::new(1);
    let mut hist_pos = asura::history::HistoryTables::default();
    for mask in [asura::types::Bitboard(0), asura::types::Bitboard(!0)] {
        hist_pos.noisy_history.update(
            mask,
            asura::types::Piece::BlackBishop,
            Square::A3,
            asura::types::PieceType::Queen,
            500,
        );
    }
    let res_pos = search_with_context(&pos, 1, &tt_pos, &mut hist_pos);
    assert!(
        res_pos.nodes < res_zero.nodes,
        "Positive noisy history must trigger QS early fail-high before make_move ({} vs {})",
        res_pos.nodes,
        res_zero.nodes
    );
}

#[test]
fn quiescence_see_pruning_skips_negative_see_captures_and_checks() {
    let italian = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
    let it_on = search_with_zero_draws(&italian, 7);
    let it_off = search_without_qs_see_pruning(&italian, 7);
    assert_eq!(it_on.best_move, it_off.best_move);
    assert_eq!(it_on.score, it_off.score);
    assert!(
        it_on.nodes < it_off.nodes,
        "Quiescence SEE pruning should reduce nodes on Italian ({} vs {})",
        it_on.nodes,
        it_off.nodes
    );

    let midgame = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
    let mid_on = search_with_zero_draws(&midgame, 7);
    let mid_off = search_without_qs_see_pruning(&midgame, 7);
    assert_eq!(mid_on.best_move, mid_off.best_move);
    assert_eq!(mid_on.score, mid_off.score);
    assert!(
        mid_on.nodes < mid_off.nodes,
        "Quiescence SEE pruning should reduce nodes on Midgame ({} vs {})",
        mid_on.nodes,
        mid_off.nodes
    );

    // Direct negative-SEE pruning in quiescence: after White's root moves,
    // Black at ply 1 (q_depth = 0) has a checking capture 1... Qxf2+ into a
    // square defended by White's Rook on f1 and King on g1 (SEE = +100 - 900 = -800 < 0)
    // and a quiet check 1... Qf8-f3+ into a square defended by White's Pawn on
    // g2 and Rook on f1 (SEE = -900 < 0). Both are skipped before make_move by
    // selective QS SEE pruning.
    let pos = Board::from_fen("5qk1/8/8/8/8/8/P4PP1/5RK1 w - - 0 1").unwrap();
    let p_on = search_with_zero_draws(&pos, 1);
    let p_off = search_without_qs_see_pruning(&pos, 1);
    assert_eq!(p_on.best_move, p_off.best_move);
    assert_eq!(p_on.score, p_off.score);
    assert!(
        p_on.nodes < p_off.nodes,
        "Quiescence SEE pruning must skip negative-SEE checking captures and quiet checks ({} vs {})",
        p_on.nodes,
        p_off.nodes
    );
}
