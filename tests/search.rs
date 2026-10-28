use asura::{
    board::Board,
    movepick::{MainMovePicker, MainStage},
    search::{
        lmr_table_term, search, search_reference, search_with_tt, search_with_zero_draws, search_without_lmr,
        search_without_nmp, search_without_probcut, search_without_rfp,
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
