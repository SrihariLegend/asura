use asura::{
    board::Board,
    search::{search, search_reference, search_with_zero_draws},
};

#[test]
fn checked_nodes_cannot_stand_pat_or_drop_their_only_evasion_type() {
    let board = Board::from_fen("6rk/8/8/8/8/8/8/6K1 w - - 0 1").unwrap();
    assert!(board.in_check());
    assert!(board.generate_all_moves().iter().all(|mv| mv.is_quiet()));
    let original = board.clone();
    let result = search_with_zero_draws(&board, 0);
    assert_eq!(result.score, -500);
    assert!(result.nodes > 1, "a checked node must play an escape, not stand pat");
    assert!(board == original);
}

#[test]
fn quiet_evasions_remain_available_while_the_best_reply_loses_by_mate() {
    // Ka1 permits ...Qb2#, but Kc1 avoids immediate mate. Do not switch off
    // the quiet-evasion stage merely because one legal reply was searched.
    let board = Board::from_fen("8/8/8/8/8/kq6/8/1K6 w - - 0 1").unwrap();
    assert!(board.in_check());
    assert_eq!(board.generate_all_moves()[0].to_uci(&board), "b1a1");
    let original = board.clone();
    assert_eq!(search_with_zero_draws(&board, 0).score, -900);
    assert_eq!(search_reference(&board, 0).score, -900);
    assert!(board == original);
}

#[test]
fn quiet_checks_are_limited_to_the_first_two_quiescence_plies() {
    // Qh6+ Kg8 Qg7# requires a quiet check at the third QS ply. Depth zero
    // must not find it, but one full-width ply gives QS enough checking budget.
    let board = Board::from_fen("7k/8/5K2/8/8/8/3Q4/8 w - - 0 1").unwrap();
    let original = board.clone();
    for run in [search, search_with_zero_draws, search_reference] {
        assert_eq!(run(&board, 0).score, 900);
        assert_eq!(run(&board, 1).score, 31_997);
    }
    assert!(board == original);
}

#[test]
fn an_unchecked_qs_node_without_candidates_keeps_stand_pat_even_if_stalemated() {
    let board = Board::from_fen("7k/7P/8/6K1/8/8/8/8 w - - 0 1").unwrap();
    let original = board.clone();
    for run in [search_with_zero_draws, search_reference] {
        let shallow = run(&board, 1);
        assert_eq!(shallow.score, 100, "the interior QS node must not probe for stalemate");
        let mut child = board.clone();
        child.make_move(shallow.best_move.unwrap());
        assert!(!child.in_check() && !child.has_legal_moves());
        assert_eq!(run(&board, 2).score, 0, "full-width search still recognizes stalemate");
        assert_eq!(run(&child, 0).score, 0, "the public root still recognizes stalemate");
    }
    assert!(board == original);
}

#[test]
fn quiescence_finds_checkmate_by_chess960_castling() {
    for (placement, right, castling) in [("6KR", "H", "g1h1"), ("5KR1", "G", "f1g1")] {
        // The king blocks the rook's ordinary move to f1; only castling
        // puts the rook on the mating file. Cover a stationary king and a swap.
        let mut board = Board::from_fen(&format!("4rkr1/4p1p1/8/8/8/8/8/{placement} w {right} - 0 1")).unwrap();
        board.set_frc(true);
        let original = board.clone();
        for run in [search, search_with_zero_draws, search_reference] {
            assert_eq!(run(&board, 0).score, 31_999);
            let result = run(&board, 1);
            assert_eq!(result.score, 31_999);
            assert_eq!(result.best_move.unwrap().to_uci(&board), castling);
        }
        assert!(board == original);
    }
}

#[test]
fn quiescence_searches_non_capturing_underpromotions() {
    let board = Board::from_fen("8/kBPN4/2K5/8/8/8/8/8 w - - 0 1").unwrap();
    let original = board.clone();
    for run in [search, search_with_zero_draws, search_reference] {
        assert_eq!(run(&board, 0).score, 31_999);
        let result = run(&board, 1);
        assert_eq!(result.score, 31_999);
        assert_eq!(result.best_move.unwrap().to_uci(&board), "c7c8n");
    }
    assert!(board == original);
}

#[test]
fn a_quiescence_capture_delivering_mate_beats_the_material_draw_policy() {
    for fen in ["7k/5K2/5Nr1/4N3/8/8/8/8 w - - 0 1", "8/8/8/8/4n3/5nR1/5k2/7K b - - 0 1"] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        for run in [search, search_with_zero_draws, search_reference] {
            assert_eq!(run(&board, 0).score, 31_999);
        }
        assert!(board == original);
    }
}

#[test]
fn depth_one_rejects_a_poisoned_pawn_instead_of_hanging_the_queen() {
    let board = Board::from_fen("6k1/8/4p3/3p4/8/8/8/3Q2K1 w - - 0 1").unwrap();
    let original = board.clone();
    let result = search(&board, 1);
    assert_ne!(result.best_move.unwrap().to_uci(&board), "d1d5");
    assert!(result.score >= 700);
    assert!(board == original);
}

#[test]
fn skipping_quiet_evasions_keeps_a_better_deferred_capture_reachable() {
    // Ka1-b1 avoids check, but ...Rxh8 then wins the queen. Qh8xa8 fails
    // SEE (-400), yet ...Kxa8 leaves bare kings and saves the game.
    let board = Board::from_fen("r6Q/1k6/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let original = board.clone();
    let capture = board.generate_all_moves().iter().copied().find(|mv| mv.to_uci(&board) == "h8a8").unwrap();
    assert!(board.in_check());
    assert!(!board.see_ge(capture, 0));
    let selective = search_with_zero_draws(&board, 0);
    let reference = search_reference(&board, 0);
    assert_eq!(selective.score, 0, "do not stop at the -500 quiet escape");
    assert_eq!(reference.score, 0);
    assert!(selective.nodes < reference.nodes, "skip the remaining quiet escapes, not the deferred capture");
    assert!((-2..=2).contains(&search(&board, 0).score));
    assert!(board == original);
}

#[test]
fn deep_search_keeps_recaptures_but_drops_unrelated_nonchecking_captures() {
    // In the Rh4+ Kg5 a8=Q Rxa8 Rxa8 continuation, the second Rxa8 is
    // still eligible at QS depth -4. The unrelated ...Kxh4 at depth -5
    // is not. Black instead takes on h4 earlier, producing -430.
    // The unrestricted reference sees the later capture and returns -730;
    // selective QS intentionally need not reproduce its minimax value.
    let board = Board::from_fen("7r/P5q1/Rp5k/8/4R3/4b3/8/7K w - - 0 1").unwrap();
    let original = board.clone();
    assert_eq!(search_with_zero_draws(&board, 0).score, -430);
    assert_eq!(search_reference(&board, 0).score, -730);
    assert!(board == original);
}

#[test]
fn deep_checking_captures_are_not_limited_to_recaptures() {
    // Rb6+ Kg5 Rxb7 Rxg4 leaves Rxg7+ at QS depth -4. It is not a
    // recapture, but its check must keep it eligible (otherwise the score is -100).
    let board = Board::from_fen("8/1b4p1/7k/8/4r1P1/1R6/8/7K w - - 0 1").unwrap();
    let original = board.clone();
    assert_eq!(search_with_zero_draws(&board, 0).score, 0);
    assert!(board == original);
}

#[test]
fn deep_nonchecking_promotions_are_not_limited_to_recaptures() {
    // The Rh2+ line needs nonchecking d8=Q / dxc8=Q continuations beyond
    // the recapture threshold. Restricting those promotions gives -900 instead.
    let board = Board::from_fen("1qr5/3P4/7k/8/8/8/1p1R4/7K w - - 0 1").unwrap();
    let original = board.clone();
    assert_eq!(search_with_zero_draws(&board, 0).score, -500);
    assert!(board == original);
}

#[test]
fn depth_zero_searches_quiet_checks_and_finds_mate() {
    for fen in ["7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "8/8/8/8/8/6k1/5q2/7K b - - 0 1"] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        for run in [search, search_with_zero_draws] {
            let result = run(&board, 0);
            assert_eq!(result.score, 31_999);
            assert_eq!(result.best_move, None);
            assert!(result.nodes > 1);
            assert!(board == original);
        }
    }
}

#[test]
fn quiescence_includes_en_passant_for_both_sides() {
    for fen in ["4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1", "4k3/8/8/8/5Pp1/8/8/4K3 b - f3 0 1"] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        for run in [search, search_with_zero_draws, search_reference] {
            let result = run(&board, 0);
            assert_eq!(result.score, 100);
            assert_eq!(result.nodes, 1);
        }
        assert!(board == original);
    }
}

#[test]
fn depth_zero_resolves_a_hanging_queen_instead_of_stopping_at_material() {
    // ...exd5 wins the queen, leaving Black with a pawn against the bare king.
    let board = Board::from_fen("6k1/8/4p3/3Q4/8/8/8/6K1 b - - 0 1").unwrap();
    let original = board.clone();
    for run in [search, search_with_zero_draws] {
        let result = run(&board, 0);
        assert_eq!(result.score, 100);
        assert_eq!(result.best_move, None);
        assert_eq!(result.nodes, 1);
        assert!(board == original);
    }
}
