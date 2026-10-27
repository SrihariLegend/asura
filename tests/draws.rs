use asura::{
    board::Board,
    search::{search, search_with_zero_draws},
    tools::perft,
};

#[test]
fn material_draw_policy_matches_piece_counts_and_minor_combinations() {
    for (name, placement, expected) in [
        ("bare kings", "7k/8/8/8/8/8/8/7K", true),
        ("white knight", "7k/8/8/8/8/8/N7/7K", true),
        ("white bishop", "7k/8/8/8/8/8/B7/7K", true),
        ("black knight", "7k/8/n7/8/8/8/8/7K", true),
        ("black bishop", "7k/8/b7/8/8/8/8/7K", true),
        ("white two knights", "7k/8/8/8/8/8/NN6/7K", true),
        ("black two knights", "7k/8/nn6/8/8/8/8/7K", true),
        ("white bishop and knight", "7k/8/8/8/8/8/BN6/7K", false),
        ("black bishop and knight", "7k/8/bn6/8/8/8/8/7K", false),
        ("white light-squared bishops", "7k/8/8/8/8/8/B1B5/7K", true),
        ("white dark-squared bishops", "7k/8/8/8/8/B7/8/2B4K", true),
        ("white opposite-complex bishops", "7k/8/8/8/8/8/B7/2B4K", false),
        ("black same-complex bishops", "7k/8/b3b3/8/8/8/8/7K", true),
        ("black opposite-complex bishops", "7k/8/b7/2b5/8/8/8/7K", false),
        ("knight each, kings in corners", "7k/8/1n6/8/8/8/N7/7K", true),
        ("same-complex bishop each", "7k/8/b7/8/8/8/B7/7K", true),
        ("opposite-complex bishop each", "7k/8/1b6/8/8/8/B7/7K", true),
        ("white bishop versus knight", "7k/8/1n6/8/8/8/B7/7K", true),
        ("white knight versus bishop", "7k/8/1b6/8/8/8/N7/7K", true),
        ("white pawn", "7k/8/8/8/8/8/P7/7K", false),
        ("black pawn", "7k/8/p7/8/8/8/8/7K", false),
        ("white rook", "7k/8/8/8/8/8/R7/7K", false),
        ("black rook", "7k/8/r7/8/8/8/8/7K", false),
        ("white queen", "7k/8/8/8/8/8/Q7/7K", false),
        ("black queen", "7k/8/q7/8/8/8/8/7K", false),
        ("three minors", "7k/8/8/8/8/8/NNN5/7K", false),
        ("one minor versus two", "7k/8/nn6/8/8/8/N7/7K", false),
        ("three same-complex bishops", "7k/8/8/8/8/8/B1B1B3/7K", false),
        ("four minors", "7k/8/8/8/8/8/NNNN4/7K", false),
    ] {
        for side in ["w", "b"] {
            let fen = format!("{placement} {side} - - 0 1");
            let board = Board::from_fen(&fen).unwrap_or_else(|error| panic!("{name}: {error}: {fen}"));
            let original = board.clone();
            assert_eq!(board.draw_by_material(), expected, "{name}, {side} to move");
            assert!(board == original);
        }
    }
}

#[test]
fn material_draws_use_the_configured_score_and_keep_a_legal_root_move() {
    for fen in [
        "7k/8/8/8/8/8/8/7K w - - 0 1",
        "7k/8/8/8/8/8/N7/7K w - - 0 1",
        "7k/8/8/8/8/8/NN6/7K w - - 0 1",
        "7k/8/nn6/8/8/8/8/7K b - - 0 1",
        "7k/8/8/8/8/B7/8/2B4K w - - 0 1",
        "7k/8/1b6/8/8/8/N7/7K b - - 0 1",
        "7k/5K2/6N1/8/8/8/N7/8 b - - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        assert!(board.has_legal_moves());
        for depth in [0, 1, 4, 255] {
            let result = search(&board, depth);
            assert_eq!(result.score, -2, "{fen}, depth {depth}");
            assert_eq!(result.nodes, 0);
            if depth == 0 {
                assert_eq!(result.best_move, None);
            } else {
                assert!(board.is_legal(result.best_move.expect("a material draw still has legal moves")));
            }

            let exact = search_with_zero_draws(&board, depth);
            assert_eq!(exact.score, 0);
            assert_eq!(exact.nodes, 0);
            assert_eq!(exact.best_move, result.best_move);
            assert!(board == original);
        }
    }
}

#[test]
fn checkmate_takes_precedence_over_the_material_policy() {
    for fen in [
        "7k/5K2/5NN1/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/5nn1/5k2/7K w - - 0 1",
        "7k/5K1n/6N1/8/8/8/8/8 b - - 0 1",
        "7k/5KBb/8/8/8/8/8/8 b - - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        assert!(board.draw_by_material());
        assert!(board.in_check() && !board.has_legal_moves());
        for run in [search, search_with_zero_draws] {
            for depth in [0, 1, 3] {
                let result = run(&board, depth);
                assert_eq!(result.score, -32_000);
                assert_eq!(result.best_move, None);
                assert_eq!(result.nodes, 0);
                assert!(board == original);
            }
        }
    }
}

#[test]
fn a_capture_delivering_mate_overrides_the_new_material_draw() {
    for (fen, best_move) in
        [("7k/5K2/5Nr1/4N3/8/8/8/8 w - - 0 1", "e5g6"), ("8/8/8/8/4n3/5nR1/5k2/7K b - - 0 1", "e4g3")]
    {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        assert!(!board.draw_by_material());
        for run in [search, search_with_zero_draws] {
            for depth in [1, 2, 3] {
                let result = run(&board, depth);
                let mv = result.best_move.unwrap();
                assert_eq!(mv.to_uci(&board), best_move);
                assert_eq!(result.score, 31_999);
                assert!(board == original);

                let mut child = board.clone();
                child.make_move(mv);
                assert!(child.draw_by_material());
                assert!(child.in_check() && !child.has_legal_moves());
            }
        }
    }
}

#[test]
fn search_does_not_value_a_capture_into_two_knights_as_a_material_win() {
    let board = Board::from_fen("7k/8/8/8/8/2r5/NN6/7K w - - 0 1").unwrap();
    let original = board.clone();
    let capture = board.generate_all_moves().iter().copied().find(|mv| mv.to_uci(&board) == "a2c3").unwrap();
    let mut child = board.clone();
    child.make_move(capture);
    assert!(child.draw_by_material());

    for run in [search, search_with_zero_draws] {
        let result = run(&board, 1);
        assert_eq!(result.score, 140);
        let mv = result.best_move.unwrap();
        assert!(board.is_legal(mv));
        assert_ne!(mv, capture, "capturing the rook would enter an immediate material draw");
        assert!(board == original);
    }
}

#[test]
fn fifty_move_rule_requires_clock_100_and_yields_to_checkmate() {
    let before = Board::from_fen("7k/8/8/8/8/8/Q7/7K w - - 99 1").unwrap();
    assert!(!before.draw_by_fifty_move_rule());
    assert!(!before.is_draw(0));

    let draw = Board::from_fen("7k/8/8/8/8/8/Q7/7K w - - 100 1").unwrap();
    assert!(draw.draw_by_fifty_move_rule());
    assert!(draw.is_draw(0));

    let checked_with_escape = Board::from_fen("7k/8/8/8/8/8/6q1/7K w - - 100 1").unwrap();
    assert!(checked_with_escape.in_check());
    assert!(checked_with_escape.has_legal_moves());
    assert!(checked_with_escape.draw_by_fifty_move_rule());
    assert!(checked_with_escape.is_draw(0));

    let checkmated = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1").unwrap();
    assert!(checkmated.in_check() && !checkmated.has_legal_moves());
    assert!(!checkmated.draw_by_fifty_move_rule());
    assert!(!checkmated.is_draw(0));
    assert_eq!(search(&checkmated, 1).score, -32_000);
    assert_eq!(search_with_zero_draws(&checkmated, 1).score, -32_000);

    // Delivering checkmate on the 100th halfmove must still score as mate, not a 50-move draw.
    let mate_on_100th = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 99 1").unwrap();
    let result = search_with_zero_draws(&mate_on_100th, 1);
    assert_eq!(result.score, 31_999);

    // When no mate exists on the 100th halfmove, a quiet move reaches the 50-move draw,
    // so White must capture or push a pawn to reset the clock and keep its winning advantage.
    let reset_clock = Board::from_fen("7k/8/8/8/8/8/P1Q5/7K w - - 99 1").unwrap();
    let result = search_with_zero_draws(&reset_clock, 1);
    assert_eq!(result.score, 1000);
    let best = result.best_move.unwrap().to_uci(&reset_clock);
    assert!(best == "a2a3" || best == "a2a4", "expected pawn push to reset 50-move clock, got {best}");
}

#[test]
fn repetition_and_upcoming_repetition_track_reversible_cycles() {
    let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K1N1 w Q - 0 1").unwrap();
    let start = board.clone();

    let play = |board: &mut Board, uci: &str| {
        let mv = board.generate_all_moves().iter().copied().find(|m| m.to_uci(board) == uci).unwrap();
        board.make_move(mv);
        mv
    };

    let m1 = play(&mut board, "g1f3");
    let m2 = play(&mut board, "e8d8");
    let m3 = play(&mut board, "f3g1");

    // After 3 reversible plies, Black can play d8e8 to repeat the root position:
    // At ply <= 3 (root was at or before the target and had not repeated), upcoming_repetition is false;
    // at ply > 3 (target was strictly inside the search tree), upcoming_repetition is true.
    assert!(!board.upcoming_repetition(3));
    assert!(board.upcoming_repetition(4));
    assert!(!board.has_repeated());

    let m4 = play(&mut board, "d8e8");
    // Twofold repetition: draw strictly after the root (ply > 4), not at the root (ply = 0..=4).
    assert!(!board.draw_by_repetition(0));
    assert!(!board.draw_by_repetition(4));
    assert!(board.draw_by_repetition(5));
    assert!(board.is_draw(5));
    assert!(board.has_repeated());

    let m5 = play(&mut board, "g1f3");
    let m6 = play(&mut board, "e8d8");
    // Before m7, White is up +820. Playing f3g1 would allow Black an immediate
    // upcoming_repetition draw at ply 1 (since the target position after m4 already
    // had repetition != 0), so even a depth-1 search avoids f3g1 and keeps +820.
    let avoid_upcoming = search_with_zero_draws(&board, 1);
    assert_ne!(avoid_upcoming.best_move.unwrap().to_uci(&board), "f3g1");
    assert_eq!(avoid_upcoming.score, 820);

    let m7 = play(&mut board, "f3g1");
    // Now the target position (after m4) already had repetition != 0, so upcoming_repetition
    // triggers even at ply = 1.
    assert!(board.upcoming_repetition(1));

    let m8 = play(&mut board, "d8e8");
    // Threefold repetition sets negative repetition distance, which is a draw at every non-negative ply.
    assert!(board.draw_by_repetition(0));
    assert!(board.is_draw(0));

    // Losing side chooses the third repetition to save a draw instead of staying down 820 cp.
    board.undo_move(m8);
    let claim_rep = search_with_zero_draws(&board, 1);
    assert_eq!(claim_rep.best_move.unwrap().to_uci(&board), "d8e8");
    assert_eq!(claim_rep.score, 0);

    // Null move resets plies_from_null and restores cleanly on undo.
    board.make_null_move();
    assert!(!board.upcoming_repetition(10));
    board.undo_null_move();

    for mv in [m7, m6, m5, m4, m3, m2, m1] {
        board.undo_move(mv);
    }
    assert!(board == start);
}

#[test]
fn perft_keeps_traversing_material_draws_and_restores_the_board() {
    let mut board = Board::from_fen("7k/8/8/8/8/8/8/7K w - - 0 1").unwrap();
    let original = board.clone();
    assert!(board.draw_by_material());
    for (depth, nodes) in [1, 3, 9, 54].into_iter().enumerate() {
        assert_eq!(perft(&mut board, depth), nodes);
        assert!(board == original);
    }
}
