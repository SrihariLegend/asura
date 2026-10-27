use asura::{board::Board, tools::perft, types::Move};

fn legal_move(board: &Board, uci: &str) -> Move {
    board
        .generate_all_moves()
        .iter()
        .copied()
        .find(|mv| mv.to_uci(board) == uci)
        .unwrap_or_else(|| panic!("missing legal move {uci}\n{board}"))
}

fn assert_round_trip(fen: &str, uci: &str, expected: &str, frc: bool) {
    let mut board = Board::from_fen(fen).unwrap();
    board.set_frc(frc);
    let original = board.clone();
    let mv = legal_move(&board, uci);

    board.make_move(mv);
    assert_eq!(board.to_fen(), expected, "after {uci}");
    board.undo_move(mv);
    assert!(board == original, "not all board state was restored after {uci}");
}

#[test]
fn special_moves_have_the_expected_effect_and_restore_all_state() {
    for (fen, uci, expected) in [
        ("7k/8/8/8/3p4/8/4P3/4K3 w - - 17 4", "e2e4", "7k/8/8/8/3pP3/8/8/4K3 b - e3 0 4"),
        ("4k3/4p3/8/3P4/8/8/8/7K b - - 17 4", "e7e5", "4k3/8/8/3Pp3/8/8/8/7K w - e6 0 5"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 9 1", "e4d5", "4k3/8/8/3P4/8/8/8/4K3 b - - 0 1"),
        ("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1", "g5f6", "4k3/8/5P2/8/8/8/8/4K3 b - - 0 1"),
        ("4k3/8/8/8/5Pp1/8/8/4K3 b - f3 0 1", "g4f3", "4k3/8/8/8/8/5p2/8/4K3 w - - 0 2"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", "r3k2r/8/8/8/8/8/8/2KR3R b kq - 1 1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", "r4rk1/8/8/8/8/8/8/R3K2R w KQ - 1 2"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "2kr3r/8/8/8/8/8/8/R3K2R w KQ - 1 2"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 12 7", "a1a8", "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 7"),
    ] {
        assert_round_trip(fen, uci, expected, false);
    }
}

#[test]
fn every_promotion_and_promotion_capture_restores_all_state() {
    for (suffix, piece) in [('q', 'Q'), ('r', 'R'), ('b', 'B'), ('n', 'N')] {
        assert_round_trip(
            "4k3/P7/8/8/8/8/8/4K3 w - - 23 9",
            &format!("a7a8{suffix}"),
            &format!("{piece}3k3/8/8/8/8/8/8/4K3 b - - 0 9"),
            false,
        );
        assert_round_trip(
            "1r2k3/P7/8/8/8/8/8/4K3 w - - 23 9",
            &format!("a7b8{suffix}"),
            &format!("1{piece}2k3/8/8/8/8/8/8/4K3 b - - 0 9"),
            false,
        );
        assert_round_trip(
            "4k3/8/8/8/8/8/p7/4K3 b - - 23 9",
            &format!("a2a1{suffix}"),
            &format!("4k3/8/8/8/8/8/8/{suffix}3K3 w - - 0 10"),
            false,
        );
        assert_round_trip(
            "4k3/8/8/8/8/8/p7/1R2K3 b - - 23 9",
            &format!("a2b1{suffix}"),
            &format!("4k3/8/8/8/8/8/8/1{suffix}2K3 w - - 0 10"),
            false,
        );
    }
}

#[test]
fn all_promotions_belong_to_the_noisy_stage() {
    for fen in ["1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1"] {
        let board = Board::from_fen(fen).unwrap();
        let mut noisy = asura::types::MoveList::new();
        let mut quiet = asura::types::MoveList::new();
        board.append_noisy_moves(&mut noisy);
        board.append_quiet_moves(&mut quiet);
        assert_eq!(noisy.len(), 8, "four push promotions and four capture promotions");
        assert!(noisy.iter().all(|mv| mv.is_promotion() && mv.is_noisy()));
        assert!(quiet.iter().all(|mv| mv.is_quiet() && !mv.is_promotion()));
        assert_eq!(noisy.len() + quiet.len(), board.generate_all_moves().len());
    }
}

#[test]
fn chess960_castling_restores_all_state() {
    for (fen, uci, expected) in [
        ("4k3/8/8/8/8/8/8/6KR w H - 0 1", "g1h1", "4k3/8/8/8/8/8/8/5RK1 b - - 1 1"),
        ("4k3/8/8/8/8/8/8/4KR2 w F - 0 1", "e1f1", "4k3/8/8/8/8/8/8/5RK1 b - - 1 1"),
        ("4k3/8/8/8/8/8/8/5KR1 w G - 0 1", "f1g1", "4k3/8/8/8/8/8/8/5RK1 b - - 1 1"),
        ("4k3/8/8/8/8/8/8/R1K5 w A - 0 1", "c1a1", "4k3/8/8/8/8/8/8/2KR4 b - - 1 1"),
    ] {
        assert_round_trip(fen, uci, expected, true);
    }
}

#[test]
fn check_detection_handles_discoveries_promotions_en_passant_and_castling() {
    for (fen, uci, frc, expected) in [
        ("6k1/8/8/8/8/8/2Q5/6K1 w - - 0 1", "c2c8", false, true),
        ("6k1/8/8/8/8/8/2Q5/6K1 w - - 0 1", "c2b2", false, false),
        ("k7/8/8/8/8/B7/8/R6K w - - 0 1", "a3b4", false, true),
        ("8/8/8/R2pP2k/8/8/8/7K w - d6 0 1", "e5d6", false, true),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q", false, true),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n", false, false),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", "a7b8q", false, true),
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", false, true),
        ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", "e1c1", false, true),
        ("5k2/8/8/8/8/8/8/5KR1 w G - 0 1", "f1g1", true, true),
        ("5k2/8/8/8/8/8/8/6KR w H - 0 1", "g1h1", true, true),
        ("8/8/8/8/8/8/8/k3KR2 w F - 0 1", "e1f1", true, true),
    ] {
        let mut board = Board::from_fen(fen).unwrap();
        board.set_frc(frc);
        let original = board.clone();
        let mv = legal_move(&board, uci);
        assert_eq!(board.gives_check(mv), expected, "{fen}: {uci}");
        assert!(board == original, "check detection must not change any saved state");
        board.make_move(mv);
        assert_eq!(board.in_check(), expected, "{fen}: {uci}");
        board.undo_move(mv);
        assert!(board == original);
    }
}

#[test]
fn quiet_checks_match_legal_move_playout_without_duplicates_or_state_changes() {
    for (fen, frc) in [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false),
        ("k7/8/8/8/8/B7/8/R6K w - - 0 1", false),
        ("6k1/8/8/8/8/8/2Q5/6K1 w - - 0 1", false),
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", false),
        ("5k2/8/8/8/8/8/8/5KR1 w G - 0 1", true),
        ("8/8/8/8/8/8/8/k3KR2 w F - 0 1", true),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", false),
        ("8/8/8/R2pP2k/8/8/8/7K w - d6 0 1", false),
    ] {
        let mut board = Board::from_fen(fen).unwrap();
        board.set_frc(frc);
        let mut seed = 23_u64;
        for _ in 0..24 {
            let original = board.clone();
            let moves = board.generate_all_moves();
            let mut expected = Vec::new();
            for &mv in moves.iter() {
                let predicted = board.gives_check(mv);
                board.make_move(mv);
                assert_eq!(predicted, board.in_check(), "{fen}: {mv:?}");
                if mv.is_quiet() && board.in_check() {
                    expected.push(mv);
                }
                board.undo_move(mv);
            }
            let mut actual = asura::types::MoveList::new();
            board.append_quiet_checks(&mut actual);
            assert_eq!(actual.iter().copied().collect::<Vec<_>>(), expected);
            let unique: std::collections::HashSet<_> = actual.iter().collect();
            assert_eq!(unique.len(), actual.len());
            assert!(board == original);
            if moves.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            board.make_move(moves[(seed as usize) % moves.len()]);
        }
    }
}

#[test]
fn static_exchanges_handle_xrays_pins_kings_en_passant_and_promotions() {
    for (fen, uci, gain) in [
        ("6k1/8/4p3/3Q4/8/8/8/6K1 b - - 0 1", "e6d5", 900),
        ("6k1/8/4p3/3p4/8/8/8/3Q2K1 w - - 0 1", "d1d5", -800),
        ("3q2k1/8/8/3p4/8/3R4/8/3R2K1 w - - 0 1", "d3d5", 100),
        ("4k3/8/4b3/3p4/2B5/8/8/4R1K1 w - - 0 1", "c4d5", 100),
        ("8/8/4k3/3p4/2B5/8/8/3R2K1 w - - 0 1", "c4d5", 100),
        ("6k1/8/8/5pP1/8/8/8/6K1 w - f6 0 1", "g5f6", 100),
        ("5rk1/8/8/5pP1/8/8/8/6K1 w - f6 0 1", "g5f6", 0),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q", 800),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n", 220),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", "a7b8q", 1300),
        ("k5rQ/6P1/8/8/8/8/8/K7 b - - 0 1", "g8h8", -400),
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", 0),
    ] {
        let board = Board::from_fen(fen).unwrap();
        let original = board.clone();
        let mv = legal_move(&board, uci);
        assert!(board.see_ge(mv, gain), "{fen}: {uci} should gain at least {gain}");
        assert!(!board.see_ge(mv, gain + 1), "{fen}: {uci} should not gain more than {gain}");
        assert!(board.see_ge(mv, i32::MIN));
        assert!(!board.see_ge(mv, i32::MAX));
        assert!(board == original);
    }
}

fn exchange_value(piece: asura::types::PieceType) -> i32 {
    use asura::types::PieceType;
    match piece {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King | PieceType::None => 0,
    }
}

fn immediate_gain(board: &Board, mv: Move) -> i32 {
    let capture = if mv.is_capture() { exchange_value(board.type_on(mv.capture_sq())) } else { 0 };
    capture + if mv.is_promotion() { exchange_value(mv.promo_piece_type()) - 100 } else { 0 }
}

// Independently play the least-valuable legal recapture on a real board.
// Do not reuse SEE's occupancy simulation or attacker selection.
fn recapture_gain(board: &Board, square: asura::types::Square) -> i32 {
    let moves = board.generate_all_moves();
    let next = moves.iter().copied().filter(|mv| mv.is_capture() && mv.to() == square).min_by_key(|mv| {
        let piece = board.type_on(mv.from());
        let cost = if piece == asura::types::PieceType::King { 20_000 } else { exchange_value(piece) };
        let promotion = if mv.is_promotion() { -exchange_value(mv.promo_piece_type()) } else { 0 };
        (cost, mv.from() as u8, promotion)
    });
    let Some(mv) = next else { return 0 };
    let gain = immediate_gain(board, mv);
    let mut child = board.clone();
    child.make_move(mv);
    (gain - recapture_gain(&child, square)).max(0)
}

#[test]
fn exchange_estimates_match_independent_legal_recapture_playouts() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
        "8/8/8/R2pP2k/8/8/8/7K w - d6 0 1",
        "k5rQ/6P1/8/8/8/8/8/K7 b - - 0 1",
    ] {
        let mut board = Board::from_fen(fen).unwrap();
        let mut seed = 193_u64;
        for _ in 0..80 {
            let original = board.clone();
            let moves = board.generate_all_moves();
            for &mv in moves.iter().filter(|mv| mv.is_capture() || mv.is_promotion()) {
                let mut child = board.clone();
                child.make_move(mv);
                let expected = immediate_gain(&board, mv) - recapture_gain(&child, mv.to());
                assert!(board.see_ge(mv, expected), "{}: {mv:?}, expected {expected}", board.to_fen());
                assert!(!board.see_ge(mv, expected + 1), "{}: {mv:?}, expected {expected}", board.to_fen());
                assert!(board == original);
            }
            if moves.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            board.make_move(moves[(seed as usize) % moves.len()]);
        }
    }
}

#[test]
fn make_undo_and_perft_preserve_existing_history() {
    let mut board = Board::starting_position();
    let start = board.clone();
    let mut history = Vec::new();

    for uci in ["e2e4", "d7d5", "e4d5", "d8d5"] {
        let mv = legal_move(&board, uci);
        board.make_move(mv);
        history.push(mv);
    }

    let original = board.clone();
    for &mv in board.generate_all_moves().iter() {
        board.make_move(mv);
        board.undo_move(mv);
        assert!(board == original);
    }
    perft(&mut board, 3);
    assert!(board == original);

    for mv in history.into_iter().rev() {
        board.undo_move(mv);
    }
    assert!(board == start, "the original history could not be undone after traversal");
}
