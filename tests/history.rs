use asura::{
    board::Board,
    history::{ContinuationHistory, HistoryTables, NoisyHistory, PawnHistory, QuietHistory},
    movepick::MainMovePicker,
    types::{Color, Move, MoveKind, Piece, PieceType, Square},
};

#[test]
fn gravity_updates_converge_to_bounds_and_distinguish_threat_and_pawn_contexts() {
    let board = Board::starting_position();
    let mv = Move::new(Square::G1, Square::F3, MoveKind::Normal);
    let mut quiet = QuietHistory::default();

    for _ in 0..40 {
        quiet.update(board.all_threats(), false, Color::White, mv, 4_000);
    }
    let high = quiet.get(board.all_threats(), false, Color::White, mv);
    assert!(high > 8_000 && high <= QuietHistory::MAX_HISTORY, "got {high}");
    assert_eq!(quiet.get(board.all_threats(), true, Color::White, mv), 0, "creates_threat bit must be distinct");

    let mut pawn = PawnHistory::default();
    pawn.update(board.pawn_key(), Piece::WhiteKnight, Square::F3, 1_500);
    assert_eq!(pawn.get(board.pawn_key(), Piece::WhiteKnight, Square::F3), 1_500);

    let mut noisy = NoisyHistory::default();
    noisy.update(board.all_threats(), Piece::WhiteKnight, Square::F3, PieceType::Pawn, 2_000);
    assert_eq!(noisy.get(board.all_threats(), Piece::WhiteKnight, Square::F3, PieceType::Pawn), 2_000);
}

#[test]
fn continuation_history_allocates_lazily_and_seeds_from_board_move_history() {
    let mut board = Board::starting_position();
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        let mv = board.generate_all_moves().iter().copied().find(|m| m.to_uci(&board) == uci).unwrap();
        board.make_move(mv);
    }

    let mut cont = ContinuationHistory::default();
    cont.seed_root(&board);

    // At ply 0, distance 1 is Black's b8c6, distance 2 is White's g1f3, distance 4 is White's e2e4.
    cont.update(0, 1, Piece::WhiteBishop, Square::B5, 1_200);
    cont.update(0, 2, Piece::WhiteBishop, Square::B5, 800);
    cont.update(0, 4, Piece::WhiteBishop, Square::B5, 400);

    assert_eq!(cont.get(0, 1, Piece::WhiteBishop, Square::B5), 1_200);
    assert_eq!(cont.get(0, 2, Piece::WhiteBishop, Square::B5), 800);
    assert_eq!(cont.get(0, 4, Piece::WhiteBishop, Square::B5), 400);
    assert_eq!(cont.get(0, 6, Piece::WhiteBishop, Square::B5), 0);

    cont.clear();
    assert_eq!(cont.get(0, 1, Piece::WhiteBishop, Square::B5), 0);
}

#[test]
fn killers_and_countermoves_update_without_duplicating_primary_killer() {
    let mut tables = HistoryTables::default();
    let prev = Move::new(Square::E7, Square::E5, MoveKind::DoublePush);
    let k1 = Move::new(Square::G1, Square::F3, MoveKind::Normal);
    let k2 = Move::new(Square::F1, Square::C4, MoveKind::Normal);

    tables.update_killer_and_countermove(2, prev, k1);
    assert_eq!(tables.killers(2), [k1, Move::NULL]);
    assert_eq!(tables.countermove(prev), k1);

    // Recording the same killer again must not overwrite slot 1 with a duplicate.
    tables.update_killer_and_countermove(2, prev, k1);
    assert_eq!(tables.killers(2), [k1, Move::NULL]);

    tables.update_killer_and_countermove(2, prev, k2);
    assert_eq!(tables.killers(2), [k2, k1]);
}

#[test]
fn move_picker_emits_killers_and_countermove_before_quiets_and_orders_by_history() {
    let board = Board::starting_position();
    let mut tables = HistoryTables::default();
    let prev = Move::new(Square::E7, Square::E5, MoveKind::DoublePush);
    let killer = Move::new(Square::B1, Square::C3, MoveKind::Normal);
    let counter = Move::new(Square::D2, Square::D4, MoveKind::DoublePush);
    let boosted = Move::new(Square::A2, Square::A3, MoveKind::Normal);

    tables.update_killer_and_countermove(1, prev, counter);
    tables.update_killer_and_countermove(1, Move::NULL, killer);
    tables.quiet_history.update(board.all_threats(), false, Color::White, boosted, 8_000);

    let mut picker = MainMovePicker::new(Move::NULL, false, tables.killers(1), tables.countermove(prev));
    let first = picker.next(&board, Some(&tables), false, 1).unwrap();
    let second = picker.next(&board, Some(&tables), false, 1).unwrap();
    let third = picker.next(&board, Some(&tables), false, 1).unwrap();

    // Priority quiets (killer and countermove) must be emitted before generating the rest,
    // and the boosted quiet move must lead the scored quiet stage.
    assert!(
        (first == killer && second == counter) || (first == counter && second == killer),
        "expected killer and countermove first, got {first:?} and {second:?}"
    );
    assert_eq!(third, boosted);
}

#[test]
fn quiet_tt_move_caps_equal_captures_at_three_while_keeping_winning_captures() {
    // White has four equal pawn captures (a4xb5, c4xd5, e4xf5, g4xh5, all SEE == 0)
    // and one winning rook capture (Rxh8, SEE > 0), plus a quiet TT move Kh1-g1.
    let board = Board::from_fen("k6r/8/pppp4/pppp4/PPPP4/8/8/6KR w - - 0 1").unwrap();
    let tt_quiet = board.generate_all_moves().iter().copied().find(|m| m.to_uci(&board) == "g1f1").unwrap();
    let tables = HistoryTables::default();

    let mut picker = MainMovePicker::new(tt_quiet, false, [Move::NULL; 2], Move::NULL);
    let first = picker.next(&board, Some(&tables), false, 1).unwrap();
    assert_eq!(first, tt_quiet);

    // Next 3 moves in GoodNoisy are the winning rook capture + 2 equal pawn captures (noisy_count reaches 3).
    // The 4th capture is an equal trade (SEE == 0), which fails the +1 threshold once noisy_count > 2
    // and is deferred to BadNoisy behind the quiet moves.
    let c1 = picker.next(&board, Some(&tables), false, 1).unwrap();
    let c2 = picker.next(&board, Some(&tables), false, 1).unwrap();
    let c3 = picker.next(&board, Some(&tables), false, 1).unwrap();
    assert!(c1.is_capture() && c2.is_capture() && c3.is_capture());
    assert_eq!(c1.to_uci(&board), "h1h8");

    let next_after_three_captures = picker.next(&board, Some(&tables), false, 1).unwrap();
    assert!(
        next_after_three_captures.is_quiet(),
        "4th equal capture should be deferred behind quiets when tt_move is quiet, got {}",
        next_after_three_captures.to_uci(&board)
    );
}

#[test]
fn cold_quiet_deferral_defers_threatened_destinations_not_escapes() {
    // White knight on c3 is attacked by a Black pawn on b4.
    // Moving c3-a4 escapes to a safe square; moving c3-d5 lands on a square
    // attacked by a Black pawn on e6. Even with cold history on both moves,
    // only c3-d5 (destination attacked by lesser piece) should be deferred behind
    // neutral king moves; the escape c3-a4 must not be treated as a threatened destination.
    let board = Board::from_fen("7k/8/4p3/8/1p6/2N5/8/7K w - - 0 1").unwrap();
    let escape = board.generate_all_moves().iter().copied().find(|m| m.to_uci(&board) == "c3a4").unwrap();
    let into_pawn = board.generate_all_moves().iter().copied().find(|m| m.to_uci(&board) == "c3d5").unwrap();

    let mut tables = HistoryTables::default();
    // Give escape a lower score (-7,000) than into_pawn (-2,000) so take_best()
    // pops into_pawn first; deferral must push into_pawn behind escape while
    // emitting escape directly without deferring it.
    tables.quiet_history.update(board.all_threats(), false, Color::White, escape, -10_000);
    tables.quiet_history.update(board.all_threats(), false, Color::White, into_pawn, -1_000);

    let mut picker = MainMovePicker::new(Move::NULL, false, [Move::NULL; 2], Move::NULL);
    let mut order = Vec::new();
    while let Some(mv) = picker.next(&board, Some(&tables), false, 1) {
        order.push(mv);
    }

    let escape_pos = order.iter().position(|&m| m == escape).unwrap();
    let into_pawn_pos = order.iter().position(|&m| m == into_pawn).unwrap();
    assert!(
        escape_pos < into_pawn_pos,
        "escape move ({escape_pos}) must be ordered ahead of deferred move into pawn threat ({into_pawn_pos})"
    );
}

#[test]
fn correction_history_updates_structural_and_continuation_tables_and_converges() {
    let mut board = Board::starting_position();
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        let mv = board.generate_all_moves().iter().copied().find(|m| m.to_uci(&board) == uci).unwrap();
        board.make_move(mv);
    }

    let mut tables = HistoryTables::default();
    tables.prepare_search(&board);
    assert_eq!(tables.eval_correction(&board, 0), 0);

    // Repeated positive updates must converge toward 5 * CORRHIST_MAX / CORRHIST_SCALE_DIV = 160.
    for _ in 0..50 {
        tables.update_correction_history(&board, 0, 256);
    }
    let corr = tables.eval_correction(&board, 0);
    assert!(corr > 120 && corr <= 160, "expected converged correction in (120, 160], got {corr}");

    tables.clear();
    assert_eq!(tables.eval_correction(&board, 0), 0);
}
