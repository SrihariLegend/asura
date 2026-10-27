use super::Board;
use crate::{
    tools::perft,
    types::{Color, Move, Piece, PieceType, Square},
};

macro_rules! assert_perft {
    ($($name:ident: $fen:tt, [$($nodes:expr),*],)*) => {$(
        #[test]
        fn $name() {
            let mut board = Board::from_fen($fen).unwrap();
            let original = board.clone();
            for (depth, &nodes) in [$($nodes),*].iter().enumerate() {
                assert_eq!(perft(&mut board, depth + 1), nodes, "depth {}", depth + 1);
                assert!(board == original, "perft changed board state at depth {}", depth + 1);
            }
        }
    )*};
}

// Test cases from https://www.chessprogramming.org/Perft_Results
assert_perft!(
    starting_position: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [20, 400, 8902, 197281, 4865609],
    kiwipete: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862, 4085603],
    position_3: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624, 11030083],
    position_4: "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467, 422333, 15833292],
    position_5: "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379, 2103487],
    position_6: "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890, 3894594],
);

// Test cases from https://www.chessprogramming.org/Chess960_Perft_Results
assert_perft!(
    chess960_1: "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", [21, 528, 12189, 326672, 8146062],
    chess960_2: "1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9", [28, 1120, 31058, 1171749, 34030312],
    chess960_3: "bqnr1kr1/pppppp1p/6p1/5n2/4B3/3N2PP/PbPPPP2/BQNR1KR1 w GDgd - 2 9", [31, 1132, 36559, 1261476, 43256823],
);

fn frc_board(fen: &str) -> Board {
    let mut board = Board::from_fen(fen).unwrap();
    board.set_frc(true); // Match UCI_Chess960 mode, in which to_fen emits Shredder castling.
    board
}

#[test]
fn to_fen_emits_shredder_castling_for_frc() {
    // FRC positions whose rooks sit on non-standard files, so to_fen must emit
    // Shredder castling (rook-file letters) rather than KQkq. Fixtures reuse the
    // Chess960 perft positions above.
    for fen in [
        "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        "bqnr1kr1/pppppp1p/6p1/5n2/4B3/3N2PP/PbPPPP2/BQNR1KR1 w GDgd - 2 9",
    ] {
        assert_eq!(frc_board(fen).to_fen(), fen);
    }
}

fn assert_hash_consistent(board: &Board) {
    let mut recomputed = board.clone();
    recomputed.update_hash_keys();
    assert_eq!(board.hash(), recomputed.hash(), "incremental hash diverged:\n{board}");
    assert_eq!(board.pawn_key(), recomputed.pawn_key());
    for color in [Color::White, Color::Black] {
        assert_eq!(board.non_pawn_key(color), recomputed.non_pawn_key(color));
    }
}

fn hash_perft(board: &mut Board, depth: usize) {
    assert_hash_consistent(board);
    if depth == 0 {
        return;
    }

    let original = board.clone();
    for &mv in board.generate_all_moves().iter() {
        assert!(board.is_legal(mv), "illegal generated move: {mv:?}\n{board}");
        board.make_move(mv);
        hash_perft(board, depth - 1);
        board.undo_move(mv);
        assert!(*board == original, "undo failed to restore all state for {mv:?}");
        assert_hash_consistent(board);
    }
}

#[test]
fn incremental_hash_and_undo_match_recomputation() {
    for fen in [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        "4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1",
    ] {
        let mut board = Board::from_fen(fen).unwrap();
        hash_perft(&mut board, 3);
    }
}

#[test]
fn from_fen_accepts_missing_optional_fields() {
    let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w").unwrap();
    assert_eq!(board.to_fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 1");

    assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w").is_ok());
    assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w").is_ok());

    assert!(Board::from_fen("").is_err());
    assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3").is_err());
}

#[test]
fn halfmove_clock_saturates_instead_of_overflowing() {
    let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 200 1").unwrap();
    assert_eq!(board.fiftymove_clock(), 200);
    let _ = board.hash();

    let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 255 1").unwrap();
    assert_eq!(board.fiftymove_clock(), 255);

    let mv = board.generate_all_moves()[0];
    board.make_move(mv);
    assert_eq!(board.fiftymove_clock(), 255);
    assert_hash_consistent(&board);
}

#[test]
fn depth_zero_and_terminal_positions() {
    for fen in ["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"] {
        let mut board = Board::from_fen(fen).unwrap();
        assert_eq!(perft(&mut board, 0), 1);
        assert_eq!(perft(&mut board, 1), 0);
        assert_eq!(perft(&mut board, 2), 0);
        assert_eq!(board.to_fen(), fen);
    }
    assert_eq!(perft(&mut Board::default(), 0), 1);
}

#[test]
fn rejects_malformed_fen_without_panicking() {
    for fen in [
        "",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "4k3/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K4 w - - 0 1",
        "4k3/8/8/8/8/8/8/44K w - - 0 1",
        "4k3/8/8/8/8/8/8/4X3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3KK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "4k3/8/8/8/8/8/8/4K2R w KK - 0 1",
        "4k3/8/8/8/8/8/8/4K2R w HZ - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - a4 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - a6 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - z9 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - bad 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 9999999999999999999999999",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
        "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",
        "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    ] {
        assert!(Board::from_fen(fen).is_err(), "accepted invalid FEN: {fen}");
    }
}

#[test]
fn pinned_en_passant_is_not_generated() {
    let board = Board::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1").unwrap();
    assert_eq!(board.en_passant(), Square::None);
    assert!(!board.generate_all_moves().iter().any(|mv| mv.is_en_passant()));
}

#[test]
fn all_underpromotions_are_generated() {
    let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let moves = board.generate_all_moves();
    let promotions: Vec<_> = moves.iter().filter(|mv| mv.from() == Square::A7).collect();
    assert_eq!(promotions.len(), 4);
    for piece in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        assert!(promotions.iter().any(|mv| mv.promo_piece_type() == piece));
    }
}

#[test]
fn chess960_castling_handles_stationary_and_swapping_pieces() {
    for (fen, uci, king_to, rook_to) in [
        ("4k3/8/8/8/8/8/8/6KR w H - 0 1", "g1h1", Square::G1, Square::F1),
        ("4k3/8/8/8/8/8/8/4KR2 w F - 0 1", "e1f1", Square::G1, Square::F1),
        ("4k3/8/8/8/8/8/8/5KR1 w G - 0 1", "f1g1", Square::G1, Square::F1),
        ("4k3/8/8/8/8/8/8/R1K5 w A - 0 1", "c1a1", Square::C1, Square::D1),
    ] {
        let mut board = frc_board(fen);
        let hash = board.hash();
        let mv = board
            .generate_all_moves()
            .iter()
            .copied()
            .find(|mv| mv.is_castling() && mv.to_uci(&board) == uci)
            .expect("missing Chess960 castling move");
        assert!(board.is_legal(mv));
        board.make_move(mv);
        assert_eq!(board.piece_on(king_to), Piece::WhiteKing);
        assert_eq!(board.piece_on(rook_to), Piece::WhiteRook);
        assert_eq!(board.captured_piece(), Piece::None);
        assert_hash_consistent(&board);
        board.undo_move(mv);
        assert_eq!(board.to_fen(), fen);
        assert_eq!(board.hash(), hash);
    }
}

#[test]
fn shredder_castling_uses_the_named_rook() {
    let board = frc_board("4k3/8/8/8/8/8/8/4KRR1 w F - 0 1");
    assert_eq!(board.get_castling_rook(Square::G1), (Square::F1, Square::F1));
    assert!(!board.generate_all_moves().iter().any(|mv| mv.is_castling()));
}

#[test]
fn null_move_has_uci_null_notation() {
    assert_eq!(Move::NULL.to_uci(&Board::starting_position()), "0000");
}
