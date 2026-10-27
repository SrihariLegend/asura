use asura::{
    board::Board,
    search::search_reference,
    types::{Color, Move, Piece, PieceType, Square},
};

struct MinimaxResult {
    white_score: i32,
    best_moves: Vec<Move>,
    nodes: u64,
}

// Enumerate minors and square parity independently of the bitboard predicate.
fn material_draw(board: &Board) -> bool {
    let mut minors = Vec::new();
    for index in 0..64 {
        let piece = board.piece_on(Square::new(index));
        match piece {
            Piece::None | Piece::WhiteKing | Piece::BlackKing => (),
            Piece::WhiteKnight | Piece::BlackKnight | Piece::WhiteBishop | Piece::BlackBishop => {
                minors.push((piece, (index / 8 + index % 8) % 2));
            }
            _ => return false,
        }
    }
    match minors.as_slice() {
        [] | [_] => true,
        [(a, a_colour), (b, b_colour)] => {
            a.color() != b.color()
                || match (a.piece_type(), b.piece_type()) {
                    (PieceType::Knight, PieceType::Knight) => true,
                    (PieceType::Bishop, PieceType::Bishop) => a_colour == b_colour,
                    _ => false,
                }
        }
        _ => false,
    }
}

fn material(board: &Board) -> i32 {
    (0..64)
        .map(|index| match board.piece_on(Square::new(index)) {
            Piece::WhitePawn => 100,
            Piece::WhiteKnight => 320,
            Piece::WhiteBishop => 330,
            Piece::WhiteRook => 500,
            Piece::WhiteQueen => 900,
            Piece::BlackPawn => -100,
            Piece::BlackKnight => -320,
            Piece::BlackBishop => -330,
            Piece::BlackRook => -500,
            Piece::BlackQueen => -900,
            Piece::WhiteKing | Piece::BlackKing | Piece::None => 0,
        })
        .sum::<i32>()
        .clamp(-30_000, 30_000)
}

// Deliberately use mailbox counting, white-centric max/min, and clone-per-move.
// Neither the production evaluator nor its negamax/make-undo loop is reused.
fn minimax(board: &Board, depth: u8, ply: u16) -> MinimaxResult {
    if depth == 0 && ply > 0 {
        return quiescent_minimax(board, 0, ply);
    }
    let moves = board.generate_all_moves();
    let maximizing = board.side_to_move() == Color::White;
    if moves.is_empty() {
        let white_score = if !board.in_check() {
            0
        } else if maximizing {
            -32_000 + i32::from(ply)
        } else {
            32_000 - i32::from(ply)
        };
        return MinimaxResult { white_score, best_moves: Vec::new(), nodes: 1 };
    }
    if material_draw(board) {
        return MinimaxResult {
            white_score: 0,
            best_moves: if depth > 0 { moves.iter().copied().collect() } else { Vec::new() },
            nodes: 1,
        };
    }
    if depth == 0 {
        return quiescent_minimax(board, 0, ply);
    }

    let mut result = MinimaxResult {
        white_score: if maximizing { i32::MIN } else { i32::MAX },
        best_moves: Vec::new(),
        nodes: 1,
    };
    for &mv in moves.iter() {
        let mut child = board.clone();
        child.make_move(mv);
        let child = minimax(&child, depth - 1, ply + 1);
        result.nodes += child.nodes;
        let better =
            if maximizing { child.white_score > result.white_score } else { child.white_score < result.white_score };
        if better {
            result.white_score = child.white_score;
            result.best_moves.clear();
        }
        if child.white_score == result.white_score {
            result.best_moves.push(mv);
        }
    }
    result
}

fn quiescent_minimax(board: &Board, q_depth: i16, ply: u16) -> MinimaxResult {
    let moves = board.generate_all_moves();
    let maximizing = board.side_to_move() == Color::White;
    let checked = board.in_check();
    let leaf = |white_score| MinimaxResult { white_score, best_moves: Vec::new(), nodes: 1 };
    if checked && moves.is_empty() {
        return leaf(if maximizing { -32_000 + i32::from(ply) } else { 32_000 - i32::from(ply) });
    }
    if material_draw(board) {
        return leaf(0);
    }
    if ply >= 256 {
        return leaf(if checked { 0 } else { material(board) });
    }

    // Only unchecked nodes can choose the static score instead of a move.
    // In particular, unchecked stalemate inside QS keeps this static score.
    let mut result = leaf(if !checked {
        material(board)
    } else if maximizing {
        i32::MIN
    } else {
        i32::MAX
    });
    for &mv in moves.iter() {
        let tactical = mv.is_capture() || mv.is_promotion();
        if !checked && !tactical && q_depth < -1 {
            continue;
        }
        let mut child = board.clone();
        child.make_move(mv);
        // Detect checking quiets by playing the move, independently of the
        // production check predictor, quiet-check generator and staged picker.
        if !checked && !tactical && !child.in_check() {
            continue;
        }
        let child = quiescent_minimax(&child, q_depth - 1, ply + 1);
        result.nodes += child.nodes;
        result.white_score = if maximizing {
            result.white_score.max(child.white_score)
        } else {
            result.white_score.min(child.white_score)
        };
    }
    result
}

#[test]
fn reference_search_matches_unpruned_quiescent_minimax_scores_and_optimal_moves() {
    let mut observed_cutoff = false;
    // Unpruned capture trees in full armies can be enormous even at depth two.
    // Keep those smoke checks shallow; exhaust the smaller tactical positions.
    for (fen, frc, max_depth) in [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, 1),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", false, 1),
        ("4k3/8/8/8/8/8/q7/R3K3 w - - 0 1", false, 3),
        ("r3k3/Q7/8/8/8/8/8/4K3 b - - 0 1", false, 3),
        ("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1", false, 3),
        ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", false, 3),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false, 3),
        ("4k3/8/8/8/8/8/8/5KR1 w G - 0 1", true, 3),
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", false, 3),
        ("8/8/8/8/8/6k1/5q2/7K b - - 0 1", false, 3),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", false, 3),
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", false, 3),
        ("7k/8/5K1Q/8/8/8/8/8 b - - 0 1", false, 3),
        ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", false, 3),
        ("7k/7P/8/6K1/8/8/8/8 w - - 0 1", false, 3),
        ("7k/7P/6K1/8/8/8/8/8 w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/8/7K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/N7/7K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/B7/7K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/NN6/7K w - - 0 1", false, 3),
        ("7k/8/nn6/8/8/8/8/7K b - - 0 1", false, 3),
        ("7k/8/1b6/8/8/8/N7/7K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/B1B5/7K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/BN6/7K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/B7/2B4K w - - 0 1", false, 3),
        ("7k/8/8/8/8/8/B1B1B3/7K w - - 0 1", false, 3),
        ("7k/5K2/5Nr1/4N3/8/8/8/8 w - - 0 1", false, 3),
        ("8/8/8/8/4n3/5nR1/5k2/7K b - - 0 1", false, 3),
        ("7k/5K2/5NN1/8/8/8/8/8 b - - 0 1", false, 3),
        ("7k/5K1n/6N1/8/8/8/8/8 b - - 0 1", false, 3),
        ("7k/8/8/8/8/2r5/NN6/7K w - - 0 1", false, 3),
        ("7k/7P/8/p5K1/P7/8/8/8 w - - 0 1", false, 3),
        ("7k/7P/6K1/p7/P7/8/8/8 w - - 0 1", false, 3),
        ("r6Q/1k6/8/8/8/8/8/K7 w - - 0 1", false, 3),
        ("8/8/8/8/8/kq6/8/1K6 w - - 0 1", false, 3),
        ("7k/8/5K2/8/8/8/3Q4/8 w - - 0 1", false, 3),
        ("8/kBPN4/2K5/8/8/8/8/8 w - - 0 1", false, 3),
        ("6k1/8/4p3/3p4/8/8/8/3Q2K1 w - - 0 1", false, 3),
    ] {
        let mut board = Board::from_fen(fen).unwrap();
        board.set_frc(frc);
        let original = board.clone();
        for depth in 0..=max_depth {
            let expected = minimax(&board, depth, 0);
            let actual = search_reference(&board, depth);
            let sign = if board.side_to_move() == Color::White { 1 } else { -1 };
            assert_eq!(actual.score, sign * expected.white_score, "{fen}, depth {depth}");
            match actual.best_move {
                Some(mv) => assert!(expected.best_moves.contains(&mv), "suboptimal move {mv:?}: {fen}, depth {depth}"),
                None => assert!(expected.best_moves.is_empty(), "missing best move: {fen}, depth {depth}"),
            }
            assert!(actual.nodes <= expected.nodes, "visited more nodes than unpruned minimax");
            observed_cutoff |= actual.nodes < expected.nodes;
            assert!(board == original, "search changed the position or its history");
        }
    }
    assert!(observed_cutoff, "alpha-beta must prune at least one branch in these fixtures");
}
