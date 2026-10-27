use crate::{
    board::Board,
    evaluation::evaluate,
    movepick::QuiescenceMovePicker,
    types::{Move, Square},
};

pub(crate) const MATE_SCORE: i32 = 32_000;
const MAX_PLY: u16 = 256;
const RECAPTURE_ONLY_DEPTH: i16 = -4;
pub(crate) const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;
const INFINITY: i32 = MATE_SCORE + 1;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SearchResult {
    /// No move is selected at depth zero or when no legal move exists.
    /// A material-drawn root returns its first legal move at positive depth.
    pub best_move: Option<Move>,
    /// Root-side score: centipawns, or +/- (32_000 - plies to checkmate).
    /// Normal search jitters draws in -2..=2; zero-draw mode scores them zero.
    /// A checkmated root scores -32_000.
    pub score: i32,
    /// Visited nodes (moves made during search).
    pub nodes: u64,
}

/// Fixed-depth, single-threaded alpha-beta with selective quiescent leaves.
/// Depth zero runs QS without selecting a root move. The public root and
/// full-width nodes recognize stalemate; unchecked interior QS nodes deliberately
/// keep stand pat rather than probing for it. Checkmate overrides material draws.
/// The caller's position/history is unchanged; root ties keep the first move.
/// Draws use `(visited_nodes % 5) - 2`, including the current node. This is
/// deterministic for a fixed traversal, but is not a position-only value.
pub fn search(position: &Board, depth: u8) -> SearchResult {
    run_search::<true, true>(position, depth)
}

/// The normal selective search, with recognized draws scored exactly zero.
pub fn search_with_zero_draws(position: &Board, depth: u8) -> SearchResult {
    run_search::<false, true>(position, depth)
}

/// Diagnostic search with zero draw scores and no selective QS pruning.
/// Keeps every evasion and capture eligible for independent minimax comparisons.
/// Alpha-beta cutoffs, the shallow quiet-check budget and maximum ply still apply.
pub fn search_reference(position: &Board, depth: u8) -> SearchResult {
    run_search::<false, false>(position, depth)
}

fn run_search<const DRAW_JITTER: bool, const SELECTIVE: bool>(position: &Board, depth: u8) -> SearchResult {
    let mut board = position.clone();
    let mut nodes = 0;
    let (score, best_move) =
        negamax::<DRAW_JITTER, SELECTIVE>(&mut board, depth, 0, None, -INFINITY, INFINITY, &mut nodes);
    debug_assert!(board == *position, "search did not undo every move");
    SearchResult { best_move, score, nodes }
}

fn negamax<const DRAW_JITTER: bool, const SELECTIVE: bool>(
    board: &mut Board, depth: u8, ply: u16, previous: Option<Square>, mut alpha: i32, beta: i32, nodes: &mut u64,
) -> (i32, Option<Move>) {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    if depth == 0 {
        // Preserve terminal adjudication at the public root. Interior QS nodes
        // deliberately do not probe unchecked positions for stalemate.
        if ply == 0 && !board.has_legal_moves() {
            let score = if board.in_check() { -MATE_SCORE } else { draw_score::<DRAW_JITTER>(*nodes) };
            return (score, None);
        }
        return (quiescence::<DRAW_JITTER, SELECTIVE>(board, 0, ply, previous, alpha, beta, nodes), None);
    }
    let draw = draw_score::<DRAW_JITTER>(*nodes);
    if ply > 0 && alpha < draw && board.upcoming_repetition(ply as usize) {
        alpha = draw;
        if alpha >= beta {
            return (alpha, None);
        }
    }

    let moves = board.generate_all_moves();

    // Full-width nodes adjudicate both checkmate and stalemate.
    if moves.is_empty() {
        let score = if board.in_check() { -MATE_SCORE + i32::from(ply) } else { draw };
        return (score, None);
    }
    if board.is_draw(ply as isize) {
        // A drawn root still needs a legal move for UCI, not 0000.
        let best_move = (ply == 0).then_some(moves[0]);
        return (draw, best_move);
    }
    let mut best_score = -INFINITY;
    let mut best_move = None;
    for &mv in moves.iter() {
        make_move(board, mv, nodes);
        let score =
            -negamax::<DRAW_JITTER, SELECTIVE>(board, depth - 1, ply + 1, Some(mv.to()), -beta, -alpha, nodes).0;
        undo_move(board, mv);

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
        alpha = alpha.max(score);
        if alpha >= beta {
            break;
        }
    }
    (best_score, best_move)
}

fn quiescence<const DRAW_JITTER: bool, const SELECTIVE: bool>(
    board: &mut Board, q_depth: i16, ply: u16, previous: Option<Square>, mut alpha: i32, beta: i32, nodes: &mut u64,
) -> i32 {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    debug_assert!(ply <= MAX_PLY && q_depth <= 0);
    let draw = draw_score::<DRAW_JITTER>(*nodes);
    if ply > 0 && alpha < draw && board.upcoming_repetition(ply as usize) {
        alpha = draw;
        if alpha >= beta {
            return alpha;
        }
    }

    let in_check = board.in_check();
    let is_draw = board.is_draw(ply as isize);
    if is_draw || ply >= MAX_PLY {
        if in_check && !board.has_legal_moves() {
            return -MATE_SCORE + i32::from(ply);
        }
        return if is_draw || in_check { draw } else { evaluate(board) };
    }

    let mut best_score = if in_check { -INFINITY } else { evaluate(board) };
    if best_score >= beta {
        return best_score;
    }
    alpha = alpha.max(best_score);

    let mut picker = QuiescenceMovePicker::new(in_check, q_depth >= -1);
    // An ordinary score (even a negative one) permits skipping quiet escapes.
    // The picker must still offer both good and deferred noisy evasions.
    while let Some(mv) = picker.next(board, SELECTIVE && in_check && best_score > -MATE_THRESHOLD) {
        // Keep this separate from picker stage control. In particular, neither
        // deep filtering nor SEE ordering may discard a checked noisy evasion.
        // En passant recaptures are identified by the victim, not the landing square.
        let recapture = mv.is_capture() && previous == Some(mv.capture_sq());
        if SELECTIVE
            && !in_check
            && q_depth <= RECAPTURE_ONLY_DEPTH
            && best_score > -MATE_THRESHOLD
            && !recapture
            && !mv.is_promotion()
            && !board.gives_check(mv)
        {
            continue;
        }
        make_move(board, mv, nodes);
        let score =
            -quiescence::<DRAW_JITTER, SELECTIVE>(board, q_depth - 1, ply + 1, Some(mv.to()), -beta, -alpha, nodes);
        undo_move(board, mv);
        best_score = best_score.max(score);
        alpha = alpha.max(score);
        if alpha >= beta {
            break;
        }
    }
    if in_check && best_score == -INFINITY { -MATE_SCORE + i32::from(ply) } else { best_score }
}

fn make_move(board: &mut Board, mv: Move, nodes: &mut u64) {
    *nodes += 1;
    board.make_move(mv);
}

fn undo_move(board: &mut Board, mv: Move) {
    board.undo_move(mv);
}

fn draw_score<const DRAW_JITTER: bool>(nodes: u64) -> i32 {
    if DRAW_JITTER { (nodes % 5) as i32 - 2 } else { 0 }
}
