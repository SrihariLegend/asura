//! Move-generation correctness and performance diagnostics.

use std::{
    io::{self, Write},
    time::Instant,
};

use crate::board::Board;

/// Count legal leaf positions at `depth`, restoring the board before returning.
/// Depth zero counts the current position, including checkmate and stalemate.
pub fn perft(board: &mut Board, depth: usize) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = board.generate_all_moves();
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for &mv in moves.iter() {
        board.make_move(mv);
        nodes += perft(board, depth - 1);
        board.undo_move(mv);
    }
    nodes
}

/// Print per-root-move counts and timing. This is diagnostic traversal, not search.
pub fn divide(board: &mut Board, depth: usize, output: &mut impl Write) -> io::Result<u64> {
    let start = Instant::now();
    let mut nodes = 0;

    if depth == 0 {
        nodes = 1;
    } else {
        for &mv in board.generate_all_moves().iter() {
            board.make_move(mv);
            let count = perft(board, depth - 1);
            board.undo_move(mv);
            nodes += count;

            writeln!(output, "{}: {count}", mv.to_uci(board))?;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let nps = (nodes as f64 / elapsed.max(f64::EPSILON)) as u64;
    writeln!(output, "total: {nodes}")?;
    writeln!(output, "time: {elapsed:.3} s")?;
    writeln!(output, "nps: {nps}")?;
    Ok(nodes)
}
