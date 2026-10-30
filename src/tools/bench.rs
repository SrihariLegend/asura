//! Fixed-depth benchmark suite for verifying search determinism and throughput.

use std::{
    io::{self, Write},
    time::Instant,
};

use crate::{board::Board, history::HistoryTables, search::search_with_context, transposition::TranspositionTable};

const POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
];

pub fn bench(depth: u8, output: &mut impl Write) -> io::Result<u64> {
    let start = Instant::now();
    let mut tt = TranspositionTable::default();
    let mut history = HistoryTables::default();
    let mut total_nodes = 0u64;

    for &fen in POSITIONS {
        let board = Board::from_fen(fen).unwrap();
        tt.clear(1);
        history.clear();
        let res = search_with_context(&board, depth, &tt, &mut history);
        total_nodes += res.nodes;
    }

    let elapsed_secs = start.elapsed().as_secs_f64().max(1e-6);
    let nps = (total_nodes as f64 / elapsed_secs) as u64;
    writeln!(output, "{total_nodes} nodes {nps} nps")?;
    Ok(total_nodes)
}
