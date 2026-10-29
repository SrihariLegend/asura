use std::{
    collections::VecDeque,
    io::{self, BufRead, Write},
};

use crate::{
    board::Board,
    evaluation::evaluate,
    history::HistoryTables,
    parameters::DEFAULT_MOVE_OVERHEAD_MS,
    search::{MATE_SCORE, MATE_THRESHOLD, search_with_context, search_with_time_manager},
    time::{Limits, TimeManager},
    tools,
    transposition::{DEFAULT_TT_SIZE, TranspositionTable},
    types::{Color, Move},
};

const BENCH_FENS: [&str; 5] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
];

pub fn message_loop(mut commands: VecDeque<String>, mut input: impl BufRead, mut output: impl Write) -> io::Result<()> {
    let mut board = Board::starting_position();
    let mut tt = TranspositionTable::default();
    let mut history = HistoryTables::default();
    let mut frc = false;
    let mut move_overhead_ms = DEFAULT_MOVE_OVERHEAD_MS;
    let mut use_stdin = commands.is_empty();

    loop {
        let message = if let Some(command) = commands.pop_front() {
            command
        } else if use_stdin {
            let mut message = String::new();
            if input.read_line(&mut message)? == 0 {
                break;
            }
            message
        } else {
            break;
        };

        let tokens: Vec<_> = message.split_whitespace().collect();
        match tokens.as_slice() {
            ["uci"] => {
                writeln!(output, "id name Asura {}", env!("CARGO_PKG_VERSION"))?;
                writeln!(output, "id author Asura contributors")?;
                writeln!(output, "option name Hash type spin default {DEFAULT_TT_SIZE} min 1 max 262144")?;
                writeln!(
                    output,
                    "option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max 5000"
                )?;
                writeln!(output, "option name Clear Hash type button")?;
                writeln!(output, "option name UCI_Chess960 type check default false")?;
                writeln!(output, "uciok")?;
                use_stdin = true;
            }
            ["isready"] => writeln!(output, "readyok")?,
            ["ucinewgame"] => {
                board = Board::starting_position();
                board.set_frc(frc);
                tt.clear(1);
                history.clear();
            }
            ["position", tokens @ ..] => match position(tokens, frc) {
                Ok(position) => board = position,
                Err(error) => writeln!(output, "info string {error}")?,
            },
            ["setoption", "name", "Hash", "value", value] => {
                match value.parse::<usize>().ok().filter(|&mb| (1..=262_144).contains(&mb)) {
                    Some(mb) => tt.resize(1, mb),
                    None => writeln!(output, "info string Hash must be an integer from 1 to 262144")?,
                }
            }
            ["setoption", "name", "Move", "Overhead", "value", value]
            | ["setoption", "name", "MoveOverhead", "value", value] => {
                match value.parse::<u64>().ok().filter(|&ms| ms <= 5_000) {
                    Some(ms) => move_overhead_ms = ms,
                    None => writeln!(output, "info string Move Overhead must be an integer from 0 to 5000")?,
                }
            }
            ["setoption", "name", "Clear", "Hash"] => tt.clear(1),
            ["setoption", "name", "UCI_Chess960", "value", value] => match value.parse::<bool>() {
                Ok(value) => {
                    frc = value;
                    board.set_frc(frc);
                }
                Err(_) => writeln!(output, "info string UCI_Chess960 requires true or false")?,
            },
            ["perft", depth] => match depth.parse::<usize>() {
                Ok(depth) => {
                    tools::divide(&mut board, depth, &mut output)?;
                }
                Err(_) => writeln!(output, "info string perft depth must be a non-negative integer")?,
            },
            ["perft", ..] => writeln!(output, "info string Usage: perft <depth>")?,
            ["d"] => writeln!(output, "{board}")?,
            ["eval"] => {
                let cp = evaluate(&board);
                writeln!(output, "info string Static evaluation: {cp} cp (side to move)")?;
            }
            ["bench", args @ ..] => match args {
                [] => run_bench(5, &mut output)?,
                [depth] => match depth.parse::<u8>().ok().filter(|&d| d > 0) {
                    Some(d) => run_bench(d, &mut output)?,
                    None => writeln!(output, "info string bench depth must be an integer from 1 to 255")?,
                },
                _ => writeln!(output, "info string Usage: bench [depth]")?,
            },
            ["go", tokens @ ..] => match parse_go(&board, tokens, move_overhead_ms) {
                Ok(ParsedGo::FixedDepth(depth)) => go_fixed_depth(&board, &tt, &mut history, depth, &mut output)?,
                Ok(ParsedGo::Managed(tm, report_depth)) => {
                    go_managed(&board, &tt, &mut history, tm, report_depth, &mut output)?
                }
                Err(error) => writeln!(output, "info string {error}")?,
            },
            ["quit"] => break,
            // Commands run synchronously; there is no background search to stop.
            ["stop"] | [] => (),
            _ => (),
        }
        output.flush()?;
    }
    output.flush()
}

enum ParsedGo {
    FixedDepth(u8),
    Managed(TimeManager, u8),
}

fn is_go_keyword(token: &str) -> bool {
    matches!(
        token,
        "depth"
            | "nodes"
            | "movetime"
            | "mate"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "searchmoves"
            | "infinite"
            | "ponder"
    )
}

fn parse_go(board: &Board, tokens: &[&str], move_overhead_ms: u64) -> Result<ParsedGo, String> {
    if tokens.is_empty() {
        return Err("Usage: go depth|nodes|movetime|mate|wtime|btime <value> [searchmoves <moves...>]".to_string());
    }

    // Fast path for plain `go depth <N>`.
    if let ["depth", depth] = tokens {
        let d = depth
            .parse::<u8>()
            .ok()
            .filter(|&d| d > 0)
            .ok_or_else(|| "depth must be an integer from 1 to 255".to_string())?;
        return Ok(ParsedGo::FixedDepth(d));
    }

    let mut depth: Option<u8> = None;
    let mut nodes: Option<u64> = None;
    let mut movetime: Option<u64> = None;
    let mut mate: Option<u8> = None;
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: Option<u64> = None;
    let mut binc: Option<u64> = None;
    let mut movestogo: Option<u64> = None;
    let mut search_moves: Option<Vec<Move>> = None;

    let legal_moves = board.generate_all_moves();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if depth.is_some() {
                    return Err("Duplicate 'depth' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u8>().ok())
                    .filter(|&d| d > 0)
                    .ok_or_else(|| "depth must be an integer from 1 to 255".to_string())?;
                depth = Some(val);
                i += 2;
            }
            "nodes" => {
                if nodes.is_some() {
                    return Err("Duplicate 'nodes' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| "nodes must be a positive integer".to_string())?;
                nodes = Some(val);
                i += 2;
            }
            "movetime" => {
                if movetime.is_some() {
                    return Err("Duplicate 'movetime' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .filter(|&ms| ms > 0)
                    .ok_or_else(|| "movetime must be a positive integer".to_string())?;
                movetime = Some(val);
                i += 2;
            }
            "mate" => {
                if mate.is_some() {
                    return Err("Duplicate 'mate' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u8>().ok())
                    .filter(|&m| (1..=128).contains(&m))
                    .ok_or_else(|| "mate must be an integer from 1 to 128".to_string())?;
                mate = Some(val);
                i += 2;
            }
            "wtime" => {
                if wtime.is_some() {
                    return Err("Duplicate 'wtime' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| "wtime must be a non-negative integer".to_string())?;
                wtime = Some(val);
                i += 2;
            }
            "btime" => {
                if btime.is_some() {
                    return Err("Duplicate 'btime' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| "btime must be a non-negative integer".to_string())?;
                btime = Some(val);
                i += 2;
            }
            "winc" => {
                if winc.is_some() {
                    return Err("Duplicate 'winc' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| "winc must be a non-negative integer".to_string())?;
                winc = Some(val);
                i += 2;
            }
            "binc" => {
                if binc.is_some() {
                    return Err("Duplicate 'binc' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| "binc must be a non-negative integer".to_string())?;
                binc = Some(val);
                i += 2;
            }
            "movestogo" => {
                if movestogo.is_some() {
                    return Err("Duplicate 'movestogo' limit".to_string());
                }
                let val = tokens
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .filter(|&m| m > 0)
                    .ok_or_else(|| "movestogo must be a positive integer".to_string())?;
                movestogo = Some(val);
                i += 2;
            }
            "searchmoves" => {
                if search_moves.is_some() {
                    return Err("Duplicate 'searchmoves' clause".to_string());
                }
                i += 1;
                let mut moves = Vec::new();
                while i < tokens.len() && !is_go_keyword(tokens[i]) {
                    let uci_token = tokens[i];
                    let mv = legal_moves
                        .iter()
                        .copied()
                        .find(|m| m.to_uci(board) == uci_token)
                        .ok_or_else(|| format!("Illegal searchmove: {uci_token}"))?;
                    if !moves.contains(&mv) {
                        moves.push(mv);
                    }
                    i += 1;
                }
                if moves.is_empty() {
                    return Err("searchmoves requires at least one legal move".to_string());
                }
                search_moves = Some(moves);
            }
            other => return Err(format!("Unsupported or unexpected go token: {other}")),
        }
    }

    let stm = board.side_to_move();
    let clock = if stm == Color::White { wtime } else { btime };
    let inc = if stm == Color::White { winc } else { binc }.unwrap_or(0);

    let has_clock = wtime.is_some() || btime.is_some();
    let has_finite_limit = depth.is_some() || nodes.is_some() || movetime.is_some() || mate.is_some() || has_clock;
    if !has_finite_limit {
        return Err("go requires at least one finite depth, nodes, movetime, mate, or clock limit".to_string());
    }

    let primary_limits = if let Some(ms) = movetime {
        Limits::MoveTime(ms)
    } else if has_clock {
        let main_ms = clock.unwrap_or(0);
        if let Some(mtg) = movestogo {
            Limits::Cyclic { main_ms, inc_ms: inc, moves_to_go: mtg }
        } else {
            Limits::Fischer { main_ms, inc_ms: inc }
        }
    } else if let Some(n) = nodes {
        Limits::Nodes(n)
    } else if let Some(m) = mate {
        Limits::Mate(m)
    } else if let Some(d) = depth {
        Limits::Depth(d)
    } else {
        unreachable!()
    };

    let tm = TimeManager::new(primary_limits, board.fullmove_number(), move_overhead_ms).with_overrides(
        depth,
        mate,
        nodes,
        search_moves,
    );
    let report_depth = depth.unwrap_or(1);
    Ok(ParsedGo::Managed(tm, report_depth))
}

fn format_uci_score(score: i32) -> String {
    if score.abs() >= MATE_THRESHOLD {
        let plies = MATE_SCORE - score.abs();
        let moves = (plies + i32::from(score > 0)) / 2;
        let mate = if score < 0 { -moves } else { moves };
        format!("mate {mate}")
    } else {
        format!("cp {score}")
    }
}

fn go_fixed_depth(
    board: &Board, tt: &TranspositionTable, history: &mut HistoryTables, depth: u8, output: &mut impl Write,
) -> io::Result<()> {
    tt.increment_age();
    let result = search_with_context(board, depth, tt, history);
    let score = format_uci_score(result.score);
    writeln!(output, "info depth {depth} score {score} nodes {}", result.nodes)?;

    let best_move = result.best_move.map(|mv| mv.to_uci(board)).unwrap_or_else(|| "0000".to_string());
    writeln!(output, "bestmove {best_move}")
}

fn go_managed(
    board: &Board, tt: &TranspositionTable, history: &mut HistoryTables, tm: TimeManager, report_depth: u8,
    output: &mut impl Write,
) -> io::Result<()> {
    tt.increment_age();
    let result = search_with_time_manager(board, tm, tt, history);
    let score = format_uci_score(result.score);
    let depth = result.pv().len().max(report_depth as usize) as u8;
    writeln!(output, "info depth {depth} score {score} nodes {}", result.nodes)?;

    let best_move = result.best_move.map(|mv| mv.to_uci(board)).unwrap_or_else(|| "0000".to_string());
    writeln!(output, "bestmove {best_move}")
}

fn run_bench(depth: u8, output: &mut impl Write) -> io::Result<()> {
    let start = std::time::Instant::now();
    let mut tt = TranspositionTable::default();
    let mut history = HistoryTables::default();
    let mut total_nodes = 0u64;

    for fen in BENCH_FENS {
        let board = Board::from_fen(fen).unwrap();
        tt.clear(1);
        history.clear();
        let res = search_with_context(&board, depth, &tt, &mut history);
        total_nodes += res.nodes;
    }

    let elapsed_secs = start.elapsed().as_secs_f64().max(1e-6);
    let nps = (total_nodes as f64 / elapsed_secs) as u64;
    writeln!(output, "{total_nodes} nodes {nps} nps")
}

/// Construct a new position before replacing the live board, so a malformed FEN
/// or an illegal move cannot leave a partially applied command behind.
fn position(tokens: &[&str], frc: bool) -> Result<Board, String> {
    let moves_index = tokens.iter().position(|&token| token == "moves").unwrap_or(tokens.len());
    let (position, moves) = tokens.split_at(moves_index);
    let mut board = match position {
        ["startpos"] => Board::starting_position(),
        ["fen", fen @ ..] => Board::from_fen(&fen.join(" ")).map_err(|error| format!("Invalid FEN: {error}"))?,
        _ => return Err("Usage: position startpos|fen <fen> [moves <moves...>]".to_string()),
    };
    board.set_frc(frc);

    for &token in moves.iter().skip(1) {
        let legal_moves = board.generate_all_moves();
        let Some(mv) = legal_moves.iter().copied().find(|mv| mv.to_uci(&board) == token) else {
            return Err(format!("Illegal move: {token}"));
        };
        board.make_move(mv);
    }

    Ok(board)
}
