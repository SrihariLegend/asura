use std::{
    collections::VecDeque,
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    board::Board,
    evaluation::evaluate,
    parameters::DEFAULT_MOVE_OVERHEAD_MS,
    search::{MATE_SCORE, MATE_THRESHOLD, search_with_context},
    thread::{SharedContext, Status},
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    tools,
    transposition::{DEFAULT_TT_SIZE, TranspositionTable},
    types::{Color, MAX_MOVES, Move},
};

struct ScopedInputPtr(*mut (dyn BufRead + 'static));
// Safety: Only accessed by the single scoped reader thread inside message_loop before the scope exits.
unsafe impl Send for ScopedInputPtr {}

pub fn message_loop(commands: VecDeque<String>, mut input: impl BufRead, mut output: impl Write) -> io::Result<()> {
    let mut board = Board::starting_position();
    let mut tt = TranspositionTable::default();
    let shared = Arc::new(SharedContext::default());
    let mut threads = ThreadPool::new();
    let mut frc = false;
    let mut move_overhead_ms = DEFAULT_MOVE_OVERHEAD_MS;
    let mut multi_pv: usize = 1;
    let use_stdin = Arc::new(AtomicBool::new(commands.is_empty()));

    let dyn_input: &mut (dyn BufRead + '_) = &mut input;
    let raw_input: *mut (dyn BufRead + 'static) =
        unsafe { std::mem::transmute::<*mut (dyn BufRead + '_), *mut (dyn BufRead + 'static)>(dyn_input) };
    let input_ptr = ScopedInputPtr(raw_input);
    let shared_reader = Arc::clone(&shared);
    let use_stdin_reader = Arc::clone(&use_stdin);

    std::thread::scope(|scope| -> io::Result<()> {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel::<io::Result<Option<String>>>(0);
        let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel::<()>(0);

        scope.spawn(move || {
            let ptr = input_ptr;
            let reader: &mut dyn BufRead = unsafe { &mut *ptr.0 };
            let mut queued = commands;

            loop {
                let message = if let Some(cmd) = queued.pop_front() {
                    cmd
                } else if use_stdin_reader.load(Ordering::Acquire) {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) => {
                            if shared_reader.infinite_search.load(Ordering::Acquire) {
                                shared_reader.status.set(Status::STOPPED);
                            }
                            let _ = cmd_tx.send(Ok(None));
                            break;
                        }
                        Ok(_) => line,
                        Err(e) => {
                            shared_reader.status.set(Status::STOPPED);
                            let _ = cmd_tx.send(Err(e));
                            break;
                        }
                    }
                } else {
                    let _ = cmd_tx.send(Ok(None));
                    break;
                };

                let trimmed = message.trim();
                if shared_reader.status.get() == Status::RUNNING && trimmed == "stop" {
                    shared_reader.status.set(Status::STOPPED);
                    continue;
                }
                if shared_reader.infinite_search.load(Ordering::Acquire) && trimmed == "quit" {
                    shared_reader.status.set(Status::STOPPED);
                }

                let is_quit = trimmed == "quit";
                if cmd_tx.send(Ok(Some(message))).is_err() || is_quit {
                    break;
                }
                if ack_rx.recv().is_err() {
                    break;
                }
            }
        });

        while let Ok(next_msg) = cmd_rx.recv() {
            let Some(message) = next_msg? else {
                break;
            };

            let mut acked = false;
            let tokens: Vec<_> = message.split_whitespace().collect();
            match tokens.as_slice() {
                ["uci"] => {
                    writeln!(output, "id name Asura {}", env!("CARGO_PKG_VERSION"))?;
                    writeln!(output, "id author Asura contributors")?;
                    writeln!(output, "option name Hash type spin default {DEFAULT_TT_SIZE} min 1 max 262144")?;
                    writeln!(
                        output,
                        "option name Threads type spin default 1 min 1 max {}",
                        ThreadPool::available_threads()
                    )?;
                    writeln!(
                        output,
                        "option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max 5000"
                    )?;
                    writeln!(output, "option name MultiPV type spin default 1 min 1 max {MAX_MOVES}")?;
                    writeln!(output, "option name Clear Hash type button")?;
                    writeln!(output, "option name UCI_Chess960 type check default false")?;
                    writeln!(output, "uciok")?;
                    use_stdin.store(true, Ordering::Release);
                }
                ["isready"] => writeln!(output, "readyok")?,
                ["ucinewgame"] => {
                    board = Board::starting_position();
                    board.set_frc(frc);
                    threads.clear();
                    tt.clear(threads.len());
                }
                ["position", tokens @ ..] => match position(tokens, frc) {
                    Ok(position) => board = position,
                    Err(error) => writeln!(output, "info string {error}")?,
                },
                ["setoption", "name", "Hash", "value", value] => {
                    match value.parse::<usize>().ok().filter(|&mb| (1..=262_144).contains(&mb)) {
                        Some(mb) => tt.resize(threads.len(), mb),
                        None => writeln!(output, "info string Hash must be an integer from 1 to 262144")?,
                    }
                }
                ["setoption", "name", "Threads", "value", value] => {
                    let max_threads = ThreadPool::available_threads();
                    match value.parse::<usize>().ok().filter(|&n| (1..=max_threads).contains(&n)) {
                        Some(n) => threads.set_count(n),
                        None => writeln!(output, "info string Threads must be an integer from 1 to {max_threads}")?,
                    }
                }
                ["setoption", "name", "Move", "Overhead", "value", value]
                | ["setoption", "name", "MoveOverhead", "value", value] => {
                    match value.parse::<u64>().ok().filter(|&ms| ms <= 5_000) {
                        Some(ms) => move_overhead_ms = ms,
                        None => writeln!(output, "info string Move Overhead must be an integer from 0 to 5000")?,
                    }
                }
                ["setoption", "name", "MultiPV", "value", value] => {
                    match value.parse::<usize>().ok().filter(|&n| (1..=MAX_MOVES).contains(&n)) {
                        Some(n) => multi_pv = n,
                        None => writeln!(output, "info string MultiPV must be an integer from 1 to {MAX_MOVES}")?,
                    }
                }
                ["setoption", "name", "Clear", "Hash"] => tt.clear(threads.len()),
                ["setoption", "name", "UCI_Chess960", "value", value] => match value.parse::<bool>() {
                    Ok(value) => {
                        frc = value;
                        board.set_frc(frc);
                    }
                    Err(_) => writeln!(output, "info string UCI_Chess960 requires true or false")?,
                },
                ["setoption", "name", name, "value", value] => {
                    if !crate::parameters::set_parameter(name, value) {
                        writeln!(output, "info string Unknown option {name}")?;
                    }
                }
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
                    [] => {
                        tools::bench(5, &mut output)?;
                    }
                    [depth] => match depth.parse::<u8>().ok().filter(|&d| d > 0) {
                        Some(d) => {
                            tools::bench(d, &mut output)?;
                        }
                        None => writeln!(output, "info string bench depth must be an integer from 1 to 255")?,
                    },
                    _ => writeln!(output, "info string Usage: bench [depth]")?,
                },
                ["go", tokens @ ..] => match parse_go(&board, tokens, move_overhead_ms) {
                    Ok(ParsedGo::FixedDepth(depth)) if threads.len() == 1 && multi_pv == 1 => {
                        threads.prepare_search(&tt, false, &shared);
                        let _ = ack_tx.send(());
                        acked = true;
                        go_fixed_depth(&board, &tt, &shared, &mut threads, depth, &mut output)?;
                    }
                    Ok(ParsedGo::FixedDepth(depth)) => {
                        let tm = TimeManager::new(Limits::Depth(depth), board.fullmove_number(), move_overhead_ms);
                        threads.prepare_search(&tt, false, &shared);
                        let _ = ack_tx.send(());
                        acked = true;
                        go_managed(&board, &tt, &shared, &mut threads, tm, multi_pv, depth, &mut output)?;
                    }
                    Ok(ParsedGo::Managed(tm, report_depth)) => {
                        threads.prepare_search(&tt, matches!(tm.limits(), Limits::Infinite), &shared);
                        let _ = ack_tx.send(());
                        acked = true;
                        go_managed(&board, &tt, &shared, &mut threads, tm, multi_pv, report_depth, &mut output)?;
                    }
                    Err(error) => writeln!(output, "info string {error}")?,
                },
                ["quit"] => break,
                ["stop"] | [] => (),
                _ => (),
            }
            output.flush()?;
            if !acked && ack_tx.send(()).is_err() {
                break;
            }
        }
        output.flush()
    })
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
        return Err("Usage: go depth|nodes|movetime|mate|wtime|btime|infinite [searchmoves <moves...>]".to_string());
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
    let mut infinite = false;

    let legal_moves = board.generate_all_moves();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "infinite" | "ponder" => {
                if infinite {
                    return Err("Duplicate 'infinite' limit".to_string());
                }
                infinite = true;
                i += 1;
            }
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
    let has_limit = depth.is_some() || nodes.is_some() || movetime.is_some() || mate.is_some() || has_clock || infinite;
    if !has_limit {
        return Err("go requires at least one depth, nodes, movetime, mate, clock, or infinite limit".to_string());
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
        Limits::Infinite
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
    board: &Board, tt: &TranspositionTable, shared: &Arc<SharedContext>, threads: &mut ThreadPool, depth: u8,
    output: &mut impl Write,
) -> io::Result<()> {
    tt.increment_age();
    let result = search_with_context(board, depth, tt, &mut threads.main_thread().history);
    shared.status.set(Status::STOPPED);
    let score = format_uci_score(result.score);
    writeln!(output, "info depth {depth} score {score} nodes {}", result.nodes)?;

    let best_move = result.best_move.map(|mv| mv.to_uci(board)).unwrap_or_else(|| "0000".to_string());
    writeln!(output, "bestmove {best_move}")
}

#[allow(clippy::too_many_arguments)]
fn go_managed(
    board: &Board, tt: &TranspositionTable, shared: &Arc<SharedContext>, threads: &mut ThreadPool, tm: TimeManager,
    multi_pv: usize, report_depth: u8, output: &mut impl Write,
) -> io::Result<()> {
    let limits = tm.limits();
    threads.execute_searches(board, tt, tm, multi_pv, shared);
    let best_idx = threads.select_best_thread(limits);
    let total_nodes = if threads.len() > 1 { shared.nodes.aggregate() } else { threads[best_idx].nodes };
    let worker = &threads[best_idx];
    let depth = worker.completed_depth.max(report_depth);

    if multi_pv > 1 && !worker.root_moves.is_empty() {
        let count = multi_pv.min(worker.root_moves.records.len());
        for i in 0..count {
            let rm = worker.root_moves.at(i);
            let res = rm.active_result();
            let score = format_uci_score(res.score);
            writeln!(output, "info depth {depth} multipv {} score {score} nodes {total_nodes}", i + 1)?;
        }
    } else {
        let score = format_uci_score(worker.score);
        writeln!(output, "info depth {depth} score {score} nodes {total_nodes}")?;
    }

    let best_move = worker.best_move.map(|mv| mv.to_uci(board)).unwrap_or_else(|| "0000".to_string());
    writeln!(output, "bestmove {best_move}")
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
