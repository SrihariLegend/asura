//! Frozen native labels and deterministic, integer-reduced style proxies.
//!
//! The caller initializes the engine and holds the global model/tunables fixed
//! throughout each call (including error draining and Drop). These proxies are
//! not calibrated probabilities, a stronger oracle, or strength certification.
//! Deadlines are checked at root-call/position boundaries, not inside search.
//! `Limits::Nodes` uses the engine's stopping policy: its first completed depth
//! is exempt, so a node budget is not an exact node cap or a hard time guarantee.
//! A separate process timeout is needed for a hard operational time limit.
//! Unwinding worker panics become errors; panic=abort, OOM aborts, and external
//! process termination cannot be recovered or joined by Rust destructors.

use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader, Read},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::Instant,
};

use asura::{
    board::Board,
    history::HistoryTables,
    nnue::{self, ModelParameters, NetworkState, PackedBoard},
    search::{SearchResult, search_with_limits},
    time::Limits,
    transposition::TranspositionTable,
    types::{Color, Move, PieceType, Score},
};

// Explicit resource ceilings, in addition to the caller's scan/node/deadline
// limits. Scan counts physical lines (including comments) or packed records.
const MAX_SCAN: usize = 1_000_000;
const MAX_CASES: usize = 4096;
const MAX_LINE: usize = 4096;
const MAX_THREADS: usize = 64;
const MAX_TOTAL_HASH_MB: usize = 4096;
const MAX_NODES: u64 = 100_000_000;
const MAX_FULLMOVE: usize = 1_000_000;

type Metrics = [[i128; 6]; 3];

#[derive(Clone)]
pub struct Sample {
    pub board: Board,
    pub root_cp: i32,
}

pub enum Input {
    Packed(PathBuf),
    Fens(PathBuf),
}

fn canonical(board: &Board) -> String {
    board.to_fen().split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

fn validate_board(board: &Board) -> Result<(), String> {
    if board.fullmove_number() > MAX_FULLMOVE {
        return Err(format!("fullmove number exceeds {MAX_FULLMOVE}"));
    }
    for side in [Color::White, Color::Black] {
        let count = |pt| board.colored_pieces(side, pt).popcount();
        let pawns = count(PieceType::Pawn);
        if count(PieceType::King) != 1 || pawns > 8 || board.colors(side).popcount() > 16 {
            return Err("invalid piece counts".into());
        }
        let promotions = count(PieceType::Knight).saturating_sub(2)
            + count(PieceType::Bishop).saturating_sub(2)
            + count(PieceType::Rook).saturating_sub(2)
            + count(PieceType::Queen).saturating_sub(1);
        if promotions > 8 - pawns {
            return Err("piece counts require more promotions than missing pawns".into());
        }
    }
    Ok(())
}

fn parse_fen(fen: &str) -> Result<Board, String> {
    if !fen.is_ascii() || fen.split_whitespace().count() != 6 {
        return Err("expected six ASCII FEN fields".into());
    }
    let board = Board::from_fen(fen).map_err(|e| format!("invalid FEN: {e}"))?;
    validate_board(&board)?;
    Ok(board)
}

// Do not call PackedBoard::to_board: it clamps invalid piece codes and can
// panic on malformed input. Validate metadata and parse the decoded placement.
fn unpack(pb: &PackedBoard) -> Result<Board, String> {
    let n = pb.occ.count_ones() as usize;
    if !(2..=32).contains(&n) || pb.ksq_w >= 64 || pb.ksq_b >= 64 {
        return Err("invalid packed occupancy or king squares".into());
    }
    let mut grid = [b' '; 64];
    let mut occupied = pb.occ;
    for i in 0..n {
        let sq = occupied.trailing_zeros() as usize;
        occupied &= occupied - 1;
        let word = if i < 16 { pb.pieces_lo } else { pb.pieces_hi };
        let code = ((word >> ((i % 16) * 4)) & 15) as usize;
        grid[sq] = *b"PNBRQKpnbrqk".get(code).ok_or("invalid packed piece code")?;
    }
    if grid[pb.ksq_w as usize] != b'K' || grid[pb.ksq_b as usize] != b'k' {
        return Err("packed king metadata disagrees with placement".into());
    }
    // stm_wdl carries teacher flags as well as side to move. Neither teacher
    // WDL, psqt_cp nor ply_clamped supplies castling/EP/history; like the frozen
    // reader, packed positions are reconstructed with '- - 0 1'.
    let mut fen = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            let pc = grid[rank * 8 + file];
            if pc == b' ' {
                empty += 1;
            } else {
                if empty != 0 {
                    fen.push((b'0' + empty) as char);
                    empty = 0;
                }
                fen.push(pc as char);
            }
        }
        if empty != 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank != 0 {
            fen.push('/');
        }
    }
    fen.push_str(if pb.stm_wdl & 1 == 0 { " w - - 0 1" } else { " b - - 0 1" });
    parse_fen(&fen)
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn shuffle<T>(values: &mut [T], rng: &mut u64) {
    for i in (1..values.len()).rev() {
        let j = (next_random(rng) % (i as u64 + 1)) as usize;
        values.swap(i, j);
    }
}

/// Read at most `scan` records/physical lines, rejecting malformed records and
/// short packed records rather than hiding corruption. Valid but terminal or
/// unsuitable packed positions are skipped. Deduplication uses the canonical
/// first four FEN fields; first eligible occurrence wins even across strata.
/// Seeded, bounded reservoirs prefer equal positive/level/negative legacy score
/// strata, then fill from other scores. FEN fixtures have no piece-count/style
/// sampling filter. Exactly `needed` positions are returned or the call fails.
pub fn load_samples(input: &Input, scan: usize, needed: usize, seed: u64) -> Result<Vec<Sample>, String> {
    if scan == 0 || scan > MAX_SCAN || needed == 0 || needed > MAX_CASES || needed > scan {
        return Err(format!("require 1 <= needed <= {MAX_CASES}, needed <= scan <= {MAX_SCAN}"));
    }
    let (path, packed) = match input {
        Input::Packed(path) => (path, true),
        Input::Fens(path) => (path, false),
    };
    // Reject obvious FIFOs/devices before open (opening a FIFO can block).
    // Regular-file I/O itself has no portable hard-time guarantee.
    let before_open = std::fs::metadata(path).map_err(|e| format!("input metadata: {e}"))?;
    if !before_open.is_file() {
        return Err("input must be a regular file".into());
    }
    let file = File::open(path).map_err(|e| format!("open input: {e}"))?;
    let metadata = file.metadata().map_err(|e| format!("input metadata: {e}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".into());
    }
    if packed && metadata.len() % 32 != 0 {
        return Err("packed input length is not a multiple of 32".into());
    }
    let mut reader = BufReader::new(file);
    let mut seen = HashSet::new();
    let mut pools: [Vec<Sample>; 4] = std::array::from_fn(|_| Vec::new());
    let mut counts = [0u64; 4];
    let mut rng = seed;
    let mut line = Vec::new();
    for index in 0..scan {
        let record: Result<Option<Sample>, String> = (|| {
            if packed {
                if reader.fill_buf().map_err(|e| format!("read input: {e}"))?.is_empty() {
                    return Ok(None);
                }
                let mut raw = [0u8; 32];
                reader.read_exact(&mut raw).map_err(|e| format!("read packed record: {e}"))?;
                let pb = PackedBoard::from_le_bytes(&raw);
                Ok(Some(Sample { board: unpack(&pb)?, root_cp: i32::from(pb.score_cp) }))
            } else {
                line.clear();
                // read_until alone would permit an unbounded allocation.
                (&mut reader)
                    .take((MAX_LINE + 1) as u64)
                    .read_until(b'\n', &mut line)
                    .map_err(|e| format!("read FEN line: {e}"))?;
                if line.len() > MAX_LINE {
                    return Err(format!("FEN line exceeds {MAX_LINE} bytes"));
                }
                if line.is_empty() {
                    return Ok(None);
                }
                let text = std::str::from_utf8(&line).map_err(|e| format!("invalid UTF-8: {e}"))?.trim();
                if text.is_empty() || text.starts_with('#') {
                    // The outer loop counts these lines against scan.
                    return Ok(None);
                }
                let (score, fen) = text.split_once('\t').ok_or("expected root_cp TAB full FEN")?;
                let root_cp = score.parse::<i32>().map_err(|e| format!("invalid root_cp: {e}"))?;
                Ok(Some(Sample { board: parse_fen(fen)?, root_cp }))
            }
        })();
        let sample = record.map_err(|e| format!("input record {}: {e}", index + 1))?;
        let Some(sample) = sample else {
            if packed || line.is_empty() {
                break;
            }
            continue;
        };
        if !(-Score::MATE..=Score::MATE).contains(&sample.root_cp) {
            return Err(format!("input record {}: root_cp outside native score range", index + 1));
        }
        let board = &sample.board;
        if packed && (!(10..=28).contains(&board.occupancies().popcount()) || board.in_check()) {
            continue;
        }
        let moves = board.generate_all_moves();
        if terminal_score(board, moves.len(), 0).is_some() || (packed && moves.len() < 8) {
            continue;
        }
        if !seen.insert(canonical(board)) {
            continue;
        }
        let stratum = if !packed {
            3
        } else if (90..=280).contains(&sample.root_cp) {
            0
        } else if (-50..=50).contains(&sample.root_cp) {
            1
        } else if (-280..=-90).contains(&sample.root_cp) {
            2
        } else {
            3
        };
        counts[stratum] += 1;
        let pool = &mut pools[stratum];
        if pool.len() < needed {
            pool.push(sample);
        } else {
            let slot = next_random(&mut rng) % counts[stratum];
            if slot < needed as u64 {
                pool[slot as usize] = sample;
            }
        }
    }
    let available: u64 = counts.iter().sum();
    if available < needed as u64 {
        return Err(format!(
            "insufficient unique nonterminal samples: needed {needed}, found {available} within scan {scan}"
        ));
    }
    for pool in &mut pools {
        shuffle(pool, &mut rng);
    }
    let mut order = [0, 1, 2];
    shuffle(&mut order, &mut rng);
    let mut result = Vec::with_capacity(needed);
    while result.len() < needed {
        let before = result.len();
        for &i in &order {
            if result.len() < needed
                && let Some(sample) = pools[i].pop()
            {
                result.push(sample);
            }
        }
        if result.len() == before {
            break;
        }
    }
    while result.len() < needed {
        result.push(pools[3].pop().ok_or("internal sample reservoir shortage")?);
    }
    shuffle(&mut result, &mut rng);
    Ok(result)
}

struct MoveLabel {
    mv: Move,
    obj_q_cp: i32,
    min_style_q_cp: i32,
    draw_permille: i32,
    def_difficulty_cp: i32,
    terminal: bool,
}

pub struct LabeledPosition {
    board: Board,
    root_cp: i32,
    best_obj_q_cp: i32,
    moves: Vec<MoveLabel>,
    model: Arc<ModelParameters>,
}

/// A frozen finite position panel, not a collection of independent games.
/// The identifier carries no statistical independence or strength guarantee.
pub struct Panel {
    pub id: u64,
    positions: Arc<Vec<LabeledPosition>>,
}

impl Panel {
    pub fn new(id: u64, positions: Vec<LabeledPosition>) -> Result<Self, String> {
        check_cases(positions.len())?;
        let mut seen = HashSet::new();
        for position in &positions {
            if !Arc::ptr_eq(&positions[0].model, &position.model) {
                return Err("panel mixes labels from different models".into());
            }
            if !seen.insert(canonical(&position.board)) {
                return Err("panel contains duplicate canonical positions".into());
            }
        }
        Ok(Self { id, positions: Arc::new(positions) })
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }
}

/// Searches counts native root calls, nodes counts their reported visited nodes
/// (including failed/deadline-overrun jobs), batches counts dispatched label or
/// evaluate calls, not worker waves. Rejected preflight calls do not count.
#[derive(Clone, Copy, Default, Debug)]
pub struct Work {
    pub searches: u64,
    pub nodes: u64,
    pub batches: u64,
}

impl Work {
    fn add(&mut self, other: Self) -> Result<(), String> {
        let updated = Self {
            searches: self.searches.checked_add(other.searches).ok_or("search count overflow")?,
            nodes: self.nodes.checked_add(other.nodes).ok_or("node count overflow")?,
            batches: self.batches.checked_add(other.batches).ok_or("batch count overflow")?,
        };
        *self = updated;
        Ok(())
    }
}

pub struct Totals {
    pub sums: [[i128; 6]; 3],
    pub cases: usize,
}

impl Totals {
    /// Integer utility in thousandths of the original proxy utility unit.
    /// Signed division truncates toward zero, after exact weighted summation.
    pub fn utility(&self) -> Result<i64, String> {
        if self.cases == 0 {
            return Err("utility requires nonempty totals".into());
        }
        let denominator = (self.cases as i128).checked_mul(100).ok_or("utility denominator overflow")?;
        let mut numerator = 0i128;
        for (r, weight) in [15i128, 30, 55].into_iter().enumerate() {
            let term = self.sums[r][5].checked_mul(weight).ok_or("utility product overflow")?;
            numerator = numerator.checked_add(term).ok_or("utility sum overflow")?;
        }
        i64::try_from(numerator / denominator).map_err(|_| "utility outside i64 range".into())
    }
}

fn check_cases(cases: usize) -> Result<(), String> {
    if cases == 0 || cases > MAX_CASES {
        return Err(format!("require 1..={MAX_CASES} positions"));
    }
    Ok(())
}

fn check_nodes(nodes: u64) -> Result<(), String> {
    if nodes == 0 || nodes > MAX_NODES {
        return Err(format!("require 1..={MAX_NODES} nodes per root"));
    }
    Ok(())
}

fn check_deadline(deadline: Instant) -> Result<(), String> {
    if Instant::now() >= deadline {
        return Err("evaluator deadline exceeded at root/position boundary".into());
    }
    Ok(())
}

fn check_model(params: &Arc<ModelParameters>) -> Result<(), String> {
    let global = nnue::global_model().ok_or("native NNUE model is not loaded; material fallback forbidden")?;
    if !nnue::is_nnue_active() || !Arc::ptr_eq(params, &global) {
        return Err("native NNUE model differs from evaluator model".into());
    }
    Ok(())
}

// Checkmate takes priority over the engine's material/rule-50 draw policy.
// These are native mate-distance scores, never NNUE evaluations of terminals.
fn terminal_score(board: &Board, legal_count: usize, ply: i32) -> Option<i32> {
    if legal_count == 0 {
        Some(if board.in_check() { -Score::MATE + ply } else { 0 })
    } else if board.is_draw(ply as isize) {
        Some(0)
    } else {
        None
    }
}

struct NativeWorker {
    params: Arc<ModelParameters>,
    net: NetworkState,
    tt: TranspositionTable,
    history: HistoryTables,
    deadline: Instant,
}

impl NativeWorker {
    fn new(params: Arc<ModelParameters>, hash_mb: usize, deadline: Instant) -> Result<Self, String> {
        check_deadline(deadline)?;
        check_model(&params)?;
        let net = NetworkState::new(params.clone());
        Ok(Self {
            params,
            net,
            tt: TranspositionTable::new(hash_mb),
            history: HistoryTables::default(),
            deadline,
        })
    }

    fn search(&mut self, board: &Board, nodes: u64, work: &mut Work) -> Result<SearchResult, String> {
        check_deadline(self.deadline)?;
        check_model(&self.params)?;
        self.tt.clear(1);
        self.history.clear();
        // search_with_limits takes its NNUE state from this OS thread's TLS.
        // Reset it before every independent root, including repeated probes.
        let mut search_net = nnue::take_tls_nnue().ok_or("native search NNUE state unavailable")?;
        if !Arc::ptr_eq(&search_net.params, &self.params) {
            return Err("native search NNUE state has wrong model".into());
        }
        search_net.reset_history();
        nnue::put_tls_nnue(search_net);
        check_deadline(self.deadline)?;
        work.add(Work { searches: 1, ..Work::default() })?;
        let result = search_with_limits(board, Limits::Nodes(nodes), &self.tt, &mut self.history);
        work.add(Work { nodes: result.nodes, ..Work::default() })?;
        check_deadline(self.deadline)?;
        check_model(&self.params)?;
        if !(-Score::MATE..=Score::MATE).contains(&result.score) {
            return Err(format!("native search returned invalid score {}", result.score));
        }
        let mv = result.best_move.ok_or("native search returned no move for nonterminal root")?;
        if !board.generate_all_moves().iter().any(|&legal| legal == mv) {
            return Err("native search returned unknown/illegal move".into());
        }
        Ok(result)
    }

    fn components(&mut self, board: &Board) -> Result<(i64, i64, i64), String> {
        // Sibling positions can have the same ply and unrelated accumulators.
        self.net.reset_history();
        let (ev, psqt, draw) = self.net.evaluate_with_components(board, 0);
        if !(-Score::MATE..=Score::MATE).contains(&ev) {
            return Err(format!("native NNUE returned invalid score {ev}"));
        }
        Ok((i64::from(ev), i64::from(psqt), i64::from(draw)))
    }

    fn label(&mut self, sample: &Sample, nodes: u64, work: &mut Work) -> Result<LabeledPosition, String> {
        check_deadline(self.deadline)?;
        check_model(&self.params)?;
        validate_board(&sample.board)?;
        if !(-Score::MATE..=Score::MATE).contains(&sample.root_cp) {
            return Err("label root_cp outside native score range".into());
        }
        let board = &sample.board;
        let root_moves = board.generate_all_moves();
        if terminal_score(board, root_moves.len(), 0).is_some() {
            return Err("cannot label a terminal position".into());
        }
        self.net.reset_history();
        let mut moves = Vec::with_capacity(root_moves.len());
        let mut best_two_ply = i32::MIN;
        for &mv in root_moves.iter() {
            check_deadline(self.deadline)?;
            let mut child = board.clone();
            child.make_move(mv);
            let replies = child.generate_all_moves();
            if let Some(score) = terminal_score(&child, replies.len(), 1) {
                let q = -score;
                best_two_ply = best_two_ply.max(q);
                moves.push(MoveLabel {
                    mv,
                    obj_q_cp: q,
                    min_style_q_cp: q,
                    draw_permille: if q == 0 { 1000 } else { 0 },
                    def_difficulty_cp: 0,
                    terminal: true,
                });
                continue;
            }
            let (ev, psqt, child_draw) = self.components(&child)?;
            let q_static = -ev;
            let q_psqt_style = -psqt;
            let mut opponent_scores = Vec::with_capacity(replies.len());
            let mut shallow_index = 0;
            let mut shallow_impulse = i64::MIN;
            for (i, &reply) in replies.iter().enumerate() {
                check_deadline(self.deadline)?;
                // Deliberately retain the legacy material-based capture impulse
                // as a handcrafted ordering proxy, NOT a NNUE fallback score.
                let cap_gain = if reply.is_capture() {
                    i64::from(asura::evaluation::evaluate(&child))
                        + (child.type_on(reply.capture_sq()) as i64 + 1) * 120
                } else {
                    0
                };
                let mut grand = child.clone();
                grand.make_move(reply);
                let grand_moves = grand.generate_all_moves();
                let grand_ev = match terminal_score(&grand, grand_moves.len(), 2) {
                    Some(score) => i64::from(score),
                    None => self.components(&grand)?.0,
                };
                let opponent_q = -grand_ev;
                opponent_scores.push(opponent_q);
                let impulse = cap_gain + opponent_q / 4;
                if impulse > shallow_impulse {
                    shallow_impulse = impulse;
                    shallow_index = i;
                }
            }
            let shallow_top = opponent_scores[shallow_index];
            opponent_scores.sort_unstable_by(|a, b| b.cmp(a));
            let opponent_best = opponent_scores[0];
            let obj_q = (65 * -opponent_best + 35 * q_static) / 100;
            let viable = opponent_scores.iter().filter(|&&q| q >= opponent_best - 35).count();
            let pieces = child.occupancies().popcount() as i64;
            let draw = (480 - obj_q.abs() * 2 - pieces * 6 + child_draw * 2).clamp(25, 920);
            let difficulty = if (1..=3).contains(&viable) && draw < 750 && !mv.is_capture() {
                let gap = (opponent_best - shallow_top).max(0);
                let scarcity = (4 - viable as i64).max(1);
                let bonus = if viable == 1 && gap > 20 { 45 } else { 0 };
                scarcity * (gap + bonus) * (1000 - draw) / 3000
            } else {
                0
            };
            let style = obj_q.min(q_static).min((obj_q + q_psqt_style) / 2);
            let obj_q = i32::try_from(obj_q).map_err(|_| "label objective score overflow")?;
            best_two_ply = best_two_ply.max(obj_q);
            moves.push(MoveLabel {
                mv,
                obj_q_cp: obj_q,
                min_style_q_cp: i32::try_from(style).map_err(|_| "label style score overflow")?,
                draw_permille: draw as i32,
                def_difficulty_cp: difficulty.clamp(0, 400) as i32,
                terminal: false,
            });
        }
        // Selected-move refinement is a frozen heuristic, not oracle dominance.
        // Explicit move-index tie breaking preserves the root generator's order.
        let mut order: Vec<usize> = (0..moves.len()).collect();
        order.sort_unstable_by(|&a, &b| moves[b].obj_q_cp.cmp(&moves[a].obj_q_cp).then(a.cmp(&b)));
        let mut best_obj_q_cp = i32::MIN;
        for (rank, &index) in order.iter().enumerate() {
            check_deadline(self.deadline)?;
            let label = &mut moves[index];
            if !label.terminal
                && label.obj_q_cp.abs() < 25_000
                && (rank < 8 || i64::from(label.obj_q_cp) >= i64::from(best_two_ply) - 160)
            {
                let mut child = board.clone();
                child.make_move(label.mv);
                let result = self.search(&child, nodes, work)?;
                let refined = (-result.score).clamp(-4000, 4000);
                label.obj_q_cp = (75 * refined + 25 * label.obj_q_cp) / 100;
                label.min_style_q_cp = label.min_style_q_cp.min(label.obj_q_cp);
            }
            best_obj_q_cp = best_obj_q_cp.max(label.obj_q_cp);
        }
        check_deadline(self.deadline)?;
        Ok(LabeledPosition {
            board: board.clone(),
            root_cp: sample.root_cp,
            best_obj_q_cp,
            moves,
            model: self.params.clone(),
        })
    }

    fn evaluate(&mut self, pos: &LabeledPosition, budgets: [u64; 3], work: &mut Work) -> Result<Metrics, String> {
        let mut metrics = [[0i128; 6]; 3];
        for (rung, nodes) in budgets.into_iter().enumerate() {
            let result = self.search(&pos.board, nodes, work).map_err(|e| format!("rung {rung}: {e}"))?;
            let chosen = result.best_move.ok_or("native probe returned no move")?;
            let label =
                pos.moves.iter().find(|m| m.mv == chosen).ok_or("native probe move is missing from frozen labels")?;
            metrics[rung] = proxy_metrics(pos, label)?;
        }
        check_deadline(self.deadline)?;
        Ok(metrics)
    }
}

fn quantize(value: f64) -> Result<i128, String> {
    let scaled = (value * 1000.0).round();
    // i64::MAX as f64 rounds UP to 2^63; that endpoint must be excluded.
    if !scaled.is_finite() || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&scaled) {
        return Err("nonfinite or out-of-range quantized proxy metric".into());
    }
    Ok(i128::from(scaled as i64))
}

fn proxy_metrics(pos: &LabeledPosition, label: &MoveLabel) -> Result<[i128; 6], String> {
    let regret = (i64::from(pos.best_obj_q_cp) - i64::from(label.obj_q_cp)).max(0) as f64;
    let draw_proxy = f64::from(label.draw_permille) / 1000.0;
    let nonloss_proxy = 1.0 / (1.0 + (-f64::from(label.obj_q_cp) / 360.0).exp());
    let objective_win_proxy = (nonloss_proxy - 0.5 * draw_proxy).clamp(0.0, 1.0);
    let difficulty = f64::from(label.def_difficulty_cp);
    let slip_proxy = (difficulty / (difficulty + 180.0)).clamp(0.0, 0.55);
    let win_proxy = objective_win_proxy + (1.0 - objective_win_proxy) * 0.45 * slip_proxy;
    let practical_draw_proxy =
        draw_proxy * (1.0 - 0.45 * slip_proxy) + (1.0 - objective_win_proxy - draw_proxy).max(0.0) * 0.55 * slip_proxy;
    let excess_regret = (regret - 35.0).max(0.0);
    let soundness_penalty = -(excess_regret * excess_regret) / 25.0;
    let style_utility = if pos.root_cp >= 80 {
        (4.0 * win_proxy - practical_draw_proxy) * 1000.0 + 0.25 * soundness_penalty
    } else {
        (3.0 * win_proxy + practical_draw_proxy) * 1000.0 + 0.25 * soundness_penalty
    };
    let raw = [
        win_proxy * 1000.0,
        (win_proxy + practical_draw_proxy) * 1000.0,
        difficulty,
        f64::from(label.min_style_q_cp),
        soundness_penalty,
        style_utility,
    ];
    let mut result = [0i128; 6];
    for (i, value) in raw.into_iter().enumerate() {
        result[i] = quantize(value).map_err(|e| format!("metric {i}: {e}"))?;
    }
    Ok(result)
}

#[derive(Clone)]
enum Batch {
    Label(Arc<Vec<Sample>>, u64),
    Evaluate(Arc<Vec<LabeledPosition>>, [u64; 3]),
}

// Keep the fixed 288-byte metric payload inline on the hot probe path.
// Only the larger, infrequently constructed label needs another allocation.
#[allow(clippy::large_enum_variant)]
enum Output {
    Label(Box<LabeledPosition>),
    Metrics(Metrics),
}

struct Completion {
    result: Result<Output, String>,
    work: Work,
}

struct Job {
    batch: Batch,
    index: usize,
    // The parent retains ONLY the Receiver. A dead worker therefore cannot
    // leave recv blocked by a forgotten parent-side completion Sender.
    done: mpsc::Sender<Completion>,
}

struct Worker {
    jobs: Option<mpsc::SyncSender<Job>>,
    handle: Option<JoinHandle<()>>,
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else {
        "non-string panic payload".into()
    }
}

pub struct Evaluator {
    params: Arc<ModelParameters>,
    workers: Vec<Worker>,
    deadline: Instant,
    work: Work,
}

impl Evaluator {
    /// hash_mb is per worker; aggregate TT memory is capped at 4096 MiB.
    /// Spawn/initialization failure closes and joins every worker already owned.
    pub fn new(
        params: Arc<ModelParameters>, threads: usize, hash_mb: usize, deadline: Instant,
    ) -> Result<Self, String> {
        if threads == 0
            || threads > MAX_THREADS
            || hash_mb == 0
            || threads.checked_mul(hash_mb).is_none_or(|mb| mb > MAX_TOTAL_HASH_MB)
        {
            return Err(format!(
                "require 1..={MAX_THREADS} workers, positive hash_mb, total hash <= {MAX_TOTAL_HASH_MB} MiB"
            ));
        }
        check_deadline(deadline)?;
        check_model(&params)?;
        let mut evaluator = Self {
            params,
            workers: Vec::with_capacity(threads),
            deadline,
            work: Work::default(),
        };
        for index in 0..threads {
            let (jobs, inbox) = mpsc::sync_channel::<Job>(1);
            let (ready, initialized) = mpsc::channel();
            let params = evaluator.params.clone();
            let handle = thread::Builder::new()
                .name(format!("native-eval-{index}"))
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let state = catch_unwind(AssertUnwindSafe(|| NativeWorker::new(params, hash_mb, deadline)));
                    let mut state = match state {
                        Ok(Ok(state)) => state,
                        Ok(Err(error)) => {
                            let _ = ready.send(Err(error));
                            return;
                        }
                        Err(panic) => {
                            let _ = ready.send(Err(format!("worker initialization panic: {}", panic_text(panic))));
                            return;
                        }
                    };
                    if ready.send(Ok(())).is_err() {
                        return;
                    }
                    drop(ready);
                    while let Ok(job) = inbox.recv() {
                        let mut work = Work::default();
                        let outcome = catch_unwind(AssertUnwindSafe(|| match &job.batch {
                            Batch::Label(samples, nodes) => state
                                .label(&samples[job.index], *nodes, &mut work)
                                .map(|label| Output::Label(Box::new(label))),
                            Batch::Evaluate(positions, budgets) => {
                                state.evaluate(&positions[job.index], *budgets, &mut work).map(Output::Metrics)
                            }
                        }));
                        let (result, panicked) = match outcome {
                            Ok(result) => (result, false),
                            Err(panic) => (Err(format!("native worker panic: {}", panic_text(panic))), true),
                        };
                        if job.done.send(Completion { result, work }).is_err() || panicked {
                            break; // Never reuse potentially poisoned native state.
                        }
                    }
                })
                .map_err(|e| format!("spawn native worker {index}: {e}"))?;
            evaluator.workers.push(Worker { jobs: Some(jobs), handle: Some(handle) });
            initialized.recv().map_err(|e| format!("initialize native worker {index}: {e}"))??;
        }
        check_deadline(deadline)?;
        Ok(evaluator)
    }

    pub fn label(&mut self, samples: Vec<Sample>, label_nodes: u64) -> Result<Vec<LabeledPosition>, String> {
        check_cases(samples.len())?;
        check_nodes(label_nodes)?;
        let count = samples.len();
        let outputs = self.run_batch(Batch::Label(Arc::new(samples), label_nodes), count)?;
        let labels = outputs
            .into_iter()
            .map(|output| match output {
                Output::Label(label) => Ok(*label),
                Output::Metrics(_) => Err("worker returned metrics for a label job".into()),
            })
            .collect::<Result<Vec<_>, String>>()?;
        check_deadline(self.deadline)?;
        Ok(labels)
    }

    pub fn evaluate(&mut self, panel: &Panel, budgets: [u64; 3]) -> Result<Totals, String> {
        check_cases(panel.len())?;
        for nodes in budgets {
            check_nodes(nodes)?;
        }
        if !Arc::ptr_eq(&panel.positions[0].model, &self.params) {
            return Err("panel labels belong to a different model".into());
        }
        let outputs = self.run_batch(Batch::Evaluate(panel.positions.clone(), budgets), panel.len())?;
        let mut totals = Totals { sums: [[0; 6]; 3], cases: panel.len() };
        // run_batch returns in position-index order, never completion order.
        for output in outputs {
            let Output::Metrics(metrics) = output else {
                return Err("worker returned label for a probe job".into());
            };
            for (sum_row, row) in totals.sums.iter_mut().zip(metrics) {
                for (sum, value) in sum_row.iter_mut().zip(row) {
                    *sum = sum.checked_add(value).ok_or("proxy metric reduction overflow")?;
                }
            }
        }
        totals.utility()?;
        check_deadline(self.deadline)?;
        Ok(totals)
    }

    pub fn work(&self) -> Work {
        self.work
    }

    fn run_batch(&mut self, batch: Batch, count: usize) -> Result<Vec<Output>, String> {
        check_deadline(self.deadline)?;
        check_model(&self.params)?;
        self.work.add(Work { batches: 1, ..Work::default() })?;
        let mut outputs = Vec::with_capacity(count);
        // Bounded waves: at most one outstanding job per persistent OS worker.
        // On error drain the entire wave before returning; dispatch no new wave.
        for start in (0..count).step_by(self.workers.len()) {
            let mut pending = Vec::with_capacity(self.workers.len());
            let mut error: Option<(usize, String)> = None;
            for (worker_index, index) in (start..count).take(self.workers.len()).enumerate() {
                let (done, receiver) = mpsc::channel();
                let job = Job { batch: batch.clone(), index, done };
                let sender = self.workers[worker_index].jobs.as_ref().ok_or("worker job channel is closed");
                let sent = match sender {
                    Ok(sender) => sender.send(job).map_err(|_| "worker job channel disconnected"),
                    Err(e) => Err(e),
                };
                if let Err(e) = sent {
                    error = Some((index, e.into()));
                    break;
                }
                pending.push((index, receiver));
            }
            for (index, receiver) in pending {
                let result = match receiver.recv() {
                    Ok(completion) => {
                        let accounting = self.work.add(completion.work);
                        match completion.result {
                            Err(e) => Err(e),
                            Ok(output) => accounting.map(|()| output),
                        }
                    }
                    Err(e) => Err(format!("worker completion disconnected: {e}")),
                };
                match result {
                    Ok(output) => outputs.push(output),
                    Err(e) => {
                        if error.as_ref().is_none_or(|(earlier, _)| index < *earlier) {
                            error = Some((index, e));
                        }
                    }
                }
            }
            if let Some((index, error)) = error {
                return Err(format!("position {index}: {error}"));
            }
            check_deadline(self.deadline)?;
        }
        if outputs.len() != count {
            return Err("worker batch did not return every position".into());
        }
        check_model(&self.params)?;
        Ok(outputs)
    }
}

#[cfg(test)]
#[path = "evaluator/tests.rs"]
mod tests;

impl Drop for Evaluator {
    fn drop(&mut self) {
        // Close ALL queues first, then join ALL workers, even when one panicked.
        // No borrowed job payloads or detached native searches survive Drop.
        for worker in &mut self.workers {
            worker.jobs.take();
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
