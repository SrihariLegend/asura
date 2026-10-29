use std::{cell::RefCell, sync::OnceLock};

pub use crate::thread::PvLine;
use crate::{
    board::Board,
    evaluation::evaluate,
    history::HistoryTables,
    movepick::{MainMovePicker, MainStage, QuiescenceMovePicker, creates_threat_on_board},
    parameters::*,
    stack::Stack,
    thread::{AspirationKind, AspirationWindow, RootMove, RootMoves, ThreadData},
    transposition::{Bound, Entry as TtEntry, TranspositionTable, TtDepth},
    types::{MAX_PLY as MAX_PLY_USIZE, Move, Piece, PieceType, Score, Square, is_decisive, is_win},
};

pub const MATE_SCORE: i32 = Score::MATE;
const MAX_PLY: u16 = MAX_PLY_USIZE as u16;
const RECAPTURE_ONLY_DEPTH: i16 = -4;
pub const MATE_THRESHOLD: i32 = Score::MATE_IN_MAX;
const INFINITY: i32 = Score::INFINITE;

// Compile-time search configuration flags.
const CFG_DRAW_JITTER: u32 = 1 << 0;
const CFG_SELECTIVE: u32 = 1 << 1;
const CFG_LMR: u32 = 1 << 2;
const CFG_NMP: u32 = 1 << 3;
const CFG_RFP: u32 = 1 << 4;
const CFG_PROBCUT: u32 = 1 << 5;
const CFG_SEE: u32 = 1 << 6;
const CFG_LMP: u32 = 1 << 7;
const CFG_FP: u32 = 1 << 8;
const CFG_HP: u32 = 1 << 9;
const CFG_IIR: u32 = 1 << 10;
const CFG_SE: u32 = 1 << 11;
const CFG_RAZOR: u32 = 1 << 12;
const CFG_CHECKED_TT_CUT: u32 = 1 << 13;
const CFG_LQC: u32 = 1 << 14;
const CFG_QS_HP: u32 = 1 << 15;
const CFG_QS_FP: u32 = 1 << 16;
const CFG_QS_LMP: u32 = 1 << 17;
const CFG_QS_EARLY_CUT: u32 = 1 << 18;
const CFG_QS_SEE: u32 = 1 << 19;
const CFG_ROOT_IIR: u32 = 1 << 20;
const CFG_QS_STANDPAT: u32 = 1 << 21;
const CFG_QS_BOUND_AVG: u32 = 1 << 22;
const CFG_HINDSIGHT: u32 = 1 << 23;
const CFG_ALL_SELECTIVE: u32 = CFG_SELECTIVE
    | CFG_LMR
    | CFG_NMP
    | CFG_RFP
    | CFG_PROBCUT
    | CFG_SEE
    | CFG_LMP
    | CFG_FP
    | CFG_HP
    | CFG_IIR
    | CFG_SE
    | CFG_RAZOR
    | CFG_CHECKED_TT_CUT
    | CFG_LQC
    | CFG_QS_HP
    | CFG_QS_FP
    | CFG_QS_LMP
    | CFG_QS_EARLY_CUT
    | CFG_QS_SEE
    | CFG_ROOT_IIR
    | CFG_QS_STANDPAT
    | CFG_QS_BOUND_AVG
    | CFG_HINDSIGHT;
const CFG_STANDARD: u32 = CFG_DRAW_JITTER | CFG_ALL_SELECTIVE;

thread_local! {
    static LOCAL_TT: RefCell<TranspositionTable> = RefCell::new(TranspositionTable::new(4));
    static LOCAL_HISTORY: RefCell<HistoryTables> = RefCell::new(HistoryTables::default());
}

/// Base LMR reduction term in 1_000 = 1 ply fixed-point units: round(400 * ln(depth) * ln(move_count)).
pub fn lmr_table_term(depth: u8, move_count: usize) -> i32 {
    static TABLE: OnceLock<[[i32; 128]; 128]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [[0; 128]; 128];
        for (d, row) in t.iter_mut().enumerate().skip(1) {
            for (m, cell) in row.iter_mut().enumerate().skip(1) {
                *cell = (400.0 * (d as f64).ln() * (m as f64).ln()).round() as i32;
            }
        }
        t
    });
    table[(depth as usize).clamp(1, 127)][move_count.clamp(1, 127)]
}

#[allow(clippy::too_many_arguments)]
fn compute_node_bias(
    pv_node: bool, in_check: bool, cut_node: bool, tt_pv: bool, improving: bool, tt_capture: bool,
    parent_reduction: i32, ply: u16, last_critical_ply: u16, laterality: i32,
) -> i32 {
    let mut bias = if pv_node { -500 } else { 2_000 * cut_node as i32 - 1_000 * tt_pv as i32 };
    bias -= 1_000 * in_check as i32;
    bias += 1_000 * (!improving) as i32;
    bias += 500 * tt_capture as i32;
    bias += 200 * parent_reduction / LMR_SCALE;
    if !pv_node {
        let chain = ply.saturating_sub(last_critical_ply).min(8) as i32;
        bias -= 125 * chain;
        bias += 30 * laterality;
    }
    bias
}

const fn lmp_limit(depth: u8, improving: bool) -> usize {
    let d = depth as usize;
    3 + d * d + (improving as usize) * (1 + d * d / 2)
}

const fn plan_scout_retry(best_score: i32, score: i32, child_depth: u8, reduced_depth: u8) -> u8 {
    let mut retry_depth = child_depth;
    let margin_deeper = 60 + 10 * (child_depth.saturating_sub(reduced_depth) as i32);
    if score > best_score.saturating_add(margin_deeper) {
        retry_depth = retry_depth.saturating_add(1);
    } else if score <= best_score.saturating_add(child_depth as i32) && retry_depth > 1 {
        retry_depth -= 1;
    }
    retry_depth
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SearchResult {
    /// No move is selected at depth zero or when no legal move exists.
    /// A drawn root returns its first legal move at positive depth.
    pub best_move: Option<Move>,
    /// Root-side score: centipawns, or +/- (32_000 - plies to checkmate).
    /// Normal search jitters draws in -2..=2; zero-draw mode scores them zero.
    /// A checkmated root scores -32_000.
    pub score: i32,
    /// Visited nodes (moves made during search).
    pub nodes: u64,
    pv: PvLine,
}

impl SearchResult {
    /// Principal variation from the last completed root search.
    pub fn pv(&self) -> &[Move] {
        self.pv.as_slice()
    }
}

/// Fixed-depth alpha-beta search over legal moves with a quiescence tail.
/// Positive depth means plies to search before dropping into quiescence.
/// Depth zero evaluates the root directly when not in check and drops into
/// quiescence when in check; material draws and mates still apply at all
/// depths and quiescence leaves.
pub fn search(position: &Board, depth: u8) -> SearchResult {
    LOCAL_TT.with_borrow_mut(|tt| {
        LOCAL_HISTORY.with_borrow_mut(|history| {
            tt.clear(1);
            history.clear();
            run_search::<CFG_STANDARD>(position, depth, Some(tt), Some(history))
        })
    })
}

/// Fixed-depth search using an caller-provided transposition table.
pub fn search_with_tt(position: &Board, depth: u8, tt: &TranspositionTable) -> SearchResult {
    LOCAL_HISTORY.with_borrow_mut(|history| {
        history.clear();
        run_search::<CFG_STANDARD>(position, depth, Some(tt), Some(history))
    })
}

/// Fixed-depth search using caller-provided transposition and history tables.
pub fn search_with_context(
    position: &Board, depth: u8, tt: &TranspositionTable, history: &mut HistoryTables,
) -> SearchResult {
    run_search::<CFG_STANDARD>(position, depth, Some(tt), Some(history))
}

macro_rules! define_diagnostic_searches {
    ($($(#[$attr:meta])* $name:ident: $cfg:expr;)*) => {
        $(
            $(#[$attr])*
            pub fn $name(position: &Board, depth: u8) -> SearchResult {
                LOCAL_TT.with_borrow_mut(|tt| {
                    LOCAL_HISTORY.with_borrow_mut(|history| {
                        tt.clear(1);
                        history.clear();
                        run_search::<{ $cfg }>(position, depth, Some(tt), Some(history))
                    })
                })
            }
        )*
    };
}

define_diagnostic_searches! {
    /// The same selective search with recognized draws scored zero instead of jittered.
    search_with_zero_draws: CFG_ALL_SELECTIVE;
    /// The selective search with LMR disabled and recognized draws scored zero.
    search_without_lmr: CFG_ALL_SELECTIVE & !CFG_LMR;
    /// The selective search with NMP disabled and recognized draws scored zero.
    search_without_nmp: CFG_ALL_SELECTIVE & !CFG_NMP;
    /// The selective search with RFP disabled and recognized draws scored zero.
    search_without_rfp: CFG_ALL_SELECTIVE & !CFG_RFP;
    /// The selective search with ProbCut disabled and recognized draws scored zero.
    search_without_probcut: CFG_ALL_SELECTIVE & !CFG_PROBCUT;
    /// The selective search with main-loop SEE pruning disabled and recognized draws scored zero.
    search_without_see_pruning: CFG_ALL_SELECTIVE & !CFG_SEE;
    /// The selective search with Late Move Pruning (LMP) disabled and recognized draws scored zero.
    search_without_lmp: CFG_ALL_SELECTIVE & !CFG_LMP;
    /// The selective search with main-loop Futility Pruning (FP) disabled and recognized draws scored zero.
    search_without_fp: CFG_ALL_SELECTIVE & !CFG_FP;
    /// The selective search with Quiet-History Pruning (HP) disabled and recognized draws scored zero.
    search_without_hp: CFG_ALL_SELECTIVE & !CFG_HP;
    /// The selective search with cut-node Internal Iterative Reduction (IIR) disabled and recognized draws scored zero.
    search_without_iir: CFG_ALL_SELECTIVE & !CFG_IIR;
    /// The selective search with Singular Extensions (SE) disabled and recognized draws scored zero.
    search_without_se: CFG_ALL_SELECTIVE & !CFG_SE;
    /// The selective search with Razoring disabled and recognized draws scored zero.
    search_without_razoring: CFG_ALL_SELECTIVE & !CFG_RAZOR;
    /// The selective search with checked TT-capture cutoff disabled and recognized draws scored zero.
    search_without_checked_tt_cutoff: CFG_ALL_SELECTIVE & !CFG_CHECKED_TT_CUT;
    /// The selective search with post-move late-quiet closure disabled and recognized draws scored zero.
    search_without_late_quiet_closure: CFG_ALL_SELECTIVE & !CFG_LQC;
    /// The selective search with quiescence quiet-history pruning disabled and recognized draws scored zero.
    search_without_qs_history_pruning: CFG_ALL_SELECTIVE & !CFG_QS_HP;
    /// The selective search with quiescence capture-futility pruning disabled and recognized draws scored zero.
    search_without_qs_futility_pruning: CFG_ALL_SELECTIVE & !CFG_QS_FP;
    /// The selective search with quiescence late-capture pruning disabled and recognized draws scored zero.
    search_without_qs_late_capture_pruning: CFG_ALL_SELECTIVE & !CFG_QS_LMP;
    /// The selective search with quiescence history-backed early fail-high disabled and recognized draws scored zero.
    search_without_qs_early_cutoff: CFG_ALL_SELECTIVE & !CFG_QS_EARLY_CUT;
    /// The selective search with quiescence SEE pruning disabled and recognized draws scored zero.
    search_without_qs_see_pruning: CFG_ALL_SELECTIVE & !CFG_QS_SEE;
    /// The selective search with root and PV Internal Iterative Reduction (IIR) disabled and recognized draws scored zero.
    search_without_root_iir: CFG_ALL_SELECTIVE & !CFG_ROOT_IIR;
    /// The selective search with quiescence stand-pat fail-high handling disabled and recognized draws scored zero.
    search_without_qs_standpat_handling: CFG_ALL_SELECTIVE & !CFG_QS_STANDPAT;
    /// The selective search with quiescence post-loop bound-score averaging disabled and recognized draws scored zero.
    search_without_qs_bound_averaging: CFG_ALL_SELECTIVE & !CFG_QS_BOUND_AVG;
    /// The selective search with hindsight depth adjustment disabled and recognized draws scored zero.
    search_without_hindsight: CFG_ALL_SELECTIVE & !CFG_HINDSIGHT;
}

/// Diagnostic search with zero draw scores and no selective QS pruning or TT/history/LMR/NMP/RFP/ProbCut/SEE/LMP/FP/HP/IIR/SE/Razoring/CheckedTtCut.
/// Keeps every evasion and capture eligible for independent minimax comparisons.
pub fn search_reference(position: &Board, depth: u8) -> SearchResult {
    run_search::<0>(position, depth, None, None)
}

fn run_search<const CFG: u32>(
    position: &Board, depth: u8, tt: Option<&TranspositionTable>, history: Option<&mut HistoryTables>,
) -> SearchResult {
    let mut td = ThreadData::new(position, tt, history);
    let mut pv = PvLine::new();

    let selective = (CFG & CFG_SELECTIVE) != 0;
    let (score, best_move) = if !selective || depth == 0 {
        let (score, best_move) = search_node::<CFG, true>(&mut td, -INFINITY, INFINITY, depth, false, 0, 0);
        if let Some(mv) = best_move {
            let line = td.pv_table.line(0);
            if line.is_empty() {
                pv.update(mv, &[]);
            } else {
                pv.update(line[0], &line[1..]);
            }
        }
        (score, best_move)
    } else {
        run_iterative_search::<CFG>(&mut td, depth, &mut pv)
    };

    debug_assert!(td.board == *position, "search did not undo every move");
    SearchResult { best_move, score, nodes: td.nodes, pv }
}

fn run_iterative_search<const CFG: u32>(
    td: &mut ThreadData<'_>, max_depth: u8, pv: &mut PvLine,
) -> (i32, Option<Move>) {
    let draw_jitter = (CFG & CFG_DRAW_JITTER) != 0;
    td.root_moves = RootMoves::new(&td.board);
    if td.root_moves.is_empty() {
        let score = if td.board.in_check() { -MATE_SCORE } else { draw_score(draw_jitter, td.nodes) };
        return (score, None);
    }
    if td.board.is_draw(0) {
        let mv = td.root_moves.at(0).mv;
        pv.update(mv, &[]);
        return (draw_score(draw_jitter, td.nodes), Some(mv));
    }

    let mut eval_stability: i32 = 0;
    let mut pv_stability: i32 = 0;
    let mut last_best_rootmove: Option<RootMove> = None;

    for step in 1..=max_depth {
        td.root_moves.begin_depth_iteration(step);

        let pv_idx = 0;
        let prev_score = td.root_moves.aspiration_previous_score(pv_idx);
        let mut window = AspirationWindow::new(step, prev_score, eval_stability, pv_stability);
        let mut aspiration_depth = step;
        let mut root_pv_move = td.root_moves.at(pv_idx).mv;

        loop {
            td.stack = Stack::default();
            let root_effective_depth = aspiration_depth.max(1);
            let score = search_root::<CFG>(td, window.alpha, window.beta, root_effective_depth, pv_idx, root_pv_move);
            td.root_moves.sort_from(pv_idx);

            match window.on_result(score) {
                AspirationKind::FailHigh => {
                    if !is_decisive(score) && aspiration_depth > 3 {
                        aspiration_depth -= 1;
                    }
                    let winning_move = td.root_moves.at(pv_idx).mv;
                    if winning_move.is_present() {
                        root_pv_move = winning_move;
                    }
                }
                AspirationKind::FailLow => {
                    aspiration_depth = step;
                }
                AspirationKind::Exact => break,
            }
        }

        let top = *td.root_moves.at(0);
        if (top.current.score - top.average).abs() < EVAL_STABILITY_THRESHOLD {
            eval_stability += 1;
        } else {
            eval_stability = 0;
        }

        if last_best_rootmove.is_some_and(|last| last.mv == top.mv) {
            pv_stability += 1;
        } else {
            pv_stability = 0;
        }

        // Forgotten-mate rollback: never replace a proven mate from the prior
        // completed iteration with a weaker or incomplete bound.
        let is_forgotten_mate = last_best_rootmove.is_some_and(|last| {
            let last_score = last.current.score;
            last_score != -INFINITY
                && is_decisive(last_score)
                && (top.current.score.abs() < last_score.abs() || top.current.upperbound || top.current.lowerbound)
        });

        if is_forgotten_mate {
            if let Some(last) = last_best_rootmove
                && let Some(pos) = td.root_moves.records.iter().position(|rm| rm.mv == last.mv)
            {
                td.root_moves.records.remove(pos);
                td.root_moves.records.insert(0, last);
                td.root_moves.records[0].current.upperbound = false;
                td.root_moves.records[0].current.lowerbound = false;
            }
        } else {
            last_best_rootmove = Some(top);
        }
    }

    let winner = td.root_moves.at(0);
    let result = winner.active_result();
    *pv = result.pv;
    (result.score, Some(winner.mv))
}

#[inline]
fn reserves_tt_verification_ply(depth: u8, has_move: bool, entry: Option<TtEntry>) -> bool {
    depth >= IIR_MIN_DEPTH && (!has_move || entry.is_none_or(|e| e.depth + IIR_TT_DEPTH_GAP < depth as i32))
}

fn search_root<const CFG: u32>(
    td: &mut ThreadData<'_>, mut alpha: i32, beta: i32, mut root_effective_depth: u8, pv_idx: usize, work_item: Move,
) -> i32 {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    let selective = (CFG & CFG_SELECTIVE) != 0;
    let use_root_iir = (CFG & CFG_ROOT_IIR) != 0;
    let hash = td.board.hash();
    let entry = td.tt.and_then(|t| t.read(hash, td.board.fiftymove_clock(), 0));
    let tt_move = entry.map(|e| e.mv).filter(|&mv| mv.is_present() && td.board.is_legal(mv)).unwrap_or(Move::NULL);
    let work_item = if work_item.is_present() && td.board.is_legal(work_item) { work_item } else { Move::NULL };
    let priority_move = if work_item.is_present() { work_item } else { tt_move };

    let in_check = td.board.in_check();
    let raw_eval = if in_check {
        TtDepth::NONE_SCORE
    } else if let Some(entry) = &entry
        && entry.raw_eval != TtDepth::NONE_SCORE
    {
        entry.raw_eval
    } else {
        let ev = evaluate(&td.board);
        if entry.is_none()
            && let Some(t) = td.tt
        {
            t.write(hash, TtDepth::EVAL_ONLY, ev, TtDepth::NONE_SCORE, Bound::None, Move::NULL, 0, true, false);
        }
        ev
    };

    td.stack[0].static_eval = raw_eval;
    td.stack[0].reduction = 0;
    td.stack[0].laterality = 0;
    td.stack[2].cutoff_count = 0;
    if let Some(h) = td.history.as_deref_mut() {
        h.clear_child_killers(0);
    }

    if selective && use_root_iir && reserves_tt_verification_ply(root_effective_depth, work_item.is_present(), entry) {
        root_effective_depth -= 1;
    }

    let node_bias = compute_node_bias(true, in_check, false, true, false, priority_move.is_capture(), 0, 0, 0, 0);

    let mut picker = MainMovePicker::new(priority_move, true, [Move::NULL; 2], Move::NULL);
    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let mut bound = Bound::Upper;
    let mut move_count = 0usize;

    let mut quiet_moves = [Move::NULL; 32];
    let mut quiet_len = 0usize;
    let mut noisy_moves = [Move::NULL; 32];
    let mut noisy_len = 0usize;

    while let Some(mv) = picker.next(&td.board, td.history.as_deref(), false, 0) {
        if !td.root_moves.is_eligible_at_root(pv_idx, mv) {
            continue;
        }
        move_count += 1;
        let child_depth = root_effective_depth - 1;
        let nodes_before = td.nodes;
        td.sel_depth = 1;
        td.pv_table.clear(1);

        let moved_piece = td.board.moved_piece(mv);
        let is_quiet = mv.is_quiet();
        let creates_threat = is_quiet && creates_threat_on_board(&td.board, mv);
        let hist = td.history.as_deref().map_or(0, |h| {
            if is_quiet {
                h.quiet_history.get(td.board.all_threats(), creates_threat, td.board.side_to_move(), mv)
                    + h.conthist(0, 1, moved_piece, mv.to())
                    + h.conthist(0, 2, moved_piece, mv.to())
            } else {
                h.noisy_history.get(td.board.all_threats(), moved_piece, mv.to(), td.board.type_on(mv.capture_sq()))
            }
        });

        if let Some(h) = td.history.as_deref_mut() {
            h.continuation_history.remember(0, in_check, mv, moved_piece);
        }
        td.stack[0].mv = mv;
        td.stack[0].move_count = move_count as u16;
        td.stack[0].stat_score = hist;
        td.stack[0].laterality = if move_count > 1 { (move_count.ilog2() as i32 - 1).max(0) } else { 0 };
        td.make_move(mv);
        let gives_check = td.board.in_check();

        let use_lmr = (CFG & CFG_LMR) != 0;
        let score = if move_count == 1 {
            td.stack[0].reduction = 0;
            -search_node::<CFG, true>(td, -beta, -alpha, child_depth, false, 1, 0).0
        } else {
            // At PV/Root nodes, LMR applies only to quiet moves.
            let mut reduced_depth = child_depth;
            let mut quotient = 0;
            if use_lmr && is_quiet && root_effective_depth >= 2 && child_depth >= 1 {
                let mut raw_reduction = lmr_table_term(root_effective_depth, move_count) + node_bias
                    - hist / 10
                    - LMR_CHECK_BONUS * gives_check as i32
                    - LMR_THREAT_BONUS * creates_threat as i32
                    + LMR_EXACT_BOUND_PENALTY * (bound == Bound::Exact) as i32
                    + LMR_CHILD_CUTOFF_PENALTY * (td.stack[1].cutoff_count > 2) as i32;
                if raw_reduction < 0 {
                    raw_reduction = 0;
                }
                quotient = raw_reduction / LMR_SCALE;
                let reduction_plies = quotient.min(child_depth as i32 - 1).max(0) as u8;
                reduced_depth = child_depth - reduction_plies;
            }

            td.stack[0].reduction = (child_depth as i32 - reduced_depth as i32) * LMR_SCALE;
            let mut scout = -search_node::<CFG, false>(td, -alpha - 1, -alpha, reduced_depth, true, 1, 1).0;
            td.stack[0].reduction = 0;

            let mut retry_depth = child_depth;
            if quotient > 0 && scout > alpha && reduced_depth < child_depth {
                let margin_deeper = 60 + 10 * (child_depth.saturating_sub(reduced_depth) as i32);
                if scout > best_score.saturating_add(margin_deeper) {
                    retry_depth = child_depth.saturating_add(1);
                }
                if retry_depth > reduced_depth {
                    td.stack[0].reduction = (child_depth as i32 - retry_depth as i32) * LMR_SCALE;
                    scout = -search_node::<CFG, false>(td, -alpha - 1, -alpha, retry_depth, true, 1, 1).0;
                    td.stack[0].reduction = 0;
                }
            }

            if scout > alpha && beta > alpha + 1 {
                td.stack[0].reduction = (child_depth as i32 - retry_depth as i32) * LMR_SCALE;
                let res = -search_node::<CFG, true>(td, -beta, -alpha, retry_depth, false, 1, 0).0;
                td.stack[0].reduction = 0;
                res
            } else {
                scout
            }
        };

        td.undo_move(mv);

        let move_nodes = td.nodes - nodes_before;
        let sel_depth = td.sel_depth;
        let mut child_pv = PvLine::new();
        let line = td.pv_table.line(1);
        if !line.is_empty() {
            child_pv.update(line[0], &line[1..]);
        }
        td.root_moves.record_searched_move(
            pv_idx,
            mv,
            score,
            alpha,
            beta,
            move_count == 1,
            move_nodes,
            sel_depth,
            child_pv.as_slice(),
        );

        if score > best_score {
            best_score = score;
            if score > alpha {
                best_move = mv;
                bound = Bound::Exact;
                if score >= beta {
                    bound = Bound::Lower;
                    break;
                }
                alpha = score;
                if pv_idx == 0
                    && mv != priority_move
                    && let Some(t) = td.tt
                {
                    t.write(hash, root_effective_depth as i32, raw_eval, score, Bound::Lower, mv, 0, true, false);
                }
            }
        }

        if mv.is_quiet() && quiet_len < 32 {
            quiet_moves[quiet_len] = mv;
            quiet_len += 1;
        } else if mv.is_noisy() && noisy_len < 32 {
            noisy_moves[noisy_len] = mv;
            noisy_len += 1;
        }
    }

    if best_score >= beta && best_move.is_present() {
        update_histories_on_best_move(
            td,
            0,
            root_effective_depth as i32,
            false,
            best_move,
            best_score,
            beta,
            &quiet_moves[..quiet_len],
            &noisy_moves[..noisy_len],
        );
    }

    if pv_idx == 0
        && let Some(t) = td.tt
    {
        let store_mv = if bound == Bound::Upper { Move::NULL } else { best_move };
        let store_depth = (root_effective_depth as i32 - i32::from(bound == Bound::Lower)).max(1);
        t.write(hash, store_depth, raw_eval, best_score, bound, store_mv, 0, true, true);
    }

    best_score
}

fn search_node<const CFG: u32, const PV: bool>(
    td: &mut ThreadData<'_>, mut alpha: i32, beta: i32, mut depth: u8, cut_node: bool, ply: u16, last_critical_ply: u16,
) -> (i32, Option<Move>) {
    let draw_jitter = (CFG & CFG_DRAW_JITTER) != 0;
    let selective = (CFG & CFG_SELECTIVE) != 0;
    let use_lmr = (CFG & CFG_LMR) != 0;
    let use_nmp = (CFG & CFG_NMP) != 0;
    let use_rfp = (CFG & CFG_RFP) != 0;
    let use_probcut = (CFG & CFG_PROBCUT) != 0;
    let use_see = (CFG & CFG_SEE) != 0;
    let use_lmp = (CFG & CFG_LMP) != 0;
    let use_fp = (CFG & CFG_FP) != 0;
    let use_hp = (CFG & CFG_HP) != 0;
    let use_iir = (CFG & CFG_IIR) != 0;
    let use_root_iir = (CFG & CFG_ROOT_IIR) != 0;
    let use_se = (CFG & CFG_SE) != 0;
    let use_razoring = (CFG & CFG_RAZOR) != 0;
    let use_checked_tt_cut = (CFG & CFG_CHECKED_TT_CUT) != 0;
    let use_lqc = (CFG & CFG_LQC) != 0;
    let use_hindsight = (CFG & CFG_HINDSIGHT) != 0;

    let ply_idx = ply as usize;
    let prior_move = if ply_idx > 0 { td.stack[ply_idx - 1].mv } else { Move::NULL };
    let previous = prior_move.is_present().then(|| prior_move.to());
    let parent_eval = if ply_idx > 0 { td.stack[ply_idx - 1].static_eval } else { TtDepth::NONE_SCORE };
    let parent_move_count = if ply_idx > 0 { td.stack[ply_idx - 1].move_count as usize } else { 0 };

    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    if PV {
        td.pv_table.clear(ply_idx);
        td.sel_depth = td.sel_depth.max(ply);
    }

    if depth == 0 {
        if ply == 0 && !td.board.has_legal_moves() {
            let score = if td.board.in_check() { -MATE_SCORE } else { draw_score(draw_jitter, td.nodes) };
            return (score, None);
        }
        return (quiescence::<CFG, PV>(td, alpha, beta, 0, ply, previous).0, None);
    }

    let draw = draw_score(draw_jitter, td.nodes);
    if ply > 0 && alpha < draw && td.board.upcoming_repetition(ply_idx) {
        alpha = draw;
        if alpha >= beta {
            return (alpha, None);
        }
    }

    let in_check = td.board.in_check();
    if ply > 0 {
        if td.board.is_draw(ply as isize) {
            if in_check && !td.board.has_legal_moves() {
                return (-MATE_SCORE + i32::from(ply), None);
            }
            return (draw, None);
        }
        if ply >= MAX_PLY {
            if in_check && !td.board.has_legal_moves() {
                return (-MATE_SCORE + i32::from(ply), None);
            }
            return (if in_check { draw } else { evaluate(&td.board) }, None);
        }

        // Mate Distance Pruning (MDP)
        alpha = alpha.max(-MATE_SCORE + i32::from(ply));
        let mate_beta = beta.min(MATE_SCORE - i32::from(ply) - 1);
        if alpha >= mate_beta {
            return (alpha, None);
        }
    }

    let excluded_move = td.stack[ply_idx].excluded_move;
    let in_singular = excluded_move.is_present();

    let hash = td.board.hash();
    let entry = if selective && !in_singular {
        td.tt.and_then(|t| t.read(hash, td.board.fiftymove_clock(), ply as isize))
    } else {
        None
    };
    let mut tt_move = Move::NULL;
    let mut tt_pv = PV;

    if let Some(entry) = &entry {
        if entry.mv.is_present() && td.board.is_legal(entry.mv) {
            tt_move = entry.mv;
        }
        tt_pv |= entry.tt_pv;

        if !PV
            && ply > 0
            && entry.depth >= depth as i32
            && entry.score != TtDepth::NONE_SCORE
            && match entry.bound {
                Bound::Exact => true,
                Bound::Lower => entry.score >= beta && (cut_node || depth > 5),
                Bound::Upper => entry.score <= alpha && (!cut_node || depth > 5),
                Bound::None => false,
            }
        {
            if tt_move.is_quiet()
                && entry.score >= beta
                && parent_move_count < 4
                && let Some(h) = td.history.as_deref_mut()
            {
                let d = depth as i32;
                let quiet_bonus = (180 * d - 70).min(1_600);
                let cont_bonus = (100 * d - 70).min(1_100);
                let creates_threat = creates_threat_on_board(&td.board, tt_move);
                h.quiet_history.update(
                    td.board.all_threats(),
                    creates_threat,
                    td.board.side_to_move(),
                    tt_move,
                    quiet_bonus,
                );
                h.update_continuation_histories(ply_idx, td.board.moved_piece(tt_move), tt_move.to(), cont_bonus);
            }

            if td.board.fiftymove_clock() < 90 {
                return (entry.score, Some(tt_move));
            }
        }
    }

    let raw_eval = if in_check {
        TtDepth::NONE_SCORE
    } else if in_singular {
        td.stack[ply_idx].static_eval
    } else if let Some(entry) = &entry
        && entry.raw_eval != TtDepth::NONE_SCORE
    {
        entry.raw_eval
    } else {
        let ev = evaluate(&td.board);
        if selective
            && entry.is_none()
            && let Some(t) = td.tt
        {
            t.write(
                hash,
                TtDepth::EVAL_ONLY,
                ev,
                TtDepth::NONE_SCORE,
                Bound::None,
                Move::NULL,
                ply as isize,
                tt_pv,
                false,
            );
        }
        ev
    };

    td.stack[ply_idx].static_eval = raw_eval;
    td.stack[ply_idx].reduction = 0;
    if ply_idx + 2 < td.stack.len() {
        td.stack[ply_idx + 2].cutoff_count = 0;
    }

    // Parent-move quiet history update from static-eval surprise.
    if selective
        && !in_singular
        && ply > 0
        && !in_check
        && prior_move.is_quiet()
        && parent_eval != TtDepth::NONE_SCORE
        && (depth < 6 || entry.is_none())
        && let Some(h) = td.history.as_deref_mut()
    {
        let eval_delta = -(raw_eval + parent_eval);
        if eval_delta != 0 {
            let bonus = (eval_delta * 6).clamp(-150, 300);
            h.quiet_history.update(td.board.prior_threats(), false, !td.board.side_to_move(), prior_move, bonus);
        }
    }

    if let Some(h) = td.history.as_deref_mut() {
        h.clear_child_killers(ply_idx);
    }

    let static_eval = raw_eval;
    let improving = td.stack.is_improving(ply_idx, in_check, static_eval);
    let parent_stat_score = if ply_idx > 0 { td.stack[ply_idx - 1].stat_score } else { 0 };
    let parent_reduction = if ply_idx > 0 { td.stack[ply_idx - 1].reduction } else { 0 };
    let parent_laterality = if ply_idx > 0 { td.stack[ply_idx - 1].laterality } else { 0 };

    // Hindsight depth adjustment from parent LMR reduction
    if selective
        && use_hindsight
        && !in_singular
        && ply > 0
        && !in_check
        && parent_eval != TtDepth::NONE_SCORE
        && parent_reduction > 0
    {
        let eval_delta = static_eval + parent_eval;
        if parent_reduction >= HINDSIGHT_EXT_MIN_REDUCTION
            && prior_move.is_quiet()
            && eval_delta < 0
            && static_eval < alpha
            && (depth as usize) < MAX_PLY_USIZE - 1
        {
            depth += 1;
        } else if !tt_pv && depth >= 2 && eval_delta > HINDSIGHT_REDUCTION_MARGIN {
            depth -= 1;
        }
    }

    // Razoring
    if selective
        && use_razoring
        && !in_singular
        && !PV
        && !in_check
        && depth <= RAZOR_MAX_DEPTH
        && !is_win(alpha)
        && static_eval + RAZOR_BASE + (depth as i32) * RAZOR_MARGIN <= alpha
        && !entry.is_some_and(|e| e.bound == Bound::Lower && e.score > alpha)
    {
        let (q_score, q_move) = quiescence::<CFG, false>(td, alpha, beta, 0, ply, previous);
        if q_score <= alpha {
            return (q_score, None);
        }
        if tt_move.is_null() && q_score >= beta + RAZOR_MOVE_MARGIN {
            tt_move = q_move;
        }
    }

    // Reverse Futility Pruning (RFP)
    if selective
        && use_rfp
        && !in_singular
        && !tt_pv
        && !in_check
        && depth <= RFP_MAX_DEPTH
        && static_eval >= beta
        && !is_decisive(beta)
        && !is_decisive(static_eval)
    {
        let stm = td.board.side_to_move();
        let tt_below_beta = entry.is_some_and(|e| e.score != TtDepth::NONE_SCORE && e.score < beta);
        if !tt_below_beta && td.board.non_pawn_count(stm) > 0 {
            let no_threats_on_us = (td.board.all_threats() & td.board.colors(stm)).is_empty();
            let margin = (RFP_DEPTH_MARGIN * depth as i32
                - RFP_IMPROVING_BONUS * i32::from(improving)
                - RFP_NO_THREATS_BONUS * i32::from(no_threats_on_us)
                + parent_stat_score / RFP_STAT_DIVISOR)
                .max(RFP_MIN_MARGIN);
            if static_eval - margin >= beta {
                return ((static_eval + beta) / 2, None);
            }
        }
    }

    // Null Move Pruning (NMP)
    if selective
        && use_nmp
        && !in_singular
        && !PV
        && !in_check
        && prior_move.is_present()
        && depth >= NMP_MIN_DEPTH
        && ply >= td.stack.nmp_min_ply
        && !is_decisive(beta)
        && !is_decisive(raw_eval)
    {
        let stm = td.board.side_to_move();
        let non_pawn_count = td.board.non_pawn_count(stm);
        let tt_upper_fail_low =
            entry.is_some_and(|e| e.bound == Bound::Upper && e.score != TtDepth::NONE_SCORE && e.score < beta);
        let eval_margin_ok = if depth <= NMP_SHALLOW_DEPTH {
            raw_eval >= beta + NMP_SHALLOW_MARGIN
        } else {
            raw_eval >= beta + NMP_DEEP_FLOOR
        };
        let tt_tactical_win = tt_move.is_noisy() && td.board.see_ge(tt_move, NMP_TT_SEE_MARGIN);

        if !tt_upper_fail_low
            && eval_margin_ok
            && non_pawn_count > u32::from(depth > NMP_TWO_PIECE_DEPTH)
            && !tt_tactical_win
        {
            let diff = raw_eval - beta;
            let extra = if diff < NMP_DEEP_FLOOR { 0 } else { (diff.max(0) / NMP_EXTRA_DIV).min(NMP_EXTRA_CAP) };
            let reduction = NMP_BASE_REDUCTION + i32::from(improving) + depth as i32 / 4 + extra;
            let null_depth = (depth as i32 - reduction).max(0) as u8;

            if let Some(h) = td.history.as_deref_mut() {
                h.continuation_history.remember(ply_idx, false, Move::NULL, Piece::None);
            }
            td.stack[ply_idx].mv = Move::NULL;
            td.stack[ply_idx].move_count = 0;
            td.stack[ply_idx].stat_score = 0;
            td.stack[ply_idx].reduction = 0;
            td.stack[ply_idx].laterality = parent_laterality;

            td.make_null_move();
            let null_score =
                -search_node::<CFG, false>(td, -beta, -beta + 1, null_depth, !cut_node, ply + 1, ply + 1).0;
            td.undo_null_move();

            if null_score >= beta {
                let pruned_score = if null_score >= MATE_THRESHOLD { beta } else { null_score };
                if td.stack.nmp_min_ply > 0 || depth < NMP_VERIFY_MIN_DEPTH {
                    return (pruned_score, None);
                }

                td.stack.nmp_min_ply = ply + 3 * u16::from(null_depth) / 4;
                let verified_score =
                    search_node::<CFG, false>(td, beta - 1, beta, null_depth, false, ply, last_critical_ply).0;
                td.stack.nmp_min_ply = 0;

                if verified_score >= beta {
                    return (pruned_score, None);
                }
            }
        }
    }

    // ProbCut
    if selective
        && use_probcut
        && !in_singular
        && cut_node
        && !in_check
        && depth >= PROBCUT_MIN_DEPTH
        && !is_decisive(beta)
        && !is_decisive(static_eval)
    {
        let prob_beta = beta + PROBCUT_BETA_MARGIN - PROBCUT_IMPROVING_BONUS * i32::from(improving);
        let tt_allows_depth = entry
            .is_none_or(|e| e.depth < depth as i32 - 3 || (e.score != TtDepth::NONE_SCORE && e.score >= prob_beta));
        let gate_ok = match entry.filter(|e| e.score != TtDepth::NONE_SCORE) {
            Some(e) => e.score >= prob_beta && !is_decisive(e.score),
            None => static_eval >= beta,
        };

        if !tt_move.is_quiet() && tt_allows_depth && gate_ok {
            let see_threshold = prob_beta - static_eval;
            let prob_depth = depth - 4;
            let mut prob_picker = MainMovePicker::new_probcut(tt_move, see_threshold);

            while let Some(mv) = prob_picker.next(&td.board, td.history.as_deref(), true, ply_idx) {
                let moved_piece = td.board.moved_piece(mv);
                let hist = td.history.as_deref().map_or(0, |h| {
                    h.noisy_history.get(td.board.all_threats(), moved_piece, mv.to(), td.board.type_on(mv.capture_sq()))
                });
                if let Some(h) = td.history.as_deref_mut() {
                    h.continuation_history.remember(ply_idx, false, mv, moved_piece);
                }
                td.stack[ply_idx].mv = mv;
                td.stack[ply_idx].move_count = 1;
                td.stack[ply_idx].stat_score = hist;
                td.stack[ply_idx].reduction = 0;
                td.stack[ply_idx].laterality = parent_laterality;

                td.make_move(mv);

                let mut score = -quiescence::<CFG, false>(td, -prob_beta, -prob_beta + 1, 0, ply + 1, Some(mv.to())).0;
                if score >= prob_beta {
                    score = -search_node::<CFG, false>(
                        td,
                        -prob_beta,
                        -prob_beta + 1,
                        prob_depth,
                        !cut_node,
                        ply + 1,
                        last_critical_ply,
                    )
                    .0;
                }

                td.undo_move(mv);

                if score >= prob_beta && !is_decisive(score) {
                    if let Some(t) = td.tt {
                        t.write(hash, depth as i32 - 3, raw_eval, score, Bound::Lower, mv, ply as isize, tt_pv, false);
                    }
                    return (score, Some(mv));
                }
            }
        }
    }

    // Internal Iterative Reduction (IIR)
    if selective
        && !in_singular
        && ((PV && use_root_iir) || (!PV && cut_node && use_iir))
        && reserves_tt_verification_ply(depth, tt_move.is_present(), entry)
    {
        depth -= 1;
    }

    let potential_singularity = selective
        && use_se
        && !in_singular
        && ply > 0
        && ply_idx < MAX_PLY_USIZE / 2
        && depth >= SE_MIN_DEPTH
        && tt_move.is_present()
        && entry.is_some_and(|e| {
            e.depth >= depth as i32 - 3
                && matches!(e.bound, Bound::Lower | Bound::Exact)
                && e.score != TtDepth::NONE_SCORE
                && !is_decisive(e.score)
        });

    // Checked TT-capture cutoff
    if selective
        && use_checked_tt_cut
        && !in_singular
        && !PV
        && in_check
        && depth >= CHECKED_TT_CAPTURE_MIN_DEPTH
        && td.board.fiftymove_clock() < 90
        && !is_decisive(beta)
        && tt_move.is_capture()
        && entry.is_some_and(|e| {
            matches!(e.bound, Bound::Lower | Bound::Exact)
                && e.score != TtDepth::NONE_SCORE
                && !is_decisive(e.score)
                && e.score >= beta + CHECKED_TT_CAPTURE_MARGIN
                && e.depth >= depth as i32 - CHECKED_TT_CAPTURE_DEPTH_GAP
        })
    {
        return (beta + CHECKED_TT_CAPTURE_MARGIN, None);
    }

    let node_bias = compute_node_bias(
        PV,
        in_check,
        cut_node,
        tt_pv,
        improving,
        tt_move.is_capture(),
        parent_reduction,
        ply,
        last_critical_ply,
        parent_laterality,
    );

    let (killers, countermove) = if let Some(h) = td.history.as_deref() {
        (h.killers(ply_idx), h.countermove(prior_move))
    } else {
        ([Move::NULL; 2], Move::NULL)
    };

    let mut best_score = -INFINITY;
    let mut best_move = None;
    let mut bound = Bound::Upper;
    let mut move_count = 0usize;
    let mut quiet_moves = [Move::NULL; 32];
    let mut quiet_len = 0usize;
    let mut noisy_moves = [Move::NULL; 32];
    let mut noisy_len = 0usize;
    let mut skip_quiets = false;

    let mut picker = MainMovePicker::new(if selective { tt_move } else { Move::NULL }, false, killers, countermove);
    while let Some(mv) = picker.next(&td.board, td.history.as_deref(), skip_quiets, ply_idx) {
        if mv == excluded_move {
            continue;
        }
        let emitted_stage = picker.emitted_stage();
        move_count += 1;
        if ply == 0 && td.board.is_draw(0) {
            return (draw, Some(mv));
        }

        let moved_piece = td.board.moved_piece(mv);
        let is_quiet = mv.is_quiet();
        let creates_threat = selective && is_quiet && creates_threat_on_board(&td.board, mv);
        let hist = if let Some(h) = td.history.as_deref() {
            if is_quiet {
                h.quiet_history.get(td.board.all_threats(), creates_threat, td.board.side_to_move(), mv)
                    + h.conthist(ply_idx, 1, moved_piece, mv.to())
                    + h.conthist(ply_idx, 2, moved_piece, mv.to())
            } else {
                h.noisy_history.get(td.board.all_threats(), moved_piece, mv.to(), td.board.type_on(mv.capture_sq()))
            }
        } else {
            0
        };

        // Late Move Pruning (LMP)
        if selective
            && use_lmp
            && !PV
            && ply > 0
            && !in_check
            && is_quiet
            && best_score > -MATE_THRESHOLD
            && !is_decisive(beta)
            && move_count >= lmp_limit(depth, improving)
        {
            skip_quiets = true;
        }

        // Main-loop Futility Pruning (FP)
        if selective
            && use_fp
            && !PV
            && ply > 0
            && !in_check
            && is_quiet
            && best_score > -MATE_THRESHOLD
            && !is_decisive(alpha)
        {
            let lmr_plies = lmr_table_term(depth, move_count) / LMR_SCALE;
            let fp_depth = (depth as i32 - lmr_plies).max(0);
            if fp_depth <= FP_MAX_DEPTH && !td.board.is_direct_check(mv) {
                let hist_adj = (hist / FP_HIST_DIV).clamp(FP_HIST_MIN, FP_HIST_MAX);
                let margin = (FP_BASE_MARGIN + FP_DEPTH_MARGIN * fp_depth + hist_adj).max(FP_MIN_MARGIN);
                if static_eval + margin <= alpha {
                    continue;
                }
            }
        }

        // Quiet-History Pruning (HP)
        if selective
            && use_hp
            && !PV
            && ply > 0
            && !in_check
            && is_quiet
            && depth <= HP_MAX_DEPTH
            && best_score > -MATE_THRESHOLD
            && !td.board.is_direct_check(mv)
        {
            let threshold = HP_DEPTH_SCALE * depth as i32 + HP_IMPROVING_OFFSET * i32::from(improving);
            if hist < threshold {
                continue;
            }
        }

        // Main-loop SEE pruning
        if selective && use_see && ply > 0 && !in_check && best_score > -MATE_THRESHOLD {
            let cut_shift = usize::from(!PV && cut_node);
            let threshold = if is_quiet {
                let lmr_plies = lmr_table_term(depth, move_count) / LMR_SCALE;
                let rank = (depth as i32 + hist / SEE_QUIET_HIST_DIV - lmr_plies).max(1);
                (SEE_QUIET_QUAD_COEFF * rank * rank) >> cut_shift
            } else {
                ((SEE_NOISY_DEPTH_COEFF * depth as i32 - hist / SEE_NOISY_HIST_DIV) >> cut_shift).min(0)
            };
            if !td.board.see_ge(mv, threshold) {
                continue;
            }
        }

        // Singular Extensions (SE)
        let mut extension = 0i32;
        if potential_singularity
            && mv == tt_move
            && let Some(e) = &entry
        {
            let s_beta = (e.score.min(beta + SE_BETA_MARGIN) - depth as i32 * SE_DEPTH_MARGIN).max(-MATE_THRESHOLD + 1);
            let s_depth = ((9 * (depth as i32 - 1)) / 16).max(1) as u8;

            td.stack[ply_idx].excluded_move = mv;
            let verify = search_node::<CFG, false>(td, s_beta - 1, s_beta, s_depth, true, ply, last_critical_ply).0;
            td.stack[ply_idx].excluded_move = Move::NULL;
            if let Some(h) = td.history.as_deref_mut() {
                h.clear_child_killers(ply_idx);
            }

            if verify < s_beta {
                extension = if PV || tt_pv {
                    1
                } else {
                    1 + i32::from(verify < s_beta - SE_DOUBLE_GAP) + i32::from(verify < s_beta - SE_TRIPLE_GAP)
                };
            } else if verify >= beta && !is_decisive(verify) {
                return (verify, None);
            } else if PV {
                if e.score >= beta {
                    extension = -1;
                }
            } else {
                extension = if cut_node { -2 } else { -1 };
            }
        }

        if let Some(h) = td.history.as_deref_mut() {
            h.continuation_history.remember(ply_idx, in_check, mv, moved_piece);
        }
        td.stack[ply_idx].mv = mv;
        td.stack[ply_idx].move_count = move_count as u16;
        td.stack[ply_idx].stat_score = hist;
        td.stack[ply_idx].laterality =
            parent_laterality + if move_count > 1 { (move_count.ilog2() as i32 - 1).max(0) } else { 0 };

        td.make_move(mv);
        let gives_check = td.board.in_check();
        let base_child_depth = depth as i32 - 1;
        let max_child_depth = (MAX_PLY_USIZE as i32 - 1 - ply as i32).clamp(0, i32::from(u8::MAX));
        let child_depth = (base_child_depth + extension).clamp(0, max_child_depth) as u8;

        let mut gate_child_depth = child_depth;
        let mut lmr_quotient = 0i32;

        let score = if !selective || move_count == 1 {
            td.stack[ply_idx].reduction = (base_child_depth - child_depth as i32) * LMR_SCALE;
            let res = -search_node::<CFG, PV>(
                td,
                -beta,
                -alpha,
                child_depth,
                !PV && !cut_node,
                ply + 1,
                if move_count == 1 { last_critical_ply } else { ply + 1 },
            )
            .0;
            td.stack[ply_idx].reduction = 0;
            res
        } else {
            let lmr_eligible = use_lmr && depth >= 2 && (is_quiet || !PV);
            let mut reduced_depth = child_depth;
            let mut quotient = 0;

            if lmr_eligible && child_depth >= 1 {
                let base_term = if is_quiet {
                    lmr_table_term(depth, move_count)
                } else if emitted_stage == MainStage::BadNoisy {
                    LMR_BAD_NOISY_BASE
                } else {
                    LMR_GOOD_NOISY_BASE
                };

                let raw_reduction = base_term + node_bias
                    - hist / 10
                    - LMR_PRIORITY_QUIET_BONUS * (emitted_stage == MainStage::PriorityQuiet) as i32
                    - LMR_CHECK_BONUS * gives_check as i32
                    - LMR_THREAT_BONUS * creates_threat as i32
                    + LMR_EXACT_BOUND_PENALTY * (bound == Bound::Exact) as i32
                    + LMR_CHILD_CUTOFF_PENALTY
                        * (ply_idx + 1 < td.stack.len() && td.stack[ply_idx + 1].cutoff_count > 2) as i32;

                quotient = raw_reduction / LMR_SCALE;
                let reduction_plies = if quotient < 0 {
                    if depth >= 3 { (quotient / 2).max(-1) } else { 0 }
                } else {
                    quotient.min(child_depth as i32 - 1)
                };
                reduced_depth = (child_depth as i32 - reduction_plies).clamp(1, child_depth as i32 + 1) as u8;
            }

            td.stack[ply_idx].reduction = (child_depth as i32 - reduced_depth as i32) * LMR_SCALE;
            let mut scout = -search_node::<CFG, false>(td, -alpha - 1, -alpha, reduced_depth, true, ply + 1, ply + 1).0;
            td.stack[ply_idx].reduction = 0;

            if is_quiet
                && (quotient > 0 || !lmr_eligible)
                && let Some(h) = td.history.as_deref_mut()
            {
                let bonus = if scout > alpha { SCOUT_HISTORY_BONUS } else { -SCOUT_HISTORY_BONUS };
                h.update_continuation_histories(ply_idx, moved_piece, mv.to(), bonus);
            }

            let mut retry_depth = child_depth;
            if quotient > 0 && scout > alpha && reduced_depth < child_depth {
                retry_depth = plan_scout_retry(best_score, scout, child_depth, reduced_depth);
                if retry_depth > reduced_depth {
                    td.stack[ply_idx].reduction = (child_depth as i32 - retry_depth as i32) * LMR_SCALE;
                    scout = -search_node::<CFG, false>(
                        td,
                        -alpha - 1,
                        -alpha,
                        retry_depth,
                        PV || !cut_node,
                        ply + 1,
                        last_critical_ply,
                    )
                    .0;
                    td.stack[ply_idx].reduction = 0;
                }
            }

            gate_child_depth = retry_depth;
            lmr_quotient = quotient;

            if PV && scout > alpha && beta > alpha + 1 {
                td.stack[ply_idx].reduction = (child_depth as i32 - retry_depth as i32) * LMR_SCALE;
                let res =
                    -search_node::<CFG, true>(td, -beta, -alpha, retry_depth, false, ply + 1, last_critical_ply).0;
                td.stack[ply_idx].reduction = 0;
                res
            } else {
                scout
            }
        };
        td.undo_move(mv);

        // Post-move late-quiet closure: after searching a move from the Quiet
        // stage, close the remaining quiet moves if static evaluation is still
        // far below alpha at shallow LMR-adjusted depth.
        let margin_depth = gate_child_depth as i32 - lmr_quotient;
        if selective
            && use_lqc
            && ply > 0
            && !in_check
            && !gives_check
            && emitted_stage == MainStage::Quiet
            && margin_depth <= LATE_QUIET_MAX_DEPTH
            && score.max(best_score) > -MATE_THRESHOLD
            && !is_decisive(alpha)
            && static_eval < alpha - LATE_QUIET_BASE_MARGIN - LATE_QUIET_DEPTH_MARGIN * margin_depth.max(0)
        {
            skip_quiets = true;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
            if score > alpha {
                bound = Bound::Exact;
                if PV {
                    td.pv_table.update(ply_idx, mv);
                }
                if score >= beta {
                    bound = Bound::Lower;
                    td.stack[ply_idx].cutoff_count += 1;
                    break;
                }
                alpha = score;
                if selective
                    && PV
                    && mv != tt_move
                    && let Some(t) = td.tt
                {
                    t.write(hash, depth as i32, raw_eval, score, Bound::Lower, mv, ply as isize, true, false);
                }
            } else if move_count == 1 && PV {
                td.pv_table.update(ply_idx, mv);
            }
        }

        if mv.is_quiet() && quiet_len < 32 {
            quiet_moves[quiet_len] = mv;
            quiet_len += 1;
        } else if mv.is_noisy() && noisy_len < 32 {
            noisy_moves[noisy_len] = mv;
            noisy_len += 1;
        }
    }

    if move_count == 0 {
        if in_singular {
            return (alpha, None);
        }
        let score = if in_check { -MATE_SCORE + i32::from(ply) } else { draw_score(draw_jitter, td.nodes) };
        return (score, None);
    }

    if selective {
        if best_score >= beta
            && let Some(best_mv) = best_move
        {
            update_histories_on_best_move(
                td,
                ply_idx,
                depth as i32,
                cut_node,
                best_mv,
                best_score,
                beta,
                &quiet_moves[..quiet_len],
                &noisy_moves[..noisy_len],
            );
        }

        // Child fail-low rewards the parent move that refuted the current side.
        if ply > 0
            && bound == Bound::Upper
            && (cut_node || PV)
            && prior_move.is_present()
            && let Some(h) = td.history.as_deref_mut()
        {
            let d = depth as i32;
            let prior_threats = td.board.prior_threats();
            if prior_move.is_quiet() {
                let factor = 90
                    + (15 * parent_move_count as i32).min(225)
                    + 140 * (!in_check && best_score <= raw_eval - 50) as i32
                    + 300 * (parent_eval != TtDepth::NONE_SCORE && best_score <= -parent_eval - 70) as i32;
                let scaled_bonus = factor * (180 * d - 40).min(2_400) / 128;
                h.quiet_history.update(prior_threats, false, !td.board.side_to_move(), prior_move, scaled_bonus);
                if ply >= 2 {
                    let bonus = (150 * d - 45).min(1_350);
                    h.continuation_history.update(
                        ply_idx - 1,
                        1,
                        td.board.piece_on(prior_move.to()),
                        prior_move.to(),
                        bonus,
                    );
                }
            } else if prior_move.is_noisy() {
                let captured_type = td.board.captured_piece().piece_type();
                let bonus = (50 * d).min(650);
                h.noisy_history.update(
                    prior_threats,
                    td.board.piece_on(prior_move.to()),
                    prior_move.to(),
                    captured_type,
                    bonus,
                );
            }
        }
    }

    tt_pv |= ply > 0 && bound == Bound::Upper && move_count > 2;

    if selective
        && !in_singular
        && let Some(t) = td.tt
    {
        let store_mv = if bound == Bound::Upper { Move::NULL } else { best_move.unwrap_or(Move::NULL) };
        let store_depth = (depth as i32 - i32::from(bound == Bound::Lower)).max(1);
        t.write(hash, store_depth, raw_eval, best_score, bound, store_mv, ply as isize, tt_pv, PV);
    }

    (best_score, best_move)
}

#[allow(clippy::too_many_arguments)]
fn update_histories_on_best_move(
    td: &mut ThreadData<'_>, ply: usize, depth: i32, cut_node: bool, best_move: Move, best_score: i32, beta: i32,
    quiet_moves: &[Move], noisy_moves: &[Move],
) {
    let Some(history) = td.history.as_deref_mut() else { return };
    let board = &td.board;
    let prior_move = if ply > 0 { td.stack[ply - 1].mv } else { Move::NULL };
    let parent_move_count = if ply > 0 { td.stack[ply - 1].move_count as usize } else { 0 };
    let stm = board.side_to_move();
    let threats = board.all_threats();
    let pawn_key = board.pawn_key();

    let noisy_bonus = (100 * depth).min(900) - 40 - 80 * cut_node as i32;
    let noisy_malus = (170 * depth).min(1_200) - 50 - 15 * noisy_moves.len() as i32;

    let quiet_bonus = (180 * depth).min(1_700) - 70 - 40 * cut_node as i32;
    let quiet_malus = (170 * depth).min(1_100) - 45 - 30 * quiet_moves.len() as i32;

    let cont_bonus = (100 * depth).min(1_100) - 70 - 45 * cut_node as i32;
    let cont_malus = (400 * depth).min(950) - 50 - 15 * quiet_moves.len() as i32;

    if best_move.is_noisy() {
        history.noisy_history.update(
            threats,
            board.moved_piece(best_move),
            best_move.to(),
            board.type_on(best_move.capture_sq()),
            noisy_bonus,
        );
    } else {
        if best_score >= beta {
            history.update_killer_and_countermove(ply, prior_move, best_move);
        }
        let best_piece = board.moved_piece(best_move);
        let best_creates_threat = creates_threat_on_board(board, best_move);
        history.quiet_history.update(threats, best_creates_threat, stm, best_move, quiet_bonus);
        history.pawn_history.update(pawn_key, best_piece, best_move.to(), quiet_bonus);
        history.update_continuation_histories(ply, best_piece, best_move.to(), cont_bonus);

        for (i, &mv) in quiet_moves.iter().enumerate() {
            let denom = 1_000 + 45 * i as i32;
            let scale = 1_000_i32 * 1_000 / (denom * denom / 1_000);
            let piece = board.moved_piece(mv);
            let creates_threat = creates_threat_on_board(board, mv);
            history.quiet_history.update(threats, creates_threat, stm, mv, -quiet_malus * scale / 1_000);
            history.pawn_history.update(pawn_key, piece, mv.to(), -quiet_malus * scale / 1_000);
            history.update_continuation_histories(ply, piece, mv.to(), -cont_malus * scale / 1_000);
        }
    }

    for &mv in noisy_moves {
        history.noisy_history.update(
            threats,
            board.moved_piece(mv),
            mv.to(),
            board.type_on(mv.capture_sq()),
            -noisy_malus,
        );
    }

    if ply > 0 && prior_move.is_quiet() && parent_move_count < 2 {
        let malus = (90 * depth - 50).min(900);
        let prior_piece = board.piece_on(prior_move.to());
        if prior_piece != Piece::None {
            history.update_continuation_histories(ply - 1, prior_piece, prior_move.to(), -malus);
        }
    }
}

const fn qs_futility_victim(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => MATE_SCORE,
        PieceType::None => 0,
    }
}

fn quiescence<const CFG: u32, const PV: bool>(
    td: &mut ThreadData<'_>, mut alpha: i32, beta: i32, q_depth: i16, ply: u16, previous: Option<Square>,
) -> (i32, Move) {
    let draw_jitter = (CFG & CFG_DRAW_JITTER) != 0;
    let selective = (CFG & CFG_SELECTIVE) != 0;
    let use_qs_hp = (CFG & CFG_QS_HP) != 0;
    let use_qs_fp = (CFG & CFG_QS_FP) != 0;
    let use_qs_lmp = (CFG & CFG_QS_LMP) != 0;
    let use_qs_early_cut = (CFG & CFG_QS_EARLY_CUT) != 0;
    let use_qs_see = (CFG & CFG_QS_SEE) != 0;
    let use_qs_standpat = (CFG & CFG_QS_STANDPAT) != 0;
    let use_qs_bound_avg = (CFG & CFG_QS_BOUND_AVG) != 0;
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    debug_assert!(ply <= MAX_PLY && q_depth <= 0);
    td.sel_depth = td.sel_depth.max(ply);
    let draw = draw_score(draw_jitter, td.nodes);
    if ply > 0 && alpha < draw && td.board.upcoming_repetition(ply as usize) {
        alpha = draw;
        if alpha >= beta {
            return (alpha, Move::NULL);
        }
    }

    let in_check = td.board.in_check();
    let is_draw = td.board.is_draw(ply as isize);
    if is_draw || ply >= MAX_PLY {
        if in_check && !td.board.has_legal_moves() {
            return (-MATE_SCORE + i32::from(ply), Move::NULL);
        }
        return (if is_draw || in_check { draw } else { evaluate(&td.board) }, Move::NULL);
    }

    let hash = td.board.hash();
    let tt_depth = i32::from(q_depth.max(RECAPTURE_ONLY_DEPTH));
    let entry =
        if selective { td.tt.and_then(|t| t.read(hash, td.board.fiftymove_clock(), ply as isize)) } else { None };
    let mut tt_move = Move::NULL;
    let mut tt_pv = PV;

    if let Some(entry) = &entry {
        tt_move = entry.mv;
        tt_pv |= entry.tt_pv;

        if entry.depth >= tt_depth
            && entry.score != TtDepth::NONE_SCORE
            && (!PV || !is_decisive(entry.score))
            && match entry.bound {
                Bound::Exact => true,
                Bound::Lower => entry.score >= beta,
                Bound::Upper => entry.score <= alpha,
                Bound::None => false,
            }
        {
            return (entry.score, Move::NULL);
        }
    }

    let raw_eval;
    let mut best_score;
    if in_check {
        raw_eval = TtDepth::NONE_SCORE;
        best_score = -INFINITY;
    } else {
        raw_eval = match &entry {
            Some(entry) if entry.raw_eval != TtDepth::NONE_SCORE => entry.raw_eval,
            _ => {
                let ev = evaluate(&td.board);
                if selective
                    && entry.is_none()
                    && let Some(t) = td.tt
                {
                    t.write(
                        hash,
                        TtDepth::EVAL_ONLY,
                        ev,
                        TtDepth::NONE_SCORE,
                        Bound::None,
                        Move::NULL,
                        ply as isize,
                        tt_pv,
                        false,
                    );
                }
                ev
            }
        };
        best_score = raw_eval;

        if let Some(entry) = &entry
            && entry.depth >= tt_depth
            && entry.score != TtDepth::NONE_SCORE
            && (!PV || !is_decisive(entry.score))
            && match entry.bound {
                Bound::Exact => true,
                Bound::Lower => entry.score > best_score,
                Bound::Upper => entry.score < best_score,
                Bound::None => false,
            }
        {
            best_score = entry.score;
        }
    }

    if best_score >= beta {
        if selective && use_qs_standpat {
            let ret = if q_depth == 0 && !is_decisive(best_score) && !is_decisive(beta) {
                (beta + best_score) / 2
            } else {
                best_score
            };
            return (ret, Move::NULL);
        }
        if selective
            && entry.is_none()
            && let Some(t) = td.tt
        {
            t.write(hash, tt_depth, raw_eval, best_score, Bound::Lower, Move::NULL, ply as isize, tt_pv, false);
        }
        return (best_score, Move::NULL);
    }
    let stand_pat = best_score;
    alpha = alpha.max(best_score);

    let mut best_move = Move::NULL;
    let mut move_count = 0usize;
    let mut picker = QuiescenceMovePicker::new(if selective { tt_move } else { Move::NULL }, in_check, q_depth >= -1);
    while let Some(mv) = picker.next(
        &td.board,
        td.history.as_deref(),
        selective && in_check && best_score > -MATE_THRESHOLD,
        ply as usize,
    ) {
        move_count += 1;
        let moved_piece = td.board.moved_piece(mv);
        let recapture = mv.is_capture() && previous == Some(mv.capture_sq());
        if selective && !in_check && best_score > -MATE_THRESHOLD {
            if !recapture && !mv.is_promotion() && !td.board.gives_check(mv) {
                if q_depth <= RECAPTURE_ONLY_DEPTH {
                    continue;
                }
                if use_qs_lmp && mv.is_capture() && move_count > QS_LATE_CAPTURE_LIMIT && !td.board.see_ge(mv, 1) {
                    continue;
                }
                if mv.is_capture() {
                    let captured_type = td.board.type_on(mv.capture_sq());
                    let victim_val = qs_futility_victim(captured_type);
                    let attacker_val = qs_futility_victim(moved_piece.piece_type());
                    if use_qs_early_cut
                        && move_count == 1
                        && !is_decisive(beta)
                        && raw_eval + victim_val - attacker_val - QS_EARLY_CUTOFF_MARGIN > beta
                        && td.history.as_deref().is_some_and(|h| {
                            h.noisy_history.get(td.board.all_threats(), moved_piece, mv.to(), captured_type) > 0
                        })
                    {
                        if let Some(t) = td.tt {
                            t.write(hash, tt_depth, raw_eval, beta, Bound::Lower, mv, ply as isize, tt_pv, false);
                        }
                        return (beta, Move::NULL);
                    }
                    if use_qs_fp {
                        let margin = stand_pat + QS_FUTILITY_MARGIN;
                        let capture_margin = margin + victim_val;
                        if alpha >= capture_margin || (alpha >= margin && !td.board.see_ge(mv, 1)) {
                            continue;
                        }
                    }
                }
            }
            if use_qs_hp
                && mv.is_quiet()
                && td
                    .history
                    .as_deref()
                    .is_some_and(|h| h.conthist(ply as usize, 1, moved_piece, mv.to()) < QS_QUIET_HIST_THRESHOLD)
            {
                continue;
            }
            if use_qs_see {
                let threshold = if mv.is_noisy() {
                    let hist = td.history.as_deref().map_or(0, |h| {
                        h.noisy_history.get(
                            td.board.all_threats(),
                            moved_piece,
                            mv.to(),
                            td.board.type_on(mv.capture_sq()),
                        )
                    });
                    ((alpha - stand_pat) / QS_SEE_EVAL_DIV - QS_SEE_BASE - hist / QS_SEE_HIST_DIV)
                        .max(QS_SEE_NOISY_FLOOR)
                } else {
                    QS_SEE_QUIET_THRESHOLD
                };
                if !td.board.see_ge(mv, threshold) {
                    continue;
                }
            }
        }
        if let Some(h) = td.history.as_deref_mut() {
            h.continuation_history.remember(ply as usize, in_check, mv, moved_piece);
        }
        td.make_move(mv);
        let score = -quiescence::<CFG, PV>(td, -beta, -alpha, q_depth - 1, ply + 1, Some(mv.to())).0;
        td.undo_move(mv);
        if score > best_score {
            best_score = score;
            if score > alpha {
                best_move = mv;
                if score >= beta {
                    break;
                }
                alpha = score;
            }
        }
    }

    if in_check && best_score == -INFINITY {
        return (-MATE_SCORE + i32::from(ply), Move::NULL);
    }

    if selective
        && best_score >= beta
        && best_move.is_noisy()
        && let Some(h) = td.history.as_deref_mut()
    {
        h.noisy_history.update(
            td.board.all_threats(),
            td.board.moved_piece(best_move),
            best_move.to(),
            td.board.type_on(best_move.capture_sq()),
            100,
        );
    }

    let bound = if best_score >= beta { Bound::Lower } else { Bound::Upper };
    if selective && use_qs_bound_avg {
        if best_score > beta && !is_decisive(best_score) && !is_decisive(beta) {
            let cutoff = (beta + best_score) / 2;
            best_score = (best_score + cutoff) / 2;
        } else if best_score < alpha && !is_decisive(best_score) && !is_decisive(alpha) {
            best_score = (alpha + best_score) / 2;
        }
    }

    if selective && let Some(t) = td.tt {
        t.write(hash, tt_depth, raw_eval, best_score, bound, best_move, ply as isize, tt_pv, false);
    }

    (best_score, best_move)
}

const fn draw_score(draw_jitter: bool, nodes: u64) -> i32 {
    if draw_jitter { (nodes % 5) as i32 - 2 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MoveKind;

    #[test]
    fn root_iir_reduces_effective_depth_when_tt_is_shallow_or_work_item_missing() {
        let pos = Board::starting_position();
        let legal_work_item = pos.generate_all_moves()[0];
        let illegal_work_item = Move::new(Square::A1, Square::H8, MoveKind::Normal);

        // 1. Cold/missing TT entry at depth 4: Root IIR reduces effective depth from 4 to 3.
        let tt_on = TranspositionTable::new(1);
        let mut hist_on = HistoryTables::default();
        let mut td_on = ThreadData::new(&pos, Some(&tt_on), Some(&mut hist_on));
        td_on.root_moves = RootMoves::new(&pos);
        td_on.root_moves.begin_depth_iteration(4);
        let _ = search_root::<CFG_ALL_SELECTIVE>(&mut td_on, -INFINITY, INFINITY, 4, 0, legal_work_item);
        assert_eq!(tt_on.read(pos.hash(), 0, 0).unwrap().depth, 3);

        let tt_off = TranspositionTable::new(1);
        let mut hist_off = HistoryTables::default();
        let mut td_off = ThreadData::new(&pos, Some(&tt_off), Some(&mut hist_off));
        td_off.root_moves = RootMoves::new(&pos);
        td_off.root_moves.begin_depth_iteration(4);
        let _ = search_root::<{ CFG_ALL_SELECTIVE & !CFG_ROOT_IIR }>(
            &mut td_off,
            -INFINITY,
            INFINITY,
            4,
            0,
            legal_work_item,
        );
        assert_eq!(tt_off.read(pos.hash(), 0, 0).unwrap().depth, 4);
        assert!(td_on.nodes < td_off.nodes);

        // 2. Sufficiently deep TT entry (depth = 1, 1 + 4 >= 5) with legal work_item: Root IIR does not reduce.
        let tt_deep = TranspositionTable::new(1);
        tt_deep.write(pos.hash(), 1, 0, 0, Bound::Exact, legal_work_item, 0, true, true);
        let mut hist_deep = HistoryTables::default();
        let mut td_deep = ThreadData::new(&pos, Some(&tt_deep), Some(&mut hist_deep));
        td_deep.root_moves = RootMoves::new(&pos);
        td_deep.root_moves.begin_depth_iteration(5);
        let _ = search_root::<CFG_ALL_SELECTIVE>(&mut td_deep, -INFINITY, INFINITY, 5, 0, legal_work_item);
        assert_eq!(tt_deep.read(pos.hash(), 0, 0).unwrap().depth, 5);

        // 3. Even with a deep TT entry (depth = 4), an absent or illegal work_item triggers Root IIR (5 -> 4).
        let tt_no_work = TranspositionTable::new(1);
        tt_no_work.write(pos.hash(), 4, 0, 0, Bound::Exact, legal_work_item, 0, true, true);
        let mut hist_no_work = HistoryTables::default();
        let mut td_no_work = ThreadData::new(&pos, Some(&tt_no_work), Some(&mut hist_no_work));
        td_no_work.root_moves = RootMoves::new(&pos);
        td_no_work.root_moves.begin_depth_iteration(5);
        let _ = search_root::<CFG_ALL_SELECTIVE>(&mut td_no_work, -INFINITY, INFINITY, 5, 0, illegal_work_item);
        assert_eq!(tt_no_work.read(pos.hash(), 0, 0).unwrap().depth, 4);
    }

    #[test]
    fn pv_node_iir_reduces_depth_independently_of_search_root() {
        let pos = Board::starting_position();
        let legal_mv = pos.generate_all_moves()[0];

        // 1. Interior PV node (ply = 1) with missing TT entry at depth 4: PV IIR reduces depth from 4 to 3.
        let tt_on = TranspositionTable::new(1);
        let mut hist_on = HistoryTables::default();
        let mut td_on = ThreadData::new(&pos, Some(&tt_on), Some(&mut hist_on));
        let _ = search_node::<CFG_ALL_SELECTIVE, true>(&mut td_on, -INFINITY, INFINITY, 4, false, 1, 0);
        assert_eq!(tt_on.read(pos.hash(), 0, 1).unwrap().depth, 3);

        let tt_off = TranspositionTable::new(1);
        let mut hist_off = HistoryTables::default();
        let mut td_off = ThreadData::new(&pos, Some(&tt_off), Some(&mut hist_off));
        let _ = search_node::<{ CFG_ALL_SELECTIVE & !CFG_ROOT_IIR }, true>(
            &mut td_off,
            -INFINITY,
            INFINITY,
            4,
            false,
            1,
            0,
        );
        assert_eq!(tt_off.read(pos.hash(), 0, 1).unwrap().depth, 4);
        assert!(td_on.nodes < td_off.nodes);

        // 2. Interior PV node with deep TT entry (depth = 1, 1 + 4 >= 5) and legal tt_move: PV IIR does not reduce.
        let tt_deep = TranspositionTable::new(1);
        tt_deep.write(pos.hash(), 1, 0, 0, Bound::Exact, legal_mv, 1, true, true);
        let mut hist_deep = HistoryTables::default();
        let mut td_deep = ThreadData::new(&pos, Some(&tt_deep), Some(&mut hist_deep));
        let _ = search_node::<CFG_ALL_SELECTIVE, true>(&mut td_deep, -INFINITY, INFINITY, 5, false, 1, 0);
        assert_eq!(tt_deep.read(pos.hash(), 0, 1).unwrap().depth, 5);
    }

    #[test]
    fn qs_standpat_fail_high_damps_at_qdepth_zero_and_preserves_eval_only_tt_entry() {
        // White has a Queen (+900 cp).
        let pos = Board::from_fen("7k/7p/8/8/2Q5/8/7P/7K w - - 0 1").unwrap();

        // 1. At q_depth == 0 with beta = 300 and raw_eval = 900:
        //    Returns damped (300 + 900) / 2 = 600 and leaves the TT entry as EVAL_ONLY (Bound::None).
        let tt_q0 = TranspositionTable::new(1);
        let mut hist_q0 = HistoryTables::default();
        let mut td_q0 = ThreadData::new(&pos, Some(&tt_q0), Some(&mut hist_q0));
        let (score_q0, mv_q0) = quiescence::<CFG_ALL_SELECTIVE, false>(&mut td_q0, 299, 300, 0, 1, None);
        assert_eq!(score_q0, 600);
        assert_eq!(mv_q0, Move::NULL);
        let entry_q0 = tt_q0.read(pos.hash(), 0, 1).unwrap();
        assert_eq!(entry_q0.depth, TtDepth::EVAL_ONLY);
        assert_eq!(entry_q0.bound, Bound::None);
        assert_eq!(entry_q0.raw_eval, 900);

        // 2. At q_depth == -1 (deeper QS ply) with beta = 300 and raw_eval = 900:
        //    Returns undamped 900.
        let tt_q1 = TranspositionTable::new(1);
        let mut hist_q1 = HistoryTables::default();
        let mut td_q1 = ThreadData::new(&pos, Some(&tt_q1), Some(&mut hist_q1));
        let (score_q1, _) = quiescence::<CFG_ALL_SELECTIVE, false>(&mut td_q1, 299, 300, -1, 2, None);
        assert_eq!(score_q1, 900);

        // 3. When CFG_QS_STANDPAT is disabled, q_depth == 0 returns undamped 900 and writes Bound::Lower.
        let tt_off = TranspositionTable::new(1);
        let mut hist_off = HistoryTables::default();
        let mut td_off = ThreadData::new(&pos, Some(&tt_off), Some(&mut hist_off));
        let (score_off, _) =
            quiescence::<{ CFG_ALL_SELECTIVE & !CFG_QS_STANDPAT }, false>(&mut td_off, 299, 300, 0, 1, None);
        assert_eq!(score_off, 900);
        assert_eq!(tt_off.read(pos.hash(), 0, 1).unwrap().bound, Bound::Lower);
    }

    #[test]
    fn qs_post_loop_bound_averaging_damps_fail_high_and_fail_low_scores() {
        // 1. Post-loop fail-high (best_score > beta):
        //    White starts at raw_eval = 0 (Rook vs Rook) and captures 1. Ra1xa8.
        //    With alpha = 99, beta = 100: the child at ply 2 (alpha = -100, raw_eval = -500)
        //    damps its fail-low to (-100 + -500) / 2 = -300, so 1. Ra1xa8 returns +300 > beta.
        //    Then ply 1 damps its fail-high: cutoff = (100 + 300) / 2 = 200; (300 + 200) / 2 = 250.
        let fh_pos = Board::from_fen("r6k/8/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let tt_fh = TranspositionTable::new(1);
        let mut hist_fh = HistoryTables::default();
        let mut td_fh = ThreadData::new(&fh_pos, Some(&tt_fh), Some(&mut hist_fh));
        let (score_fh, mv_fh) = quiescence::<CFG_ALL_SELECTIVE, false>(&mut td_fh, 99, 100, 0, 1, None);
        assert_eq!(score_fh, 250);
        assert_ne!(mv_fh, Move::NULL);
        let entry_fh = tt_fh.read(fh_pos.hash(), 0, 1).unwrap();
        assert_eq!(entry_fh.bound, Bound::Lower);
        assert_eq!(entry_fh.score, 250);

        // When CFG_QS_BOUND_AVG is disabled, returns undamped 500.
        let tt_fh_off = TranspositionTable::new(1);
        let mut hist_fh_off = HistoryTables::default();
        let mut td_fh_off = ThreadData::new(&fh_pos, Some(&tt_fh_off), Some(&mut hist_fh_off));
        let (score_fh_off, _) =
            quiescence::<{ CFG_ALL_SELECTIVE & !CFG_QS_BOUND_AVG }, false>(&mut td_fh_off, 99, 100, 0, 1, None);
        assert_eq!(score_fh_off, 500);

        // 2. Post-loop fail-low (best_score < alpha):
        //    White starts at raw_eval = 0 with no captures and alpha = 200, beta = 201.
        //    stand_pat = 0 < alpha; damped = (200 + 0) / 2 = 100 (Bound::Upper).
        let fl_pos = Board::from_fen("7k/7p/8/8/8/8/7P/7K w - - 0 1").unwrap();
        let tt_fl = TranspositionTable::new(1);
        let mut hist_fl = HistoryTables::default();
        let mut td_fl = ThreadData::new(&fl_pos, Some(&tt_fl), Some(&mut hist_fl));
        let (score_fl, mv_fl) = quiescence::<CFG_ALL_SELECTIVE, false>(&mut td_fl, 200, 201, 0, 1, None);
        assert_eq!(score_fl, 100);
        assert_eq!(mv_fl, Move::NULL);
        let entry_fl = tt_fl.read(fl_pos.hash(), 0, 1).unwrap();
        assert_eq!(entry_fl.bound, Bound::Upper);
        assert_eq!(entry_fl.score, 100);
    }

    #[test]
    fn hindsight_depth_adjustment_extends_and_reduces_based_on_parent_eval_delta() {
        let pos = Board::starting_position();
        let quiet_mv = pos.generate_all_moves()[0];

        // 1. Hindsight extension (depth += 1):
        //    parent_reduction = 2_000 (>= HINDSIGHT_EXT_MIN_REDUCTION), prior_move is quiet,
        //    static_eval (0) + parent_eval (-100) = -100 < 0, and static_eval (0) < alpha (500).
        //    depth = 3 is extended to 4 (> RAZOR_MAX_DEPTH), storing depth 4 on fail-low.
        let tt_ext = TranspositionTable::new(1);
        let mut hist_ext = HistoryTables::default();
        let mut td_ext = ThreadData::new(&pos, Some(&tt_ext), Some(&mut hist_ext));
        td_ext.stack[0].mv = quiet_mv;
        td_ext.stack[0].static_eval = -100;
        td_ext.stack[0].reduction = 2_000;
        td_ext.make_move(quiet_mv);
        let child_hash = td_ext.board.hash();
        let _ = search_node::<CFG_ALL_SELECTIVE, false>(&mut td_ext, 500, 501, 3, false, 1, 0);
        assert_eq!(tt_ext.read(child_hash, 0, 1).unwrap().depth, 4);

        // 2. Hindsight reduction (depth -= 1):
        //    parent_reduction = 1_000 (> 0), !tt_pv, depth = 5 >= 2, and
        //    static_eval (0) + parent_eval (100) = 100 > HINDSIGHT_REDUCTION_MARGIN (60).
        //    depth = 5 is reduced to 4 (> RAZOR_MAX_DEPTH), storing depth 4 on fail-low.
        let tt_red = TranspositionTable::new(1);
        let mut hist_red = HistoryTables::default();
        let mut td_red = ThreadData::new(&pos, Some(&tt_red), Some(&mut hist_red));
        td_red.stack[0].mv = quiet_mv;
        td_red.stack[0].static_eval = 100;
        td_red.stack[0].reduction = 1_000;
        td_red.make_move(quiet_mv);
        let _ = search_node::<CFG_ALL_SELECTIVE, false>(&mut td_red, 500, 501, 5, false, 1, 0);
        assert_eq!(tt_red.read(child_hash, 0, 1).unwrap().depth, 4);
    }
}
