use std::cell::RefCell;

use crate::{
    board::Board,
    evaluation::evaluate,
    history::HistoryTables,
    movepick::{MainMovePicker, QuiescenceMovePicker, creates_threat_on_board},
    transposition::{Bound, TranspositionTable, TtDepth},
    types::{Move, Piece, Square},
};

pub const MATE_SCORE: i32 = 32_000;
const MAX_PLY: u16 = 256;
const MAX_PLY_USIZE: usize = MAX_PLY as usize;
const RECAPTURE_ONLY_DEPTH: i16 = -4;
pub const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;
const INFINITY: i32 = MATE_SCORE + 1;

// Centipawn-scale (100 = 1 pawn) aspiration constants using power-of-10 ratios.
const ASPIRATION_MIN_DEPTH: u8 = 2;
const ASPIRATION_BASE_DELTA: i32 = 10;
const ASPIRATION_STABILITY_MAX: i32 = 5;
const ASPIRATION_VOLATILITY_NUM: i32 = 25;
const ASPIRATION_QUADRATIC_DIV: i32 = 10_000;
const ASPIRATION_FAIL_LOW_WIDEN_NUM: i32 = 25;
const ASPIRATION_FAIL_HIGH_WIDEN_NUM: i32 = 50;
const ASPIRATION_RATIO_DEN: i32 = 100;
const ASPIRATION_INFINITE_BOUND: i32 = 1_000;
const EVAL_STABILITY_THRESHOLD: i32 = 10;
const SCOUT_HISTORY_BONUS: i32 = 50;

thread_local! {
    static LOCAL_TT: RefCell<TranspositionTable> = RefCell::new(TranspositionTable::new(4));
    static LOCAL_HISTORY: RefCell<HistoryTables> = RefCell::new(HistoryTables::default());
}

/// Fixed-capacity principal variation buffer stored on the stack without heap allocation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PvLine {
    moves: [Move; MAX_PLY_USIZE + 1],
    len: usize,
}

impl PvLine {
    pub const fn new() -> Self {
        Self { moves: [Move::NULL; MAX_PLY_USIZE + 1], len: 0 }
    }

    pub const fn clear(&mut self) {
        self.len = 0;
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    fn update(&mut self, mv: Move, child: &Self) {
        let child_len = child.len.min(MAX_PLY_USIZE);
        self.moves[0] = mv;
        self.moves[1..=child_len].copy_from_slice(&child.moves[..child_len]);
        self.len = child_len + 1;
    }
}

impl Default for PvLine {
    fn default() -> Self {
        Self::new()
    }
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct RootScore {
    last: i32,
    average: i32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum AspirationKind {
    FailHigh,
    FailLow,
    Exact,
}

struct AspirationWindow {
    alpha: i32,
    beta: i32,
    delta: i32,
}

impl AspirationWindow {
    fn new(depth: u8, previous: RootScore, eval_stability: i32, pv_stability: i32) -> Self {
        if depth < ASPIRATION_MIN_DEPTH || previous.average <= -MATE_SCORE {
            return Self { alpha: -INFINITY, beta: INFINITY, delta: INFINITY };
        }
        let stability_shrink = eval_stability.min(pv_stability).min(ASPIRATION_STABILITY_MAX);
        let volatility = (previous.last - previous.average).abs() * ASPIRATION_VOLATILITY_NUM / ASPIRATION_RATIO_DEN;
        let magnitude = previous.average * previous.average / ASPIRATION_QUADRATIC_DIV;
        let delta = (ASPIRATION_BASE_DELTA - stability_shrink) + volatility + magnitude;

        let mut alpha = (previous.average - delta).max(-INFINITY);
        let mut beta = (previous.average + delta).min(INFINITY);
        if alpha < -ASPIRATION_INFINITE_BOUND {
            alpha = -INFINITY;
        }
        if beta > ASPIRATION_INFINITE_BOUND {
            beta = INFINITY;
        }
        Self { alpha, beta, delta }
    }

    fn on_result(&mut self, score: i32) -> AspirationKind {
        if score >= self.beta {
            self.beta = (score + self.delta).min(INFINITY);
            self.delta += self.delta * ASPIRATION_FAIL_HIGH_WIDEN_NUM / ASPIRATION_RATIO_DEN;
            if self.beta > ASPIRATION_INFINITE_BOUND {
                self.beta = INFINITY;
            }
            return AspirationKind::FailHigh;
        }
        if score <= self.alpha {
            let failed_alpha = self.alpha;
            self.alpha = (score - self.delta).max(-INFINITY);
            self.beta = (self.beta + failed_alpha) / 2;
            self.delta += self.delta * ASPIRATION_FAIL_LOW_WIDEN_NUM / ASPIRATION_RATIO_DEN;
            if self.alpha < -ASPIRATION_INFINITE_BOUND {
                self.alpha = -INFINITY;
            }
            if self.beta > ASPIRATION_INFINITE_BOUND {
                self.beta = INFINITY;
            }
            return AspirationKind::FailLow;
        }
        AspirationKind::Exact
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct IterationResult {
    score: i32,
    alpha: i32,
    beta: i32,
    depth: u8,
    upperbound: bool,
    lowerbound: bool,
    pv: PvLine,
}

impl Default for IterationResult {
    fn default() -> Self {
        Self {
            score: -INFINITY,
            alpha: -INFINITY,
            beta: INFINITY,
            depth: 0,
            upperbound: false,
            lowerbound: false,
            pv: PvLine::new(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct RootMove {
    mv: Move,
    nodes: u64,
    sel_depth: u16,
    searched_this_iteration: bool,
    current: IterationResult,
    previous: IterationResult,
    average: i32,
}

impl RootMove {
    fn new(mv: Move) -> Self {
        Self {
            mv,
            nodes: 0,
            sel_depth: 0,
            searched_this_iteration: false,
            current: IterationResult::default(),
            previous: IterationResult::default(),
            average: -INFINITY,
        }
    }

    const fn active_result(&self) -> &IterationResult {
        if self.searched_this_iteration { &self.current } else { &self.previous }
    }
}

struct RootMoves {
    records: Vec<RootMove>,
    best_move_changes: usize,
}

impl RootMoves {
    fn new(board: &Board) -> Self {
        let moves = board.generate_all_moves();
        let records = moves.iter().copied().map(RootMove::new).collect();
        Self { records, best_move_changes: 0 }
    }

    fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn at(&self, index: usize) -> &RootMove {
        &self.records[index]
    }

    fn begin_depth_iteration(&mut self, depth: u8) {
        self.best_move_changes = 0;
        for rec in &mut self.records {
            rec.previous = rec.current;
            rec.current.depth = depth;
            rec.searched_this_iteration = false;
        }
    }

    fn aspiration_previous_score(&self, pv_idx: usize) -> RootScore {
        let rec = &self.records[pv_idx];
        RootScore { last: rec.active_result().score, average: rec.average }
    }

    fn is_eligible_at_root(&self, pv_idx: usize, mv: Move) -> bool {
        self.records[pv_idx..].iter().any(|rec| rec.mv == mv)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_searched_move(
        &mut self, pv_idx: usize, mv: Move, score: i32, alpha_before: i32, beta: i32, first_candidate: bool,
        move_nodes: u64, sel_depth: u16, child_pv: &PvLine,
    ) {
        let Some(rec) = self.records.iter_mut().find(|rec| rec.mv == mv) else { return };
        if pv_idx == 0 {
            rec.nodes += move_nodes;
            rec.average = if rec.average == -INFINITY { score } else { (rec.average + score) / 2 };
        }
        if !first_candidate && score <= alpha_before {
            rec.current.score = -INFINITY;
            return;
        }
        rec.searched_this_iteration = true;
        rec.sel_depth = sel_depth;
        rec.current.alpha = alpha_before;
        rec.current.beta = beta;
        rec.current.score = score;
        rec.current.upperbound = score <= alpha_before;
        rec.current.lowerbound = score >= beta;
        rec.current.pv.update(mv, child_pv);
        if !first_candidate && pv_idx == 0 {
            self.best_move_changes += 1;
        }
    }

    fn sort_from(&mut self, pv_idx: usize) {
        if pv_idx >= self.records.len() {
            return;
        }
        // Stable sort preserves the previous iteration's ordering on equal scores.
        self.records[pv_idx..].sort_by(|a, b| {
            b.searched_this_iteration
                .cmp(&a.searched_this_iteration)
                .then_with(|| b.active_result().score.cmp(&a.active_result().score))
        });
    }
}

/// Iterative-deepening, single-threaded alpha-beta search with aspiration
/// windows, dedicated root search, principal variation search, transposition
/// table, history-based move ordering, and hybrid quiescence leaves.
pub fn search(position: &Board, depth: u8) -> SearchResult {
    LOCAL_TT.with_borrow_mut(|tt| {
        LOCAL_HISTORY.with_borrow_mut(|history| {
            tt.clear(1);
            history.clear();
            run_search::<true, true>(position, depth, Some(tt), Some(history))
        })
    })
}

/// Search using a caller-supplied transposition table.
pub fn search_with_tt(position: &Board, depth: u8, tt: &TranspositionTable) -> SearchResult {
    LOCAL_HISTORY.with_borrow_mut(|history| {
        history.clear();
        run_search::<true, true>(position, depth, Some(tt), Some(history))
    })
}

/// Search using caller-supplied transposition and history tables (persisting across UCI moves).
pub fn search_with_context(
    position: &Board, depth: u8, tt: &TranspositionTable, history: &mut HistoryTables,
) -> SearchResult {
    run_search::<true, true>(position, depth, Some(tt), Some(history))
}

/// The normal selective search, with recognized draws scored exactly zero.
pub fn search_with_zero_draws(position: &Board, depth: u8) -> SearchResult {
    LOCAL_TT.with_borrow_mut(|tt| {
        LOCAL_HISTORY.with_borrow_mut(|history| {
            tt.clear(1);
            history.clear();
            run_search::<false, true>(position, depth, Some(tt), Some(history))
        })
    })
}

/// Diagnostic search with zero draw scores and no selective QS pruning or TT/history ordering.
/// Keeps every evasion and capture eligible for independent minimax comparisons.
pub fn search_reference(position: &Board, depth: u8) -> SearchResult {
    run_search::<false, false>(position, depth, None, None)
}

fn run_search<const DRAW_JITTER: bool, const SELECTIVE: bool>(
    position: &Board, depth: u8, tt: Option<&TranspositionTable>, mut history: Option<&mut HistoryTables>,
) -> SearchResult {
    let mut board = position.clone();
    let mut nodes = 0;
    let mut pv = PvLine::new();

    if let Some(h) = history.as_deref_mut() {
        h.prepare_search(&board);
    }

    let (score, best_move) = if !SELECTIVE || depth == 0 {
        let mut sel_depth = 0;
        let (score, best_move) = search_node::<DRAW_JITTER, SELECTIVE, true>(
            &mut board,
            depth,
            0,
            None,
            Move::NULL,
            false,
            TtDepth::NONE_SCORE,
            0,
            -INFINITY,
            INFINITY,
            &mut pv,
            &mut nodes,
            &mut sel_depth,
            tt,
            history,
        );
        if let Some(mv) = best_move
            && pv.len == 0
        {
            pv.update(mv, &PvLine::new());
        }
        (score, best_move)
    } else {
        run_iterative_search::<DRAW_JITTER>(&mut board, depth, &mut nodes, &mut pv, tt.unwrap(), history.unwrap())
    };

    debug_assert!(board == *position, "search did not undo every move");
    SearchResult { best_move, score, nodes, pv }
}

fn run_iterative_search<const DRAW_JITTER: bool>(
    board: &mut Board, max_depth: u8, nodes: &mut u64, pv: &mut PvLine, tt: &TranspositionTable,
    history: &mut HistoryTables,
) -> (i32, Option<Move>) {
    let mut root_moves = RootMoves::new(board);
    if root_moves.is_empty() {
        let score = if board.in_check() { -MATE_SCORE } else { draw_score::<DRAW_JITTER>(*nodes) };
        return (score, None);
    }
    if board.is_draw(0) {
        let mv = root_moves.at(0).mv;
        pv.update(mv, &PvLine::new());
        return (draw_score::<DRAW_JITTER>(*nodes), Some(mv));
    }

    let mut eval_stability: i32 = 0;
    let mut pv_stability: i32 = 0;
    let mut last_best_rootmove: Option<RootMove> = None;
    let mut optimism: i32 = 0;

    for step in 1..=max_depth {
        root_moves.begin_depth_iteration(step);

        let pv_idx = 0;
        let prev_score = root_moves.aspiration_previous_score(pv_idx);
        let mut window = AspirationWindow::new(step, prev_score, eval_stability, pv_stability);
        let mut aspiration_depth = step;
        let mut root_pv_move = root_moves.at(pv_idx).mv;

        if prev_score.average > -MATE_SCORE {
            optimism = 50 * prev_score.average / (prev_score.average.abs() + 100);
        }
        let _ = optimism;

        loop {
            let root_effective_depth = aspiration_depth.max(1);
            let score = search_root::<DRAW_JITTER>(
                board,
                window.alpha,
                window.beta,
                root_effective_depth,
                pv_idx,
                root_pv_move,
                &mut root_moves,
                nodes,
                tt,
                history,
            );
            root_moves.sort_from(pv_idx);

            match window.on_result(score) {
                AspirationKind::FailHigh => {
                    if !is_decisive(score) && aspiration_depth > 3 {
                        aspiration_depth -= 1;
                    }
                    let winning_move = root_moves.at(pv_idx).mv;
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

        let top = *root_moves.at(0);
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
                && let Some(pos) = root_moves.records.iter().position(|rm| rm.mv == last.mv)
            {
                root_moves.records.remove(pos);
                root_moves.records.insert(0, last);
                root_moves.records[0].current.upperbound = false;
                root_moves.records[0].current.lowerbound = false;
            }
        } else {
            last_best_rootmove = Some(top);
        }
    }

    let winner = root_moves.at(0);
    let result = winner.active_result();
    *pv = result.pv;
    (result.score, Some(winner.mv))
}

#[allow(clippy::too_many_arguments)]
fn search_root<const DRAW_JITTER: bool>(
    board: &mut Board, mut alpha: i32, beta: i32, root_effective_depth: u8, pv_idx: usize, work_item: Move,
    root_moves: &mut RootMoves, nodes: &mut u64, tt: &TranspositionTable, history: &mut HistoryTables,
) -> i32 {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    let hash = board.hash();
    let entry = tt.read(hash, board.fiftymove_clock(), 0);
    let tt_move = entry.map(|e| e.mv).unwrap_or(Move::NULL);
    let priority_move = if work_item.is_present() { work_item } else { tt_move };

    let in_check = board.in_check();
    let raw_eval = if in_check {
        TtDepth::NONE_SCORE
    } else if let Some(entry) = &entry
        && entry.raw_eval != TtDepth::NONE_SCORE
    {
        entry.raw_eval
    } else {
        let ev = evaluate(board);
        if entry.is_none() {
            tt.write(hash, TtDepth::EVAL_ONLY, ev, TtDepth::NONE_SCORE, Bound::None, Move::NULL, 0, true, false);
        }
        ev
    };

    history.clear_child_killers(0);

    let mut picker = MainMovePicker::new(priority_move, true, [Move::NULL; 2], Move::NULL);
    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let mut bound = Bound::Upper;
    let mut move_count = 0usize;

    let mut quiet_moves = [Move::NULL; 32];
    let mut quiet_len = 0usize;
    let mut noisy_moves = [Move::NULL; 32];
    let mut noisy_len = 0usize;

    while let Some(mv) = picker.next(board, Some(history), false, 0) {
        if !root_moves.is_eligible_at_root(pv_idx, mv) {
            continue;
        }
        move_count += 1;
        let child_depth = root_effective_depth - 1;
        let nodes_before = *nodes;
        let mut child_pv = PvLine::new();
        let mut sel_depth = 1u16;

        let moved_piece = board.moved_piece(mv);
        history.continuation_history.remember(0, in_check, mv, moved_piece);
        make_move(board, mv, nodes, Some(tt));

        let score = if move_count == 1 {
            -search_node::<DRAW_JITTER, true, true>(
                board,
                child_depth,
                1,
                Some(mv.to()),
                mv,
                false,
                raw_eval,
                move_count,
                -beta,
                -alpha,
                &mut child_pv,
                nodes,
                &mut sel_depth,
                Some(tt),
                Some(history),
            )
            .0
        } else {
            let mut scout_pv = PvLine::new();
            let scout = -search_node::<DRAW_JITTER, true, false>(
                board,
                child_depth,
                1,
                Some(mv.to()),
                mv,
                true,
                raw_eval,
                move_count,
                -alpha - 1,
                -alpha,
                &mut scout_pv,
                nodes,
                &mut sel_depth,
                Some(tt),
                Some(history),
            )
            .0;
            if scout > alpha && beta > alpha + 1 {
                -search_node::<DRAW_JITTER, true, true>(
                    board,
                    child_depth,
                    1,
                    Some(mv.to()),
                    mv,
                    false,
                    raw_eval,
                    move_count,
                    -beta,
                    -alpha,
                    &mut child_pv,
                    nodes,
                    &mut sel_depth,
                    Some(tt),
                    Some(history),
                )
                .0
            } else {
                scout
            }
        };

        undo_move(board, mv);

        root_moves.record_searched_move(
            pv_idx,
            mv,
            score,
            alpha,
            beta,
            move_count == 1,
            *nodes - nodes_before,
            sel_depth,
            &child_pv,
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
                if pv_idx == 0 && mv != priority_move {
                    tt.write(hash, root_effective_depth as i32, raw_eval, score, Bound::Lower, mv, 0, true, false);
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
            board,
            history,
            0,
            root_effective_depth as i32,
            false,
            Move::NULL,
            0,
            best_move,
            best_score,
            beta,
            &quiet_moves[..quiet_len],
            &noisy_moves[..noisy_len],
        );
    }

    if pv_idx == 0 {
        let store_mv = if bound == Bound::Upper { Move::NULL } else { best_move };
        let store_depth = (root_effective_depth as i32 - i32::from(bound == Bound::Lower)).max(1);
        tt.write(hash, store_depth, raw_eval, best_score, bound, store_mv, 0, true, true);
    }

    best_score
}

#[allow(clippy::too_many_arguments)]
fn search_node<const DRAW_JITTER: bool, const SELECTIVE: bool, const PV: bool>(
    board: &mut Board, depth: u8, ply: u16, previous: Option<Square>, prior_move: Move, cut_node: bool,
    parent_eval: i32, parent_move_count: usize, mut alpha: i32, beta: i32, pv: &mut PvLine, nodes: &mut u64,
    sel_depth: &mut u16, tt: Option<&TranspositionTable>, mut history: Option<&mut HistoryTables>,
) -> (i32, Option<Move>) {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    if PV {
        pv.clear();
        *sel_depth = (*sel_depth).max(ply);
    }

    if depth == 0 {
        if ply == 0 && !board.has_legal_moves() {
            let score = if board.in_check() { -MATE_SCORE } else { draw_score::<DRAW_JITTER>(*nodes) };
            return (score, None);
        }
        return (
            quiescence::<DRAW_JITTER, SELECTIVE, PV>(
                board, 0, ply, previous, alpha, beta, nodes, sel_depth, tt, history,
            ),
            None,
        );
    }

    let draw = draw_score::<DRAW_JITTER>(*nodes);
    if ply > 0 && alpha < draw && board.upcoming_repetition(ply as usize) {
        alpha = draw;
        if alpha >= beta {
            return (alpha, None);
        }
    }

    let in_check = board.in_check();
    if ply > 0 {
        if board.is_draw(ply as isize) {
            if in_check && !board.has_legal_moves() {
                return (-MATE_SCORE + i32::from(ply), None);
            }
            return (draw, None);
        }
        if ply >= MAX_PLY {
            if in_check && !board.has_legal_moves() {
                return (-MATE_SCORE + i32::from(ply), None);
            }
            return (if in_check { draw } else { evaluate(board) }, None);
        }
    }

    let hash = board.hash();
    let entry = if SELECTIVE { tt.and_then(|t| t.read(hash, board.fiftymove_clock(), ply as isize)) } else { None };
    let mut tt_move = Move::NULL;
    let mut tt_pv = PV;

    if let Some(entry) = &entry {
        tt_move = entry.mv;
        tt_pv |= entry.tt_pv;

        if !PV
            && ply > 0
            && entry.depth >= depth as i32
            && entry.score != TtDepth::NONE_SCORE
            && board.fiftymove_clock() < 90
            && match entry.bound {
                Bound::Exact => true,
                Bound::Lower => entry.score >= beta,
                Bound::Upper => entry.score <= alpha,
                Bound::None => false,
            }
        {
            if entry.mv.is_quiet()
                && entry.score >= beta
                && parent_move_count < 4
                && let Some(h) = history.as_deref_mut()
            {
                let d = depth as i32;
                let quiet_bonus = (180 * d - 80).min(1_600);
                let cont_bonus = (95 * d - 70).min(1_200);
                let creates_threat = creates_threat_on_board(board, entry.mv);
                h.quiet_history.update(
                    board.all_threats(),
                    creates_threat,
                    board.side_to_move(),
                    entry.mv,
                    quiet_bonus,
                );
                h.update_continuation_histories(ply as usize, board.moved_piece(entry.mv), entry.mv.to(), cont_bonus);
            }
            return (entry.score, Some(entry.mv));
        }
    }

    let raw_eval = if in_check {
        TtDepth::NONE_SCORE
    } else if let Some(entry) = &entry
        && entry.raw_eval != TtDepth::NONE_SCORE
    {
        entry.raw_eval
    } else {
        let ev = evaluate(board);
        if SELECTIVE
            && entry.is_none()
            && let Some(t) = tt
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

    // Parent-move quiet history update from static-eval surprise.
    if SELECTIVE
        && ply > 0
        && !in_check
        && prior_move.is_quiet()
        && parent_eval != TtDepth::NONE_SCORE
        && (depth < 6 || entry.is_none())
        && let Some(h) = history.as_deref_mut()
    {
        let eval_delta = -(raw_eval + parent_eval);
        if eval_delta != 0 {
            let bonus = (eval_delta * 6).clamp(-150, 300);
            h.quiet_history.update(board.prior_threats(), false, !board.side_to_move(), prior_move, bonus);
        }
    }

    if let Some(h) = history.as_deref_mut() {
        h.clear_child_killers(ply as usize);
    }

    let (killers, countermove) = if let Some(h) = history.as_deref() {
        (h.killers(ply as usize), h.countermove(prior_move))
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

    let mut picker = MainMovePicker::new(if SELECTIVE { tt_move } else { Move::NULL }, false, killers, countermove);
    while let Some(mv) = picker.next(board, history.as_deref(), false, ply as usize) {
        move_count += 1;
        if ply == 0 && board.is_draw(0) {
            return (draw, Some(mv));
        }

        let moved_piece = board.moved_piece(mv);
        if let Some(h) = history.as_deref_mut() {
            h.continuation_history.remember(ply as usize, in_check, mv, moved_piece);
        }

        let mut child_pv = PvLine::new();
        make_move(board, mv, nodes, tt);
        let score = if !SELECTIVE || !PV || move_count == 1 {
            -search_node::<DRAW_JITTER, SELECTIVE, PV>(
                board,
                depth - 1,
                ply + 1,
                Some(mv.to()),
                mv,
                !cut_node,
                raw_eval,
                move_count,
                -beta,
                -alpha,
                &mut child_pv,
                nodes,
                sel_depth,
                tt,
                history.as_deref_mut(),
            )
            .0
        } else {
            let mut scout_pv = PvLine::new();
            let scout = -search_node::<DRAW_JITTER, SELECTIVE, false>(
                board,
                depth - 1,
                ply + 1,
                Some(mv.to()),
                mv,
                true,
                raw_eval,
                move_count,
                -alpha - 1,
                -alpha,
                &mut scout_pv,
                nodes,
                sel_depth,
                tt,
                history.as_deref_mut(),
            )
            .0;
            if mv.is_quiet()
                && let Some(h) = history.as_deref_mut()
            {
                let bonus = if scout > alpha { SCOUT_HISTORY_BONUS } else { -SCOUT_HISTORY_BONUS };
                h.update_continuation_histories(ply as usize, moved_piece, mv.to(), bonus);
            }
            if scout > alpha && beta > alpha + 1 {
                -search_node::<DRAW_JITTER, SELECTIVE, true>(
                    board,
                    depth - 1,
                    ply + 1,
                    Some(mv.to()),
                    mv,
                    false,
                    raw_eval,
                    move_count,
                    -beta,
                    -alpha,
                    &mut child_pv,
                    nodes,
                    sel_depth,
                    tt,
                    history.as_deref_mut(),
                )
                .0
            } else {
                scout
            }
        };
        undo_move(board, mv);

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
            if score > alpha {
                bound = Bound::Exact;
                if PV {
                    pv.update(mv, &child_pv);
                }
                if score >= beta {
                    bound = Bound::Lower;
                    break;
                }
                alpha = score;
                if SELECTIVE
                    && PV
                    && mv != tt_move
                    && let Some(t) = tt
                {
                    t.write(hash, depth as i32, raw_eval, score, Bound::Lower, mv, ply as isize, true, false);
                }
            } else if move_count == 1 && PV {
                pv.update(mv, &child_pv);
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
        let score = if in_check { -MATE_SCORE + i32::from(ply) } else { draw_score::<DRAW_JITTER>(*nodes) };
        return (score, None);
    }

    if SELECTIVE && let Some(h) = history {
        if best_score >= beta
            && let Some(best_mv) = best_move
        {
            update_histories_on_best_move(
                board,
                h,
                ply as usize,
                depth as i32,
                cut_node,
                prior_move,
                parent_move_count,
                best_mv,
                best_score,
                beta,
                &quiet_moves[..quiet_len],
                &noisy_moves[..noisy_len],
            );
        }

        // Child fail-low rewards the parent move that refuted the current side.
        if ply > 0 && bound == Bound::Upper && (cut_node || PV) && prior_move.is_present() {
            let d = depth as i32;
            let prior_threats = board.prior_threats();
            if prior_move.is_quiet() {
                let factor = 90
                    + (15 * parent_move_count as i32).min(225)
                    + 140 * (!in_check && best_score <= raw_eval - 50) as i32
                    + 300 * (parent_eval != TtDepth::NONE_SCORE && best_score <= -parent_eval - 70) as i32;
                let scaled_bonus = factor * (180 * d - 40).min(2_400) / 128;
                h.quiet_history.update(prior_threats, false, !board.side_to_move(), prior_move, scaled_bonus);
                if ply >= 2 {
                    let bonus = (150 * d - 45).min(1_350);
                    h.continuation_history.update(
                        ply as usize - 1,
                        1,
                        board.piece_on(prior_move.to()),
                        prior_move.to(),
                        bonus,
                    );
                }
            } else if prior_move.is_noisy() {
                let captured_type = board.captured_piece().piece_type();
                let bonus = (50 * d).min(650);
                h.noisy_history.update(
                    prior_threats,
                    board.piece_on(prior_move.to()),
                    prior_move.to(),
                    captured_type,
                    bonus,
                );
            }
        }
    }

    tt_pv |= ply > 0 && bound == Bound::Upper && move_count > 2;

    if SELECTIVE && let Some(t) = tt {
        let store_mv = if bound == Bound::Upper { Move::NULL } else { best_move.unwrap_or(Move::NULL) };
        let store_depth = (depth as i32 - i32::from(bound == Bound::Lower)).max(1);
        t.write(hash, store_depth, raw_eval, best_score, bound, store_mv, ply as isize, tt_pv, PV);
    }

    (best_score, best_move)
}

#[allow(clippy::too_many_arguments)]
fn update_histories_on_best_move(
    board: &Board, history: &mut HistoryTables, ply: usize, depth: i32, cut_node: bool, prior_move: Move,
    parent_move_count: usize, best_move: Move, best_score: i32, beta: i32, quiet_moves: &[Move], noisy_moves: &[Move],
) {
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

#[allow(clippy::too_many_arguments)]
fn quiescence<const DRAW_JITTER: bool, const SELECTIVE: bool, const PV: bool>(
    board: &mut Board, q_depth: i16, ply: u16, previous: Option<Square>, mut alpha: i32, beta: i32, nodes: &mut u64,
    sel_depth: &mut u16, tt: Option<&TranspositionTable>, mut history: Option<&mut HistoryTables>,
) -> i32 {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    debug_assert!(ply <= MAX_PLY && q_depth <= 0);
    *sel_depth = (*sel_depth).max(ply);
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

    let hash = board.hash();
    let tt_depth = i32::from(q_depth.max(RECAPTURE_ONLY_DEPTH));
    let entry = if SELECTIVE { tt.and_then(|t| t.read(hash, board.fiftymove_clock(), ply as isize)) } else { None };
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
            return entry.score;
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
                let ev = evaluate(board);
                if SELECTIVE
                    && entry.is_none()
                    && let Some(t) = tt
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
        if SELECTIVE
            && entry.is_none()
            && let Some(t) = tt
        {
            t.write(hash, tt_depth, raw_eval, best_score, Bound::Lower, Move::NULL, ply as isize, tt_pv, false);
        }
        return best_score;
    }
    alpha = alpha.max(best_score);

    let mut best_move = Move::NULL;
    let mut picker = QuiescenceMovePicker::new(if SELECTIVE { tt_move } else { Move::NULL }, in_check, q_depth >= -1);
    while let Some(mv) =
        picker.next(board, history.as_deref(), SELECTIVE && in_check && best_score > -MATE_THRESHOLD, ply as usize)
    {
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
        let moved_piece = board.moved_piece(mv);
        if let Some(h) = history.as_deref_mut() {
            h.continuation_history.remember(ply as usize, in_check, mv, moved_piece);
        }
        make_move(board, mv, nodes, tt);
        let score = -quiescence::<DRAW_JITTER, SELECTIVE, PV>(
            board,
            q_depth - 1,
            ply + 1,
            Some(mv.to()),
            -beta,
            -alpha,
            nodes,
            sel_depth,
            tt,
            history.as_deref_mut(),
        );
        undo_move(board, mv);
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
        return -MATE_SCORE + i32::from(ply);
    }

    if SELECTIVE
        && best_score >= beta
        && best_move.is_noisy()
        && let Some(h) = history
    {
        h.noisy_history.update(
            board.all_threats(),
            board.moved_piece(best_move),
            best_move.to(),
            board.type_on(best_move.capture_sq()),
            100,
        );
    }

    if SELECTIVE && let Some(t) = tt {
        let bound = if best_score >= beta { Bound::Lower } else { Bound::Upper };
        t.write(hash, tt_depth, raw_eval, best_score, bound, best_move, ply as isize, tt_pv, false);
    }

    best_score
}

const fn is_decisive(score: i32) -> bool {
    score.abs() >= MATE_THRESHOLD
}

fn make_move(board: &mut Board, mv: Move, nodes: &mut u64, tt: Option<&TranspositionTable>) {
    if let Some(t) = tt {
        t.prefetch(board.key_after(mv));
    }
    *nodes += 1;
    board.make_move(mv);
    if let Some(t) = tt {
        t.prefetch(board.hash());
    }
}

fn undo_move(board: &mut Board, mv: Move) {
    board.undo_move(mv);
}

fn draw_score<const DRAW_JITTER: bool>(nodes: u64) -> i32 {
    if DRAW_JITTER { (nodes % 5) as i32 - 2 } else { 0 }
}
