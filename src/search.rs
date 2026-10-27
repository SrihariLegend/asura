use crate::{
    board::Board,
    evaluation::evaluate,
    movepick::{MainMovePicker, QuiescenceMovePicker},
    types::{Move, Square},
};

pub(crate) const MATE_SCORE: i32 = 32_000;
const MAX_PLY: u16 = 256;
const MAX_PLY_USIZE: usize = MAX_PLY as usize;
const RECAPTURE_ONLY_DEPTH: i16 = -4;
pub(crate) const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;
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
/// windows, dedicated root search, principal variation search, and hybrid
/// quiescence leaves.
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
    let mut pv = PvLine::new();

    let (score, best_move) = if !SELECTIVE || depth == 0 {
        let mut sel_depth = 0;
        let (score, best_move) = search_node::<DRAW_JITTER, SELECTIVE, true>(
            &mut board,
            depth,
            0,
            None,
            -INFINITY,
            INFINITY,
            &mut pv,
            &mut nodes,
            &mut sel_depth,
        );
        if let Some(mv) = best_move
            && pv.len == 0
        {
            pv.update(mv, &PvLine::new());
        }
        (score, best_move)
    } else {
        run_iterative_search::<DRAW_JITTER>(&mut board, depth, &mut nodes, &mut pv)
    };

    debug_assert!(board == *position, "search did not undo every move");
    SearchResult { best_move, score, nodes, pv }
}

fn run_iterative_search<const DRAW_JITTER: bool>(
    board: &mut Board, max_depth: u8, nodes: &mut u64, pv: &mut PvLine,
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
        let mut root_pv_move = root_moves.at(pv_idx).mv;

        if prev_score.average > -MATE_SCORE {
            optimism = 50 * prev_score.average / (prev_score.average.abs() + 100);
        }
        let _ = optimism;

        // Note: fail-high depth reduction relies on the transposition table to
        // retain the deeper subtree score of the fail-high move across retries.
        // Without a TT, re-searching at `step - 1` would drop the horizon ply
        // that discovered the tactic.
        let root_effective_depth = step.max(1);
        loop {
            let score = search_root::<DRAW_JITTER>(
                board,
                window.alpha,
                window.beta,
                root_effective_depth,
                pv_idx,
                root_pv_move,
                &mut root_moves,
                nodes,
            );
            root_moves.sort_from(pv_idx);

            match window.on_result(score) {
                AspirationKind::FailHigh => {
                    let winning_move = root_moves.at(pv_idx).mv;
                    if winning_move.is_present() {
                        root_pv_move = winning_move;
                    }
                }
                AspirationKind::FailLow => {}
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
    root_moves: &mut RootMoves, nodes: &mut u64,
) -> i32 {
    debug_assert!(-INFINITY <= alpha && alpha < beta && beta <= INFINITY);
    let mut picker = MainMovePicker::new(work_item, true);
    let mut best_score = -INFINITY;
    let mut move_count = 0usize;

    while let Some(mv) = picker.next(board) {
        if !root_moves.is_eligible_at_root(pv_idx, mv) {
            continue;
        }
        move_count += 1;
        let cand_depth = root_effective_depth;
        let child_depth = cand_depth - 1;
        let nodes_before = *nodes;
        let mut child_pv = PvLine::new();
        let mut sel_depth = 1u16;

        make_move(board, mv, nodes);

        let score = if move_count == 1 {
            -search_node::<DRAW_JITTER, true, true>(
                board,
                child_depth,
                1,
                Some(mv.to()),
                -beta,
                -alpha,
                &mut child_pv,
                nodes,
                &mut sel_depth,
            )
            .0
        } else {
            let mut scout_pv = PvLine::new();
            let scout = -search_node::<DRAW_JITTER, true, false>(
                board,
                child_depth,
                1,
                Some(mv.to()),
                -alpha - 1,
                -alpha,
                &mut scout_pv,
                nodes,
                &mut sel_depth,
            )
            .0;
            if scout > alpha && beta > alpha + 1 {
                -search_node::<DRAW_JITTER, true, true>(
                    board,
                    child_depth,
                    1,
                    Some(mv.to()),
                    -beta,
                    -alpha,
                    &mut child_pv,
                    nodes,
                    &mut sel_depth,
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
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }
    }

    best_score
}

#[allow(clippy::too_many_arguments)]
fn search_node<const DRAW_JITTER: bool, const SELECTIVE: bool, const PV: bool>(
    board: &mut Board, depth: u8, ply: u16, previous: Option<Square>, mut alpha: i32, beta: i32, pv: &mut PvLine,
    nodes: &mut u64, sel_depth: &mut u16,
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
        return (quiescence::<DRAW_JITTER, SELECTIVE>(board, 0, ply, previous, alpha, beta, nodes, sel_depth), None);
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
        let best_move = (ply == 0).then_some(moves[0]);
        return (draw, best_move);
    }

    let mut best_score = -INFINITY;
    let mut best_move = None;
    for (index, &mv) in moves.iter().enumerate() {
        let mut child_pv = PvLine::new();
        make_move(board, mv, nodes);
        let score = if !SELECTIVE || !PV || index == 0 {
            -search_node::<DRAW_JITTER, SELECTIVE, PV>(
                board,
                depth - 1,
                ply + 1,
                Some(mv.to()),
                -beta,
                -alpha,
                &mut child_pv,
                nodes,
                sel_depth,
            )
            .0
        } else {
            let mut scout_pv = PvLine::new();
            let scout = -search_node::<DRAW_JITTER, SELECTIVE, false>(
                board,
                depth - 1,
                ply + 1,
                Some(mv.to()),
                -alpha - 1,
                -alpha,
                &mut scout_pv,
                nodes,
                sel_depth,
            )
            .0;
            if scout > alpha && beta > alpha + 1 {
                -search_node::<DRAW_JITTER, SELECTIVE, true>(
                    board,
                    depth - 1,
                    ply + 1,
                    Some(mv.to()),
                    -beta,
                    -alpha,
                    &mut child_pv,
                    nodes,
                    sel_depth,
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
                if PV {
                    pv.update(mv, &child_pv);
                }
                alpha = score;
                if alpha >= beta {
                    break;
                }
            } else if index == 0 && PV {
                pv.update(mv, &child_pv);
            }
        }
    }
    (best_score, best_move)
}

#[allow(clippy::too_many_arguments)]
fn quiescence<const DRAW_JITTER: bool, const SELECTIVE: bool>(
    board: &mut Board, q_depth: i16, ply: u16, previous: Option<Square>, mut alpha: i32, beta: i32, nodes: &mut u64,
    sel_depth: &mut u16,
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

    let mut best_score = if in_check { -INFINITY } else { evaluate(board) };
    if best_score >= beta {
        return best_score;
    }
    alpha = alpha.max(best_score);

    let mut picker = QuiescenceMovePicker::new(in_check, q_depth >= -1);
    while let Some(mv) = picker.next(board, SELECTIVE && in_check && best_score > -MATE_THRESHOLD) {
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
        let score = -quiescence::<DRAW_JITTER, SELECTIVE>(
            board,
            q_depth - 1,
            ply + 1,
            Some(mv.to()),
            -beta,
            -alpha,
            nodes,
            sel_depth,
        );
        undo_move(board, mv);
        best_score = best_score.max(score);
        alpha = alpha.max(score);
        if alpha >= beta {
            break;
        }
    }
    if in_check && best_score == -INFINITY { -MATE_SCORE + i32::from(ply) } else { best_score }
}

const fn is_decisive(score: i32) -> bool {
    score.abs() >= MATE_THRESHOLD
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
