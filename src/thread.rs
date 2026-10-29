use crate::{
    board::Board,
    history::HistoryTables,
    parameters::*,
    stack::Stack,
    time::{Limits, TimeManager},
    transposition::TranspositionTable,
    types::{MAX_PLY, Move, Score},
};

const MATE_SCORE: i32 = Score::MATE;
const INFINITY: i32 = Score::INFINITE;

/// Fixed-capacity principal variation buffer stored without heap allocation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PvLine {
    moves: [Move; MAX_PLY + 1],
    len: usize,
}

impl PvLine {
    pub const fn new() -> Self {
        Self { moves: [Move::NULL; MAX_PLY + 1], len: 0 }
    }

    pub const fn clear(&mut self) {
        self.len = 0;
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn update(&mut self, mv: Move, child: &[Move]) {
        let child_len = child.len().min(MAX_PLY);
        self.moves[0] = mv;
        self.moves[1..=child_len].copy_from_slice(&child[..child_len]);
        self.len = child_len + 1;
    }
}

impl Default for PvLine {
    fn default() -> Self {
        Self::new()
    }
}

/// Triangular principal variation table indexed by search ply.
#[derive(Clone)]
pub struct PrincipalVariationTable {
    table: Box<[[Move; MAX_PLY + 1]]>,
    len: [usize; MAX_PLY + 1],
}

impl PrincipalVariationTable {
    pub const fn clear(&mut self, ply: usize) {
        self.len[ply] = 0;
    }

    pub fn line(&self, ply: usize) -> &[Move] {
        &self.table[ply][..self.len[ply]]
    }

    pub fn update(&mut self, ply: usize, mv: Move) {
        self.table[ply][0] = mv;
        let child_len = if ply < MAX_PLY { self.len[ply + 1].min(MAX_PLY - ply) } else { 0 };
        for i in 0..child_len {
            self.table[ply][i + 1] = self.table[ply + 1][i];
        }
        self.len[ply] = child_len + 1;
    }
}

impl Default for PrincipalVariationTable {
    fn default() -> Self {
        Self {
            table: vec![[Move::NULL; MAX_PLY + 1]; MAX_PLY + 1].into_boxed_slice(),
            len: [0; MAX_PLY + 1],
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RootScore {
    pub last: i32,
    pub average: i32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AspirationKind {
    FailHigh,
    FailLow,
    Exact,
}

pub struct AspirationWindow {
    pub alpha: i32,
    pub beta: i32,
    pub delta: i32,
}

impl AspirationWindow {
    pub fn new(depth: u8, previous: RootScore, eval_stability: i32, pv_stability: i32) -> Self {
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

    pub fn on_result(&mut self, score: i32) -> AspirationKind {
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
pub struct IterationResult {
    pub score: i32,
    pub alpha: i32,
    pub beta: i32,
    pub depth: u8,
    pub upperbound: bool,
    pub lowerbound: bool,
    pub pv: PvLine,
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
pub struct RootMove {
    pub mv: Move,
    pub nodes: u64,
    pub sel_depth: u16,
    pub searched_this_iteration: bool,
    pub current: IterationResult,
    pub previous: IterationResult,
    pub average: i32,
}

impl RootMove {
    pub fn new(mv: Move) -> Self {
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

    pub const fn active_result(&self) -> &IterationResult {
        if self.searched_this_iteration { &self.current } else { &self.previous }
    }
}

#[derive(Default)]
pub struct RootMoves {
    pub records: Vec<RootMove>,
    pub best_move_changes: usize,
}

impl RootMoves {
    pub fn new(board: &Board) -> Self {
        Self::with_filter(board, None)
    }

    pub fn with_filter(board: &Board, filter: Option<&[Move]>) -> Self {
        let moves = board.generate_all_moves();
        let records = moves
            .iter()
            .copied()
            .filter(|mv| filter.is_none_or(|allowed| allowed.contains(mv)))
            .map(RootMove::new)
            .collect();
        Self { records, best_move_changes: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn at(&self, index: usize) -> &RootMove {
        &self.records[index]
    }

    pub fn begin_depth_iteration(&mut self, depth: u8) {
        self.best_move_changes = 0;
        for rec in &mut self.records {
            rec.previous = rec.current;
            rec.current.depth = depth;
            rec.searched_this_iteration = false;
        }
    }

    pub fn aspiration_previous_score(&self, pv_idx: usize) -> RootScore {
        let rec = &self.records[pv_idx];
        RootScore { last: rec.active_result().score, average: rec.average }
    }

    pub fn is_eligible_at_root(&self, pv_idx: usize, mv: Move) -> bool {
        self.records[pv_idx..].iter().any(|rec| rec.mv == mv)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_searched_move(
        &mut self, pv_idx: usize, mv: Move, score: i32, alpha_before: i32, beta: i32, first_candidate: bool,
        move_nodes: u64, sel_depth: u16, child_pv: &[Move],
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

    pub fn sort_from(&mut self, start_idx: usize) {
        self.records[start_idx..].sort_by(|a, b| {
            let sa = if a.searched_this_iteration { a.current.score } else { -INFINITY };
            let sb = if b.searched_this_iteration { b.current.score } else { -INFINITY };
            sb.cmp(&sa).then_with(|| b.previous.score.cmp(&a.previous.score))
        });
    }
}

/// Per-thread search state bundling board, stack, PV table, root moves, TT, and histories.
pub struct ThreadData<'a> {
    pub board: Board,
    pub stack: Stack,
    pub pv_table: PrincipalVariationTable,
    pub root_moves: RootMoves,
    pub time_manager: TimeManager,
    pub nodes: u64,
    pub sel_depth: u16,
    pub completed_depth: u8,
    pub stopped: bool,
    pub tt: Option<&'a TranspositionTable>,
    pub history: Option<&'a mut HistoryTables>,
}

impl<'a> ThreadData<'a> {
    pub fn new(position: &Board, tt: Option<&'a TranspositionTable>, history: Option<&'a mut HistoryTables>) -> Self {
        Self::with_limits(position, Limits::Infinite, DEFAULT_MOVE_OVERHEAD_MS, tt, history)
    }

    pub fn with_limits(
        position: &Board, limits: Limits, move_overhead_ms: u64, tt: Option<&'a TranspositionTable>,
        history: Option<&'a mut HistoryTables>,
    ) -> Self {
        let time_manager = TimeManager::new(limits, position.fullmove_number(), move_overhead_ms);
        Self::with_time_manager(position, time_manager, tt, history)
    }

    pub fn with_time_manager(
        position: &Board, time_manager: TimeManager, tt: Option<&'a TranspositionTable>,
        mut history: Option<&'a mut HistoryTables>,
    ) -> Self {
        let board = position.clone();
        if let Some(h) = history.as_deref_mut() {
            h.prepare_search(&board);
        }
        Self {
            board,
            stack: Stack::default(),
            pv_table: PrincipalVariationTable::default(),
            root_moves: RootMoves::default(),
            time_manager,
            nodes: 0,
            sel_depth: 0,
            completed_depth: 0,
            stopped: false,
            tt,
            history,
        }
    }

    #[inline]
    pub fn check_limits(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.time_manager.check_hard_limit(self.completed_depth, self.nodes) {
            self.stopped = true;
            return true;
        }
        false
    }

    pub fn make_move(&mut self, mv: Move) {
        if let Some(t) = self.tt {
            t.prefetch(self.board.key_after(mv));
        }
        self.nodes += 1;
        self.board.make_move(mv);
        if let Some(t) = self.tt {
            t.prefetch(self.board.hash());
        }
    }

    pub fn undo_move(&mut self, mv: Move) {
        self.board.undo_move(mv);
    }

    pub fn make_null_move(&mut self) {
        self.board.make_null_move();
        if let Some(t) = self.tt {
            t.prefetch(self.board.hash());
        }
    }

    pub fn undo_null_move(&mut self) {
        self.board.undo_null_move();
    }
}
