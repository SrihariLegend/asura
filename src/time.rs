use std::time::Instant;

use crate::{
    parameters::*,
    types::{MAX_PLY, Move, Score, is_decisive},
};

/// Search resource constraints for a single root search.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Limits {
    Infinite,
    Depth(u8),
    MoveTime(u64),
    Nodes(u64),
    Mate(u8),
    Fischer { main_ms: u64, inc_ms: u64 },
    Cyclic { main_ms: u64, inc_ms: u64, moves_to_go: u64 },
}

/// Snapshot of root iteration signals used to scale the soft time target.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DynamicTimeState {
    pub cur_score: i32,
    pub prev_score: i32,
    pub best_move_nodes: u64,
    pub total_nodes: u64,
    pub pv_stability: i32,
    pub eval_stability: i32,
    pub best_move_changes: usize,
    pub legal_root_moves: usize,
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    limits: Limits,
    start_time: Instant,
    soft_bound_us: u64,
    hard_bound_us: u64,
    soft_nodes: u64,
    hard_nodes: u64,
    max_depth: u8,
    mate_moves: Option<u8>,
    search_moves: Option<Vec<Move>>,
    has_hard_budget: bool,
    has_soft_budget: bool,
}

impl TimeManager {
    pub fn new(limits: Limits, fullmove_number: usize, move_overhead_ms: u64) -> Self {
        let max_allowed_depth = (MAX_PLY - 1) as u8;
        let max_depth = match limits {
            Limits::Depth(d) => d.clamp(1, max_allowed_depth),
            _ => max_allowed_depth,
        };
        let mate_moves = match limits {
            Limits::Mate(m) => Some(m),
            _ => None,
        };
        let overhead_us = move_overhead_ms.saturating_mul(1_000);

        let (soft_bound_us, hard_bound_us, soft_nodes, hard_nodes, has_soft_budget, has_hard_budget) = match limits {
            Limits::Infinite | Limits::Depth(_) | Limits::Mate(_) => {
                (u64::MAX, u64::MAX, u64::MAX, u64::MAX, false, false)
            }
            Limits::Nodes(max_nodes) => {
                let n = max_nodes.max(1);
                (u64::MAX, u64::MAX, n, n, true, true)
            }
            Limits::MoveTime(ms) => {
                let usable = ms.saturating_mul(1_000).saturating_sub(overhead_us);
                (u64::MAX, usable, u64::MAX, u64::MAX, false, true)
            }
            Limits::Fischer { main_ms, inc_ms } => {
                let horizon = Self::expected_horizon(fullmove_number);
                let (soft, hard) = Self::compute_clock_bounds(
                    main_ms.saturating_mul(1_000),
                    inc_ms.saturating_mul(1_000),
                    horizon,
                    overhead_us,
                );
                (soft, hard, u64::MAX, u64::MAX, true, true)
            }
            Limits::Cyclic { main_ms, inc_ms, moves_to_go } => {
                let horizon = moves_to_go.max(1);
                let (soft, hard) = Self::compute_clock_bounds(
                    main_ms.saturating_mul(1_000),
                    inc_ms.saturating_mul(1_000),
                    horizon,
                    overhead_us,
                );
                (soft, hard, u64::MAX, u64::MAX, true, true)
            }
        };

        Self {
            limits,
            start_time: Instant::now(),
            soft_bound_us,
            hard_bound_us,
            soft_nodes,
            hard_nodes,
            max_depth,
            mate_moves,
            search_moves: None,
            has_hard_budget,
            has_soft_budget,
        }
    }

    pub fn with_overrides(
        mut self, max_depth: Option<u8>, mate_moves: Option<u8>, nodes: Option<u64>, search_moves: Option<Vec<Move>>,
    ) -> Self {
        let max_allowed_depth = (MAX_PLY - 1) as u8;
        if let Some(d) = max_depth {
            self.max_depth = d.clamp(1, max_allowed_depth);
        }
        if let Some(m) = mate_moves {
            self.mate_moves = Some(m);
        }
        if let Some(n) = nodes {
            let clamped = n.max(1);
            self.soft_nodes = clamped;
            self.hard_nodes = clamped;
            self.has_soft_budget = true;
            self.has_hard_budget = true;
        }
        self.search_moves = search_moves;
        self
    }

    /// Expected remaining moves in the game as a linear taper from 40 in the
    /// opening down to 22 in the middlegame and endgame.
    #[inline]
    pub const fn expected_horizon(fullmove_number: usize) -> u64 {
        let reduction = (fullmove_number as u64) / 3;
        let raw = HORIZON_OPENING_MOVES.saturating_sub(reduction);
        if raw < HORIZON_MIN_MOVES { HORIZON_MIN_MOVES } else { raw }
    }

    #[inline]
    pub const fn compute_clock_bounds(main_time: u64, inc_time: u64, horizon: u64, move_overhead: u64) -> (u64, u64) {
        let scale = TM_SCALE_UNIT as u64;
        let usable = main_time.saturating_sub(move_overhead);
        let h = if horizon == 0 { HORIZON_MIN_MOVES } else { horizon };
        let inc_share = inc_time.saturating_mul(INCREMENT_SHARE_PERMILLE) / scale;
        let soft_raw = (usable / h).saturating_add(inc_share);
        let soft_cap = usable.saturating_mul(SOFT_CLOCK_CAP_PERMILLE) / scale;
        let soft = if soft_raw < soft_cap { soft_raw } else { soft_cap };

        let hard_from_soft = soft.saturating_mul(HARD_SOFT_MULT_PERMILLE) / scale;
        let hard_cap = usable.saturating_mul(HARD_CLOCK_CAP_PERMILLE) / scale;
        let hard = if hard_from_soft < hard_cap { hard_from_soft } else { hard_cap };
        (soft, hard)
    }

    /// Compute the dynamic soft-time multiplier in 1_000 = 1.0x fixed-point units.
    pub fn compute_dynamic_scale(state: &DynamicTimeState) -> u64 {
        if is_decisive(state.cur_score) {
            return TM_SCALE_UNIT as u64;
        }

        // 1. Root node concentration.
        let total = state.total_nodes.max(1);
        let best_nodes = state.best_move_nodes.min(total);
        let frac = (best_nodes.saturating_mul(TM_SCALE_UNIT as u64) / total) as i64;
        let node_factor = TM_NODE_BASE + (TM_NODE_SLOPE * (TM_SCALE_UNIT - frac)) / TM_SCALE_UNIT;

        // 2. Cross-depth PV and evaluation stability.
        let pv_s = i64::from(state.pv_stability.clamp(0, TM_PV_STABILITY_MAX as i32));
        let ev_s = i64::from(state.eval_stability.clamp(0, TM_EVAL_STABILITY_MAX as i32));
        let stability_factor = (TM_STABILITY_BASE - TM_PV_STABILITY_STEP * pv_s - TM_EVAL_STABILITY_STEP * ev_s)
            .clamp(TM_STABILITY_FLOOR, TM_STABILITY_BASE);

        // 3. Intra-search best-move turnover.
        let changes = (state.best_move_changes as i64).min(TM_TURNOVER_MAX);
        let turnover_factor = TM_SCALE_UNIT + TM_TURNOVER_STEP * changes;

        // 4. Evaluation drop / crisis trend.
        let trend_factor = if state.prev_score == Score::NONE || is_decisive(state.prev_score) {
            TM_SCALE_UNIT
        } else {
            let diff = i64::from(state.prev_score - state.cur_score).clamp(TM_TREND_MIN_DIFF, TM_TREND_MAX_DIFF);
            TM_SCALE_UNIT + (TM_TREND_SLOPE_PERMILLE * diff) / TM_SCALE_UNIT
        };

        let combined_a = (node_factor * stability_factor) / TM_SCALE_UNIT;
        let combined_b = (turnover_factor * trend_factor) / TM_SCALE_UNIT;
        let scale = (combined_a * combined_b) / TM_SCALE_UNIT;
        scale.clamp(TM_MIN_SCALE, TM_MAX_SCALE) as u64
    }

    pub fn effective_soft_bound_us(&self, state: &DynamicTimeState) -> u64 {
        if self.soft_bound_us == u64::MAX {
            return u64::MAX;
        }
        let scale = Self::compute_dynamic_scale(state);
        let scaled = self.soft_bound_us.saturating_mul(scale) / (TM_SCALE_UNIT as u64);
        scaled.min(self.hard_bound_us)
    }

    pub fn effective_soft_bound_ms(&self, state: &DynamicTimeState) -> u64 {
        if self.soft_bound_us == u64::MAX {
            return u64::MAX;
        }
        self.effective_soft_bound_us(state) / 1_000
    }

    pub fn should_stop_soft_us(&self, elapsed_us: u64, state: &DynamicTimeState) -> bool {
        if !self.has_soft_budget {
            return false;
        }
        if state.total_nodes >= self.soft_nodes {
            return true;
        }
        if self.soft_bound_us == u64::MAX {
            return false;
        }
        if state.legal_root_moves == 1 && elapsed_us >= SINGLE_MOVE_SOFT_CAP_MS * 1_000 {
            return true;
        }
        elapsed_us >= self.effective_soft_bound_us(state)
    }

    pub fn should_stop_soft(&self, elapsed_ms: u64, state: &DynamicTimeState) -> bool {
        self.should_stop_soft_us(elapsed_ms.saturating_mul(1_000), state)
    }

    pub fn should_stop_soft_now(&self, state: &DynamicTimeState) -> bool {
        self.should_stop_soft_us(self.elapsed_us(), state)
    }

    #[inline]
    pub fn check_hard_limit(&self, completed_depth: u8, nodes: u64) -> bool {
        if !self.has_hard_budget || completed_depth == 0 {
            return false;
        }
        if nodes >= self.hard_nodes {
            return true;
        }
        if self.hard_bound_us != u64::MAX && (nodes & TIME_CHECK_NODE_MASK) == 0 {
            return self.elapsed_us() >= self.hard_bound_us;
        }
        false
    }

    #[inline]
    pub fn elapsed_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    pub const fn limits(&self) -> Limits {
        self.limits
    }

    pub const fn max_depth(&self) -> u8 {
        self.max_depth
    }

    pub const fn mate_moves(&self) -> Option<u8> {
        self.mate_moves
    }

    pub fn search_moves(&self) -> Option<&[Move]> {
        self.search_moves.as_deref()
    }

    pub const fn soft_bound_ms(&self) -> u64 {
        if self.soft_bound_us == u64::MAX { u64::MAX } else { self.soft_bound_us / 1_000 }
    }

    pub const fn hard_bound_ms(&self) -> u64 {
        if self.hard_bound_us == u64::MAX { u64::MAX } else { self.hard_bound_us / 1_000 }
    }

    pub const fn has_hard_budget(&self) -> bool {
        self.has_hard_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_horizon_tapers_from_opening_to_endgame_floor() {
        assert_eq!(TimeManager::expected_horizon(1), 40);
        assert_eq!(TimeManager::expected_horizon(18), 34);
        assert_eq!(TimeManager::expected_horizon(54), 22);
        assert_eq!(TimeManager::expected_horizon(100), 22);
    }

    #[test]
    fn clock_bounds_respect_overhead_soft_cap_and_hard_cap() {
        let tm = TimeManager::new(Limits::Fischer { main_ms: 60_000, inc_ms: 1_000 }, 1, 15);
        assert_eq!(tm.soft_bound_ms(), 2_166);
        assert_eq!(tm.hard_bound_ms(), 11_186);

        let cyclic = TimeManager::new(Limits::Cyclic { main_ms: 10_000, inc_ms: 5_000, moves_to_go: 1 }, 20, 0);
        assert_eq!(cyclic.soft_bound_ms(), 1_450);
        assert_eq!(cyclic.hard_bound_ms(), 7_486);
    }

    #[test]
    fn dynamic_scale_shrinks_on_stable_easy_moves_and_expands_on_unstable_score_drops() {
        let calm = DynamicTimeState {
            cur_score: 50,
            prev_score: 30,
            best_move_nodes: 950,
            total_nodes: 1_000,
            pv_stability: 6,
            eval_stability: 5,
            best_move_changes: 0,
            legal_root_moves: 20,
        };
        let crisis = DynamicTimeState {
            cur_score: -150,
            prev_score: 50,
            best_move_nodes: 150,
            total_nodes: 1_000,
            pv_stability: 0,
            eval_stability: 0,
            best_move_changes: 4,
            legal_root_moves: 20,
        };
        let scale_calm = TimeManager::compute_dynamic_scale(&calm);
        let scale_crisis = TimeManager::compute_dynamic_scale(&crisis);
        assert!(scale_calm < 600, "calm position should shrink soft budget below 0.6x (got {scale_calm})");
        assert!(scale_crisis > 3_000, "crisis position should expand soft budget above 3.0x (got {scale_crisis})");
    }

    #[test]
    fn hard_limit_exempts_depth_one_and_stops_once_completed_depth_is_positive() {
        let tm = TimeManager::new(Limits::Nodes(500), 1, 15);
        assert!(!tm.check_hard_limit(0, 1_000), "depth 1 (completed_depth == 0) must never abort");
        assert!(!tm.check_hard_limit(1, 499));
        assert!(tm.check_hard_limit(1, 500));
    }
}
