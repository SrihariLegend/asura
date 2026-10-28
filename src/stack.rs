use std::ops::{Index, IndexMut};

use crate::types::{MAX_PLY, Move, Score, is_valid};

#[derive(Copy, Clone)]
pub struct StackEntry {
    pub mv: Move,
    pub excluded_move: Move,
    pub static_eval: i32,
    pub stat_score: i32,
    pub reduction: i32,
    pub laterality: i32,
    pub cutoff_count: i32,
    pub move_count: u16,
}

impl Default for StackEntry {
    fn default() -> Self {
        Self {
            mv: Move::NULL,
            excluded_move: Move::NULL,
            static_eval: Score::NONE,
            stat_score: 0,
            reduction: 0,
            laterality: 0,
            cutoff_count: 0,
            move_count: 0,
        }
    }
}

pub struct Stack {
    entries: [StackEntry; MAX_PLY + 4],
    pub nmp_min_ply: u16,
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            entries: [StackEntry::default(); MAX_PLY + 4],
            nmp_min_ply: 0,
        }
    }
}

impl Stack {
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_improving(&self, ply: usize, in_check: bool, static_eval: i32) -> bool {
        if in_check || !is_valid(static_eval) {
            return false;
        }
        if ply >= 2 && is_valid(self.entries[ply - 2].static_eval) {
            return static_eval > self.entries[ply - 2].static_eval;
        }
        if ply >= 4 && is_valid(self.entries[ply - 4].static_eval) {
            return static_eval > self.entries[ply - 4].static_eval;
        }
        false
    }
}

impl Index<usize> for Stack {
    type Output = StackEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl IndexMut<usize> for Stack {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.entries[index]
    }
}
