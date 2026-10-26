use std::{mem::MaybeUninit, ops::Index};

use super::{Bitboard, MAX_MOVES, Move, MoveKind, Square};

/// A fixed-capacity, stack-allocated list of moves, without search scores.
#[derive(Clone)]
pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    len: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        Self { moves: [MaybeUninit::uninit(); MAX_MOVES], len: 0 }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, from: Square, to: Square, kind: MoveKind) {
        assert!(self.len < MAX_MOVES, "move list capacity exceeded");
        self.moves[self.len].write(Move::new(from, to, kind));
        self.len += 1;
    }

    pub fn push_setwise(&mut self, from: Square, targets: Bitboard, kind: MoveKind) {
        for to in targets {
            self.push(from, to, kind);
        }
    }

    pub fn push_pawns_setwise(&mut self, offset: i8, targets: Bitboard, kind: MoveKind) {
        for to in targets {
            self.push(to.shift(-offset), to, kind);
        }
    }

    pub fn push_promotion_capture_setwise(&mut self, offset: i8, targets: Bitboard) {
        self.push_pawns_setwise(offset, targets, MoveKind::PromotionCaptureQ);
        self.push_pawns_setwise(offset, targets, MoveKind::PromotionCaptureR);
        self.push_pawns_setwise(offset, targets, MoveKind::PromotionCaptureB);
        self.push_pawns_setwise(offset, targets, MoveKind::PromotionCaptureN);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.as_slice().iter()
    }

    fn as_slice(&self) -> &[Move] {
        // SAFETY: push initializes moves[..len] and checks the array capacity.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr().cast(), self.len) }
    }
}

impl Index<usize> for MoveList {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}
