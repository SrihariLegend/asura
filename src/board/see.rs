use crate::{
    evaluation::material_value,
    types::{Bitboard, Color, Move, PROMO_RANK, PieceType, Square},
};

impl super::Board {
    /// Estimate the material gain of a legal move by alternating least-valuable
    /// legal recaptures on its destination. Either side may decline an exchange.
    /// Pins and king safety are rechecked as occupancy changes; later pawn
    /// recaptures promote to queens. This is an ordering/pruning estimate, not
    /// a tactical search. Castling has exchange value zero.
    pub fn see_ge(&self, mv: Move, threshold: i32) -> bool {
        debug_assert!(self.is_legal(mv));
        if mv.is_castling() {
            return threshold <= 0;
        }
        let mut gains = [0; Square::NUM];
        gains[0] = if mv.is_capture() { material_value(self.type_on(mv.capture_sq())) } else { 0 };
        let mut victim = if mv.is_promotion() {
            gains[0] += material_value(mv.promo_piece_type()) - material_value(PieceType::Pawn);
            mv.promo_piece_type()
        } else {
            self.type_on(mv.from())
        };
        // The opponent can decline to recapture, so this is an upper bound.
        if gains[0] < threshold {
            return false;
        }

        let target = mv.to();
        let mut occupied = self.occupancies();
        occupied.clear(mv.from());
        if mv.is_en_passant() {
            occupied.clear(mv.capture_sq());
        }
        occupied.set(target);
        let mut side = !self.side_to_move();
        let mut depth = 0;
        while victim != PieceType::King {
            let Some((from, piece)) = self.exchange_attacker(target, occupied, side) else { break };
            depth += 1;
            let promotion = piece == PieceType::Pawn && target.rank() == PROMO_RANK[side];
            let placed = if promotion { PieceType::Queen } else { piece };
            let promotion_gain = material_value(placed) - material_value(piece);
            gains[depth] = material_value(victim) + promotion_gain - gains[depth - 1];
            occupied.clear(from);
            victim = placed;
            side = !side;
        }
        while depth > 0 {
            gains[depth - 1] = -(-gains[depth - 1]).max(gains[depth]);
            depth -= 1;
        }
        gains[0] >= threshold
    }

    fn exchange_attacker(&self, target: Square, occupied: Bitboard, side: Color) -> Option<(Square, PieceType)> {
        let attackers = self.attackers_to(target, occupied) & self.colors(side) & occupied;
        for piece in
            [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King]
        {
            for from in attackers & self.pieces(piece) {
                let after = occupied & !from.to_bb();
                let king = if piece == PieceType::King { target } else { self.king_square(side) };
                // All moving pieces end up on target. Their old squares have
                // been removed from occupancy, and the original target piece
                // must not survive as a ghost attacker of the recapturing king.
                let threats = self.attackers_to(king, after) & self.colors(!side) & after & !target.to_bb();
                if threats.is_empty() {
                    return Some((from, piece));
                }
            }
        }
        None
    }
}
