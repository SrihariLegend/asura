use super::Board;
use crate::types::{Move, Piece, PieceType, Square};

impl Board {
    fn increment_stack(&mut self) {
        self.halfmove_number += 1;
        self.state_stack.push(self.state);
        self.state.keys.toggle_side();

        if self.en_passant() != Square::None {
            self.state.keys.toggle_en_passant(self.en_passant());
            self.state.en_passant = Square::None;
        }
    }

    /// Play a legal move and save the state needed by `undo_move`.
    ///
    /// The caller must obtain the move from move generation or validate it with
    /// `is_legal`. No evaluator or search state is updated here.
    pub fn make_move(&mut self, mv: Move) {
        debug_assert!(self.is_legal(mv));

        let from = mv.from();
        let to = mv.to();
        let piece = self.piece_on(from);
        let stm = self.side_to_move();

        self.increment_stack();
        self.state.captured = Piece::None;

        if mv.is_capture() || piece.piece_type() == PieceType::Pawn {
            self.state.fiftymove_clock = 0;
        } else {
            self.state.fiftymove_clock = self.state.fiftymove_clock.saturating_add(1);
        }

        if mv.is_castling() {
            let (rook_from, rook_to) = self.get_castling_rook(to);
            let rook = self.remove_piece(rook_from);
            self.remove_piece(from);
            self.add_piece(piece, to);
            self.add_piece(rook, rook_to);
        } else {
            self.remove_piece(from);

            if mv.is_capture() {
                self.state.captured = self.remove_piece(mv.capture_sq());
            }

            let placed = if mv.is_promotion() { Piece::new(stm, mv.promo_piece_type()) } else { piece };
            self.add_piece(placed, to);

            if mv.is_double_push() {
                self.state.en_passant = to ^ 8;
                self.state.keys.toggle_en_passant(self.en_passant());
            }
        }

        self.state.keys.toggle_castling(self.state.castling);
        self.state.castling.raw &= self.castling_rights[from] & self.castling_rights[to];
        self.state.keys.toggle_castling(self.state.castling);

        self.update_threats();
        self.validate_en_passant();
    }

    /// Undo the most recent move. Moves must be undone in reverse order.
    pub fn undo_move(&mut self, mv: Move) {
        assert!(!self.state_stack.is_empty(), "no move to undo");
        self.halfmove_number -= 1;

        let from = mv.from();
        let to = mv.to();
        let mover = self.remove_piece(to);
        let stm = self.side_to_move();

        if mv.is_castling() {
            let (rook_from, rook_to) = self.get_castling_rook(to);
            self.remove_piece(rook_to);
            self.add_piece(Piece::new(stm, PieceType::Rook), rook_from);
            self.add_piece(mover, from);
        } else {
            self.add_piece(if mv.is_promotion() { Piece::new(stm, PieceType::Pawn) } else { mover }, from);

            if mv.is_capture() {
                self.add_piece(self.captured_piece(), mv.capture_sq());
            }
        }

        self.state = self.state_stack.pop().unwrap();
    }
}
