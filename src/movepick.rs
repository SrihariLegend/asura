use crate::{
    board::Board,
    evaluation::material_value,
    types::{MAX_MOVES, Move, MoveList, PieceType},
};

#[derive(Copy, Clone)]
enum Stage {
    GenerateNoisy,
    GoodNoisy,
    GenerateEvasions,
    Evasions,
    BadNoisy,
    GenerateChecks,
    Checks,
    Done,
}

/// Lazily orders legal QS candidates. Quiet evasions and quiet checks have
/// separate stages: skipping the former never disables the latter or captures.
pub(crate) struct QuiescenceMovePicker {
    moves: MoveList,
    scores: [i32; MAX_MOVES],
    bad_noisy: MoveList,
    bad_index: usize,
    quiet_index: usize,
    stage: Stage,
    in_check: bool,
    quiet_checks: bool,
}

impl QuiescenceMovePicker {
    pub(crate) fn new(in_check: bool, quiet_checks: bool) -> Self {
        Self {
            moves: MoveList::new(),
            scores: [0; MAX_MOVES],
            bad_noisy: MoveList::new(),
            bad_index: 0,
            quiet_index: 0,
            stage: Stage::GenerateNoisy,
            in_check,
            quiet_checks,
        }
    }

    pub(crate) fn next(&mut self, board: &Board, skip_quiet_evasions: bool) -> Option<Move> {
        loop {
            match self.stage {
                Stage::GenerateNoisy => {
                    board.append_noisy_moves(&mut self.moves);
                    for (index, &mv) in self.moves.iter().enumerate() {
                        let mut gain = if mv.is_capture() { material_value(board.type_on(mv.capture_sq())) } else { 0 };
                        if mv.is_promotion() {
                            gain += material_value(mv.promo_piece_type()) - material_value(PieceType::Pawn);
                        }
                        self.scores[index] = 16 * gain - board.type_on(mv.from()) as i32;
                    }
                    self.stage = Stage::GoodNoisy;
                }
                Stage::GoodNoisy => {
                    if let Some(mv) = self.take_best() {
                        if board.see_ge(mv, 0) {
                            return Some(mv);
                        }
                        self.bad_noisy.push(mv.from(), mv.to(), mv.kind());
                    } else {
                        self.stage = if self.in_check { Stage::GenerateEvasions } else { Stage::BadNoisy };
                    }
                }
                Stage::GenerateEvasions => {
                    if skip_quiet_evasions {
                        self.stage = Stage::BadNoisy;
                    } else {
                        self.moves = MoveList::new();
                        board.append_quiet_moves(&mut self.moves);
                        self.stage = Stage::Evasions;
                    }
                }
                Stage::Evasions => {
                    if !skip_quiet_evasions && self.quiet_index < self.moves.len() {
                        let mv = self.moves[self.quiet_index];
                        self.quiet_index += 1;
                        return Some(mv);
                    }
                    self.stage = Stage::BadNoisy;
                }
                Stage::BadNoisy => {
                    if self.bad_index < self.bad_noisy.len() {
                        let mv = self.bad_noisy[self.bad_index];
                        self.bad_index += 1;
                        return Some(mv);
                    }
                    self.stage = if !self.in_check && self.quiet_checks { Stage::GenerateChecks } else { Stage::Done };
                }
                Stage::GenerateChecks => {
                    self.moves = MoveList::new();
                    board.append_quiet_checks(&mut self.moves);
                    self.stage = Stage::Checks;
                }
                Stage::Checks => {
                    if self.quiet_index < self.moves.len() {
                        let mv = self.moves[self.quiet_index];
                        self.quiet_index += 1;
                        return Some(mv);
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }

    fn take_best(&mut self) -> Option<Move> {
        let mut best = None;
        let mut best_score = i32::MIN;
        for index in 0..self.moves.len() {
            if self.scores[index] > best_score {
                best = Some(index);
                best_score = self.scores[index];
            }
        }
        let index = best?;
        self.scores[index] = i32::MIN;
        Some(self.moves[index])
    }
}
