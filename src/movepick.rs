use crate::{
    board::Board,
    evaluation::material_value,
    history::HistoryTables,
    lookup::king_attacks,
    setwise::{bishop_attacks_setwise, knight_attacks_setwise, pawn_attacks_setwise, rook_attacks_setwise},
    types::{Bitboard, MAX_MOVES, Move, MoveList, PieceType},
};

const DEFER_QUIET_THRESHOLD: i32 = -1_000;

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum MainStage {
    HashMove,
    GenerateNoisy,
    GoodNoisy,
    PriorityQuiet,
    GenerateQuiet,
    Quiet,
    BadNoisy,
    Done,
}

/// Staged move picker for root and interior full-width nodes.
/// Combines TT move priority, SEE-split noisy stages with history-modulated
/// thresholds, a priority-quiet stage (two killers + countermove), scored
/// quiet moves, and progressive root re-scoring.
pub struct MainMovePicker {
    moves: MoveList,
    scores: [i32; MAX_MOVES],
    bad_noisy: MoveList,
    bad_index: usize,
    deferred_quiets: MoveList,
    deferred_index: usize,
    priority_quiets: [Move; 3],
    priority_scores: [i32; 3],
    priority_scored: bool,
    tt_move: Move,
    stage: MainStage,
    emitted_stage: MainStage,
    noisy_count: usize,
    is_root: bool,
}

impl MainMovePicker {
    pub const fn new(tt_move: Move, is_root: bool, killers: [Move; 2], countermove: Move) -> Self {
        Self {
            moves: MoveList::new(),
            scores: [0; MAX_MOVES],
            bad_noisy: MoveList::new(),
            bad_index: 0,
            deferred_quiets: MoveList::new(),
            deferred_index: 0,
            priority_quiets: [killers[0], killers[1], countermove],
            priority_scores: [i32::MIN; 3],
            priority_scored: false,
            tt_move,
            stage: if tt_move.is_present() { MainStage::HashMove } else { MainStage::GenerateNoisy },
            emitted_stage: MainStage::HashMove,
            noisy_count: 0,
            is_root,
        }
    }

    pub const fn stage(&self) -> MainStage {
        self.stage
    }

    pub const fn emitted_stage(&self) -> MainStage {
        self.emitted_stage
    }

    pub fn next(
        &mut self, board: &Board, history: Option<&HistoryTables>, skip_quiets: bool, ply: usize,
    ) -> Option<Move> {
        loop {
            match self.stage {
                MainStage::HashMove => {
                    self.stage = MainStage::GenerateNoisy;
                    if board.is_legal(self.tt_move) {
                        self.emitted_stage = MainStage::HashMove;
                        return Some(self.tt_move);
                    }
                }
                MainStage::GenerateNoisy => {
                    board.append_noisy_moves(&mut self.moves);
                    self.mark_excluded_noisy();
                    self.score_noisy(board, history);
                    self.stage = MainStage::GoodNoisy;
                }
                MainStage::GoodNoisy => {
                    if let Some((mv, _score)) = self.take_best() {
                        let threshold = if self.tt_move.is_quiet() && self.noisy_count > 2 {
                            1
                        } else {
                            let hist = history.map_or(0, |h| {
                                h.noisy_history.get(
                                    board.all_threats(),
                                    board.moved_piece(mv),
                                    mv.to(),
                                    board.type_on(mv.capture_sq()),
                                )
                            });
                            -hist / 100
                        };
                        if !board.see_ge(mv, threshold) {
                            self.bad_noisy.push(mv.from(), mv.to(), mv.kind());
                            continue;
                        }
                        if self.is_root {
                            self.score_noisy(board, history);
                        }
                        self.noisy_count += 1;
                        self.emitted_stage = MainStage::GoodNoisy;
                        return Some(mv);
                    }
                    self.stage = if skip_quiets {
                        MainStage::BadNoisy
                    } else if board.in_check() {
                        MainStage::GenerateQuiet
                    } else {
                        MainStage::PriorityQuiet
                    };
                }
                MainStage::PriorityQuiet => {
                    if skip_quiets {
                        self.stage = MainStage::BadNoisy;
                        continue;
                    }
                    if !self.priority_scored {
                        self.priority_scored = true;
                        self.score_priority_quiets(board, history, ply);
                    }
                    if let Some(mv) = self.take_best_priority() {
                        self.emitted_stage = MainStage::PriorityQuiet;
                        return Some(mv);
                    }
                    self.stage = MainStage::GenerateQuiet;
                }
                MainStage::GenerateQuiet => {
                    if skip_quiets {
                        self.stage = MainStage::BadNoisy;
                    } else {
                        self.moves = MoveList::new();
                        board.append_quiet_moves(&mut self.moves);
                        self.mark_excluded_quiets(board.in_check());
                        self.score_quiet(board, history, ply);
                        self.stage = MainStage::Quiet;
                    }
                }
                MainStage::Quiet => {
                    if skip_quiets {
                        self.stage = MainStage::BadNoisy;
                        continue;
                    }
                    if let Some((mv, score)) = self.take_best() {
                        if history.is_some()
                            && !board.in_check()
                            && score <= DEFER_QUIET_THRESHOLD
                            && is_threatened_by_lesser(board, mv)
                        {
                            self.deferred_quiets.push(mv.from(), mv.to(), mv.kind());
                            continue;
                        }
                        if self.is_root {
                            self.score_quiet(board, history, ply);
                        }
                        self.emitted_stage = MainStage::Quiet;
                        return Some(mv);
                    }
                    if self.deferred_index < self.deferred_quiets.len() {
                        let mv = self.deferred_quiets[self.deferred_index];
                        self.deferred_index += 1;
                        self.emitted_stage = MainStage::Quiet;
                        return Some(mv);
                    }
                    self.stage = MainStage::BadNoisy;
                }
                MainStage::BadNoisy => {
                    if self.bad_index < self.bad_noisy.len() {
                        let mv = self.bad_noisy[self.bad_index];
                        self.bad_index += 1;
                        self.emitted_stage = MainStage::BadNoisy;
                        return Some(mv);
                    }
                    self.stage = MainStage::Done;
                }
                MainStage::Done => return None,
            }
        }
    }

    fn mark_excluded_noisy(&mut self) {
        for index in 0..self.moves.len() {
            if self.moves[index] == self.tt_move {
                self.scores[index] = i32::MIN;
            } else {
                self.scores[index] = 0;
            }
        }
    }

    fn mark_excluded_quiets(&mut self, in_check: bool) {
        for index in 0..self.moves.len() {
            let mv = self.moves[index];
            let is_priority = !in_check
                && (mv == self.priority_quiets[0] || mv == self.priority_quiets[1] || mv == self.priority_quiets[2]);
            if mv == self.tt_move || is_priority {
                self.scores[index] = i32::MIN;
            } else {
                self.scores[index] = 0;
            }
        }
    }

    fn score_noisy(&mut self, board: &Board, history: Option<&HistoryTables>) {
        let threats = board.all_threats();
        for (index, &mv) in self.moves.iter().enumerate() {
            if self.scores[index] == i32::MIN {
                continue;
            }
            let captured = board.type_on(mv.capture_sq());
            let pt = board.type_on(mv.from());
            let mut gain = if mv.is_capture() { material_value(captured) } else { 0 };
            if mv.is_promotion() {
                gain += material_value(mv.promo_piece_type()) - material_value(PieceType::Pawn);
            }
            let hist = history.map_or(0, |h| h.noisy_history.get(threats, board.moved_piece(mv), mv.to(), captured));
            self.scores[index] = 16 * gain
                + hist
                + 4_000 * (mv.is_promotion() && mv.promo_piece_type() == PieceType::Queen) as i32
                + (200_000 - 20_000 * pt as i32) * board.in_check() as i32
                - pt as i32;
        }
    }

    fn score_priority_quiets(&mut self, board: &Board, history: Option<&HistoryTables>, ply: usize) {
        let Some(h) = history else { return };
        let ctx = QuietScoringContext::new(board);
        for i in 0..3 {
            let mv = self.priority_quiets[i];
            if mv.is_null()
                || mv.is_noisy()
                || mv == self.tt_move
                || (i > 0 && mv == self.priority_quiets[0])
                || (i > 1 && mv == self.priority_quiets[1])
                || !board.is_legal(mv)
            {
                self.priority_scores[i] = i32::MIN;
            } else {
                self.priority_scores[i] = ctx.score_move(board, h, ply, mv);
            }
        }
    }

    fn take_best_priority(&mut self) -> Option<Move> {
        let mut best = None;
        let mut best_score = i32::MIN;
        for i in 0..3 {
            if self.priority_scores[i] > best_score {
                best = Some(i);
                best_score = self.priority_scores[i];
            }
        }
        let idx = best?;
        self.priority_scores[idx] = i32::MIN;
        Some(self.priority_quiets[idx])
    }

    fn score_quiet(&mut self, board: &Board, history: Option<&HistoryTables>, ply: usize) {
        let Some(h) = history else {
            for index in 0..self.moves.len() {
                if self.scores[index] != i32::MIN {
                    self.scores[index] = 0;
                }
            }
            return;
        };
        let ctx = QuietScoringContext::new(board);
        for (index, &mv) in self.moves.iter().enumerate() {
            if self.scores[index] == i32::MIN {
                continue;
            }
            self.scores[index] = ctx.score_move(board, h, ply, mv);
        }
    }

    fn take_best(&mut self) -> Option<(Move, i32)> {
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
        Some((self.moves[index], best_score))
    }
}

pub(crate) struct QuietScoringContext {
    threats: Bitboard,
    threatened: [Bitboard; 6],
    offense: [Bitboard; 6],
    wall_pawns: Bitboard,
}

impl QuietScoringContext {
    pub(crate) fn new(board: &Board) -> Self {
        let threats = board.all_threats();
        let side = board.side_to_move();
        let occupancies = board.occupancies();
        let pawn_threats = board.piece_threats(PieceType::Pawn);

        let non_pawn_threats = board.piece_threats(PieceType::Knight)
            | board.piece_threats(PieceType::Bishop)
            | board.piece_threats(PieceType::Rook)
            | board.piece_threats(PieceType::Queen)
            | board.piece_threats(PieceType::King);

        let minor_threats =
            pawn_threats | board.piece_threats(PieceType::Knight) | board.piece_threats(PieceType::Bishop);
        let rook_threats = minor_threats | board.piece_threats(PieceType::Rook);
        let threatened = [Bitboard(0), pawn_threats, pawn_threats, minor_threats, rook_threats, Bitboard(0)];

        let knight_vulnerable = (board.colored_pieces(!side, PieceType::Bishop) & !threats)
            | board.colored_pieces(!side, PieceType::Rook)
            | board.colored_pieces(!side, PieceType::Queen);
        let bishop_vulnerable =
            board.colored_pieces(!side, PieceType::Rook) | board.colored_pieces(!side, PieceType::Queen);
        let queen_orth_vulnerable = board.colored_pieces(!side, PieceType::Bishop) & !threats;
        let queen_diag_vulnerable = board.colored_pieces(!side, PieceType::Rook) & !threats;

        let mut p = pawn_attacks_setwise(board.colors(!side), !side) & !threats;
        p |= pawn_threats & Bitboard::LEVER_RANKS[side] & !non_pawn_threats;

        let n = knight_attacks_setwise(knight_vulnerable) & !threats;
        let b = bishop_attacks_setwise(bishop_vulnerable, occupancies) & !threats;
        let r = Bitboard::file(board.king_square(!side).file()) & !threats;
        let q = (rook_attacks_setwise(queen_orth_vulnerable, occupancies)
            | bishop_attacks_setwise(queen_diag_vulnerable, occupancies))
            & !threats;
        let offense = [p, n, b, r, q, Bitboard(0)];

        let my_king = board.king_square(side);
        let wall_pawns = if Bitboard::HOME_ROWS[side].contains(my_king) {
            king_attacks(my_king) & board.pieces(PieceType::Pawn)
        } else {
            Bitboard(0)
        };

        Self { threats, threatened, offense, wall_pawns }
    }

    pub(crate) fn creates_threat(&self, pt: PieceType, mv: Move) -> bool {
        (pt as usize) < 6 && self.offense[pt].contains(mv.to())
    }

    fn score_move(&self, board: &Board, history: &HistoryTables, ply: usize, mv: Move) -> i32 {
        const ESCAPE: [i32; 6] = [0, 8_000, 8_000, 12_000, 18_000, 0];

        let side = board.side_to_move();
        let pawn_key = board.pawn_key();
        let pt = board.type_on(mv.from());
        let piece = board.moved_piece(mv);
        let creates_threat = self.creates_threat(pt, mv);

        1_500 * history.quiet_history.get(self.threats, creates_threat, side, mv) / 1_000
            + 1_000 * history.pawn_history.get(pawn_key, piece, mv.to()) / 1_000
            + 1_500 * history.conthist(ply, 1, piece, mv.to()) / 1_000
            + 1_000 * history.conthist(ply, 2, piece, mv.to()) / 1_000
            + 1_000 * history.conthist(ply, 4, piece, mv.to()) / 1_000
            + 1_000 * history.conthist(ply, 6, piece, mv.to()) / 1_000
            + ESCAPE[pt] * self.threatened[pt].contains(mv.from()) as i32
            + 10_000 * board.checking_squares(pt).contains(mv.to()) as i32
            - 8_000 * self.threatened[pt].contains(mv.to()) as i32
            + 3_500 * creates_threat as i32
            - 4_000 * self.wall_pawns.contains(mv.from()) as i32
    }
}

pub(crate) fn creates_threat_on_board(board: &Board, mv: Move) -> bool {
    let ctx = QuietScoringContext::new(board);
    ctx.creates_threat(board.type_on(mv.from()), mv)
}

fn is_threatened_by_lesser(board: &Board, mv: Move) -> bool {
    let pt = board.type_on(mv.from());
    let pawn_threats = board.piece_threats(PieceType::Pawn);
    let minor_threats = pawn_threats | board.piece_threats(PieceType::Knight) | board.piece_threats(PieceType::Bishop);
    let rook_threats = minor_threats | board.piece_threats(PieceType::Rook);
    let mask = match pt {
        PieceType::Knight | PieceType::Bishop => pawn_threats,
        PieceType::Rook => minor_threats,
        PieceType::Queen => rook_threats,
        _ => Bitboard(0),
    };
    mask.contains(mv.to())
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Stage {
    HashMove,
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
    tt_move: Move,
    stage: Stage,
    in_check: bool,
    quiet_checks: bool,
}

impl QuiescenceMovePicker {
    pub(crate) fn new(tt_move: Move, in_check: bool, quiet_checks: bool) -> Self {
        Self {
            moves: MoveList::new(),
            scores: [0; MAX_MOVES],
            bad_noisy: MoveList::new(),
            bad_index: 0,
            quiet_index: 0,
            tt_move,
            stage: if tt_move.is_present() { Stage::HashMove } else { Stage::GenerateNoisy },
            in_check,
            quiet_checks,
        }
    }

    pub(crate) fn next(
        &mut self, board: &Board, history: Option<&HistoryTables>, skip_quiet_evasions: bool, ply: usize,
    ) -> Option<Move> {
        loop {
            match self.stage {
                Stage::HashMove => {
                    self.stage = Stage::GenerateNoisy;
                    if board.is_legal(self.tt_move)
                        && (self.in_check
                            || self.tt_move.is_noisy()
                            || (self.quiet_checks && board.gives_check(self.tt_move)))
                    {
                        return Some(self.tt_move);
                    }
                }
                Stage::GenerateNoisy => {
                    let threats = board.all_threats();
                    board.append_noisy_moves(&mut self.moves);
                    for (index, &mv) in self.moves.iter().enumerate() {
                        if mv == self.tt_move {
                            self.scores[index] = i32::MIN;
                            continue;
                        }
                        let captured = board.type_on(mv.capture_sq());
                        let mut gain = if mv.is_capture() { material_value(captured) } else { 0 };
                        if mv.is_promotion() {
                            gain += material_value(mv.promo_piece_type()) - material_value(PieceType::Pawn);
                        }
                        let hist = history
                            .map_or(0, |h| h.noisy_history.get(threats, board.moved_piece(mv), mv.to(), captured));
                        self.scores[index] = 16 * gain + hist - board.type_on(mv.from()) as i32;
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
                        for (index, &mv) in self.moves.iter().enumerate() {
                            if mv == self.tt_move {
                                self.scores[index] = i32::MIN;
                            } else if let Some(h) = history {
                                let piece = board.moved_piece(mv);
                                self.scores[index] =
                                    h.quiet_history.get(board.all_threats(), false, board.side_to_move(), mv)
                                        + h.conthist(ply, 1, piece, mv.to())
                                        + h.conthist(ply, 2, piece, mv.to());
                            } else {
                                self.scores[index] = 0;
                            }
                        }
                        self.stage = Stage::Evasions;
                    }
                }
                Stage::Evasions => {
                    if !skip_quiet_evasions {
                        if history.is_some() {
                            if let Some(mv) = self.take_best() {
                                return Some(mv);
                            }
                        } else {
                            while self.quiet_index < self.moves.len() {
                                let mv = self.moves[self.quiet_index];
                                self.quiet_index += 1;
                                if mv != self.tt_move {
                                    return Some(mv);
                                }
                            }
                        }
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
                    self.quiet_index = 0;
                    self.stage = Stage::Checks;
                }
                Stage::Checks => {
                    while self.quiet_index < self.moves.len() {
                        let mv = self.moves[self.quiet_index];
                        self.quiet_index += 1;
                        if mv != self.tt_move {
                            return Some(mv);
                        }
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
