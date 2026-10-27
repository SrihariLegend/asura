use crate::{board::Board, types::PieceType};

/// Shared centipawn scale for material evaluation and exchange estimates.
pub(crate) const fn material_value(piece: PieceType) -> i32 {
    match piece {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King | PieceType::None => 0,
    }
}

/// Temporary material-only evaluation, in centipawns for the side to move.
pub(crate) fn evaluate(board: &Board) -> i32 {
    let side = board.side_to_move();
    let mut score = 0;
    for piece in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let ours = board.colored_pieces(side, piece).popcount() as i32;
        let theirs = board.colored_pieces(!side, piece).popcount() as i32;
        score += (ours - theirs) * material_value(piece);
    }

    // Keep even artificial FEN positions outside the reserved mate-score range.
    score.clamp(-30_000, 30_000)
}
