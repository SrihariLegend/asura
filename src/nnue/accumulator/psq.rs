use crate::{
    board::Board,
    nnue::{HEAVY_REGIMES, KING_BUCKET_LAYOUT},
    types::{Color, PieceType, Square},
};

#[inline]
pub fn heavy_regime(board: &Board, pov: Color) -> usize {
    let opp_queens = board.colored_pieces(!pov, PieceType::Queen);
    if !opp_queens.is_empty() {
        2
    } else if !(board.pieces(PieceType::Rook) | board.pieces(PieceType::Queen)).is_empty() {
        1
    } else {
        0
    }
}

#[inline]
pub fn orient_flip(pov: Color, ksq: Square) -> u8 {
    let mirror = if ksq.is_kingside() { 7u8 } else { 0u8 };
    let vflip = if pov == Color::Black { 56u8 } else { 0u8 };
    mirror ^ vflip
}

#[inline]
pub fn psq_row_index(color: Color, pt: PieceType, sq: Square, ksq: Square, regime: usize, pov: Color) -> usize {
    let flip = orient_flip(pov, ksq);
    let kb = KING_BUCKET_LAYOUT[(ksq as u8 ^ flip) as usize] as usize;
    let bucket_id = kb * HEAVY_REGIMES + regime;
    let rel_c = (color != pov) as usize;
    bucket_id * 768 + rel_c * 384 + (pt as usize) * 64 + ((sq as u8 ^ flip) as usize)
}

pub fn collect_psq_rows(board: &Board, pov: Color, out: &mut Vec<usize>) {
    let ksq = board.king_square(pov);
    let regime = heavy_regime(board, pov);
    for color in [Color::White, Color::Black] {
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            for sq in board.colored_pieces(color, pt) {
                out.push(psq_row_index(color, pt, sq, ksq, regime, pov));
            }
        }
    }
}
