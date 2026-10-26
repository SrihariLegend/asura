use super::Board;
use crate::{
    lookup::between,
    types::{Bitboard, CastlingKind, Color, HOME_RANK, KING_TO_FILE, Piece, PieceType, ROOK_TO_FILE, Rank, Square},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParseFenError {
    MissingPlacementData,
    InvalidPlacement,
    InvalidPieceType,
    InvalidActiveColor,
    InvalidKings,
    InvalidPosition,
    InvalidCastling,
    InvalidEnPassant,
    InvalidMoveCounters,
    TooManyFields,
}

impl std::fmt::Display for ParseFenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MissingPlacementData => "missing piece placement",
            Self::InvalidPlacement => "piece placement must contain eight ranks of eight squares",
            Self::InvalidPieceType => "invalid piece character",
            Self::InvalidActiveColor => "active color must be w or b",
            Self::InvalidKings => "each side must have exactly one king",
            Self::InvalidPosition => "invalid king or pawn placement",
            Self::InvalidCastling => "invalid castling rights or missing castling pieces",
            Self::InvalidEnPassant => "invalid en passant square or missing double-pushed pawn",
            Self::InvalidMoveCounters => "invalid halfmove or fullmove counter",
            Self::TooManyFields => "too many FEN fields",
        })
    }
}

impl std::error::Error for ParseFenError {}

impl Board {
    /// Parse standard or Shredder-FEN, validating the core's structural invariants.
    ///
    /// Only placement and active color are required. Missing fields default to
    /// no castling, no en passant, a zero halfmove clock, and fullmove number one.
    /// Uncapturable en passant squares are normalized away.
    /// This does not prove that the position is reachable from the starting position.
    pub fn from_fen(fen: &str) -> Result<Self, ParseFenError> {
        let mut board = Self::empty();
        let mut parts = fen.split_whitespace();
        let rows: Vec<_> = parts.next().ok_or(ParseFenError::MissingPlacementData)?.split('/').collect();

        if rows.len() != 8 {
            return Err(ParseFenError::InvalidPlacement);
        }

        for (rank, row) in rows.iter().rev().enumerate() {
            let mut file = 0;
            for symbol in row.chars() {
                if let '1'..='8' = symbol {
                    file += symbol as u8 - b'0';
                } else {
                    if file >= 8 {
                        return Err(ParseFenError::InvalidPlacement);
                    }
                    let piece = symbol.try_into().map_err(|()| ParseFenError::InvalidPieceType)?;
                    board.add_piece(piece, Square::from_rank_file(rank as u8, file));
                    file += 1;
                }
                if file > 8 {
                    return Err(ParseFenError::InvalidPlacement);
                }
            }
            if file != 8 {
                return Err(ParseFenError::InvalidPlacement);
            }
        }

        for color in [Color::White, Color::Black] {
            if board.colored_pieces(color, PieceType::King).popcount() != 1 {
                return Err(ParseFenError::InvalidKings);
            }
        }

        let side = match parts.next() {
            Some("w") => Color::White,
            Some("b") => Color::Black,
            _ => return Err(ParseFenError::InvalidActiveColor),
        };

        board.set_castling(parts.next().unwrap_or("-"))?;
        board.state.en_passant = match parts.next().unwrap_or("-") {
            "-" => Square::None,
            square => square.try_into().map_err(|()| ParseFenError::InvalidEnPassant)?,
        };
        board.state.fiftymove_clock =
            parts.next().unwrap_or("0").parse().map_err(|_| ParseFenError::InvalidMoveCounters)?;
        let fullmove: usize = parts.next().unwrap_or("1").parse().map_err(|_| ParseFenError::InvalidMoveCounters)?;
        if fullmove == 0 {
            return Err(ParseFenError::InvalidMoveCounters);
        }
        board.halfmove_number = fullmove
            .checked_mul(2)
            .and_then(|n| n.checked_add(side as usize))
            .ok_or(ParseFenError::InvalidMoveCounters)?;
        if parts.next().is_some() {
            return Err(ParseFenError::TooManyFields);
        }

        // The side that just moved cannot have left its own king in check.
        if !(board.attackers_to(board.king_square(!side), board.occupancies()) & board.colors(side)).is_empty()
            || !(board.pieces(PieceType::Pawn) & Bitboard::BOTH_HOME_ROWS).is_empty()
        {
            return Err(ParseFenError::InvalidPosition);
        }

        let ep = board.en_passant();
        if ep != Square::None {
            let rank = if side == Color::White { Rank::R6 } else { Rank::R3 };
            if ep.rank() != rank
                || board.piece_on(ep) != Piece::None
                || board.piece_on(ep ^ 8) != Piece::new(!side, PieceType::Pawn)
                || board.piece_on(ep.shift(Square::UP[side])) != Piece::None
            {
                return Err(ParseFenError::InvalidEnPassant);
            }
        }

        board.update_threats();
        board.update_hash_keys();
        board.validate_en_passant();
        Ok(board)
    }

    fn set_castling(&mut self, rights: &str) -> Result<(), ParseFenError> {
        if rights == "-" {
            return Ok(());
        }

        for right in rights.chars() {
            let color = if right.is_ascii_uppercase() { Color::White } else { Color::Black };
            let king_from = self.king_square(color);
            let home = HOME_RANK[color];
            if king_from.rank() != home {
                return Err(ParseFenError::InvalidCastling);
            }

            let rooks = self.colored_pieces(color, PieceType::Rook) & Bitboard::rank(home);
            let rook_from = match right.to_ascii_uppercase() {
                'K' => rooks.filter(|&sq| sq > king_from).last(),
                'Q' => rooks.into_iter().find(|&sq| sq < king_from),
                file @ 'A'..='H' => Some(Square::from_rank_file(home as u8, file as u8 - b'A')),
                _ => None,
            }
            .ok_or(ParseFenError::InvalidCastling)?;

            if !rooks.contains(rook_from) || rook_from == king_from {
                return Err(ParseFenError::InvalidCastling);
            }
            let king_side = (rook_from > king_from) as usize;
            let kind = CastlingKind::KINDS[color][king_side];
            if self.castling().is_allowed(kind) {
                return Err(ParseFenError::InvalidCastling);
            }

            let king_to = Square::from_rank_file(home as u8, KING_TO_FILE[king_side] as u8);
            let rook_to = Square::from_rank_file(home as u8, ROOK_TO_FILE[king_side] as u8);
            self.set_castling_for(kind, king_from, king_to, rook_from, rook_to);
        }
        Ok(())
    }

    fn set_castling_for(
        &mut self, kind: CastlingKind, king_from: Square, king_to: Square, rook_from: Square, rook_to: Square,
    ) {
        self.state.castling.raw |= kind as u8;

        self.castling_rights[king_from] ^= kind as u8;
        self.castling_rights[rook_from] ^= kind as u8;

        self.castling_path[kind] |= between(king_from, king_to) | king_to.to_bb();
        self.castling_path[kind] |= between(rook_from, rook_to) | rook_to.to_bb();

        self.castling_path[kind] &= !king_from.to_bb();
        self.castling_path[kind] &= !rook_from.to_bb();

        self.castling_threat[kind] |= between(king_from, king_to) | king_from.to_bb() | king_to.to_bb();

        self.castling_rooks[kind] = rook_from;
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_count = 0;

            for file in 0..8 {
                let piece = self.piece_on(Square::from_rank_file(rank, file));
                if piece == Piece::None {
                    empty_count += 1;
                    continue;
                }

                if empty_count > 0 {
                    fen.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                fen.push_str(&piece.to_string());
            }

            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }

            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push_str(&self.side_to_move().to_string());
        fen.push(' ');
        fen.push_str(&self.state.castling.to_string(self));
        fen.push(' ');
        fen.push_str(&self.state.en_passant.to_string());
        fen.push(' ');
        fen.push_str(&self.fiftymove_clock().to_string());
        fen.push(' ');
        fen.push_str(&self.fullmove_number().to_string());
        fen
    }

    pub fn to_ascii(&self) -> String {
        let mut ascii = String::new();
        ascii.push_str("+---+---+---+---+---+---+---+---+\n");
        for rank in (0..8).rev() {
            ascii.push('|');
            for file in 0..8 {
                let square = Square::from_rank_file(rank, file);
                let piece = self.piece_on(square);
                let symbol = piece.try_into().unwrap_or(' ');
                ascii.push_str(&format!(" {symbol} |"));
            }
            ascii.push_str(&format!(" {}\n", rank + 1));
            ascii.push_str("+---+---+---+---+---+---+---+---+\n");
        }
        ascii.push_str("  a   b   c   d   e   f   g   h\n");
        ascii
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\nFEN: {}", self.to_ascii(), self.to_fen())
    }
}
