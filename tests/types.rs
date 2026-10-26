use asura::types::{Bitboard, Color, File, Move, MoveKind, MoveList, Piece, PieceType, Rank, Square};

#[test]
fn squares_use_little_endian_rank_file_mapping() {
    assert_eq!(Square::A1 as u8, 0);
    assert_eq!(Square::H8 as u8, 63);
    for index in 0..64 {
        let square = Square::new(index);
        assert_eq!(Square::try_from(square.to_string().as_str()), Ok(square));
        assert_eq!(square.to_bb().popcount(), 1);
        assert_eq!(square.to_bb().lsb(), square);
        assert_eq!(square.flip_rank().flip_rank(), square);
        assert_eq!(Square::from_rank_file(square.rank() as u8, square.file() as u8), square);
    }
}

#[test]
fn bitboard_iteration_and_masks() {
    let mut board = Bitboard::default();
    assert!(board.next().is_none());
    board.set(Square::H8);
    board.set(Square::A1);
    assert!(board.is_multiple());
    assert_eq!(board.collect::<Vec<_>>(), [Square::A1, Square::H8]);
    board.clear(Square::H8);
    assert_eq!(board.popcount(), 1);
    assert_eq!(Bitboard::rank(Rank::R1).0, 0xff);
    assert_eq!(Bitboard::file(File::A).0, 0x0101_0101_0101_0101);
}

#[test]
fn move_encoding_roundtrips_all_squares_and_flags() {
    assert_eq!(std::mem::size_of::<Move>(), 2);
    let kinds = [
        MoveKind::Normal,
        MoveKind::DoublePush,
        MoveKind::Castling,
        MoveKind::Capture,
        MoveKind::EnPassant,
        MoveKind::PromotionN,
        MoveKind::PromotionB,
        MoveKind::PromotionR,
        MoveKind::PromotionQ,
        MoveKind::PromotionCaptureN,
        MoveKind::PromotionCaptureB,
        MoveKind::PromotionCaptureR,
        MoveKind::PromotionCaptureQ,
    ];
    for from in 0..64 {
        for to in 0..64 {
            for kind in kinds {
                let mv = Move::new(Square::new(from), Square::new(to), kind);
                assert_eq!(mv.from(), Square::new(from));
                assert_eq!(mv.to(), Square::new(to));
                assert_eq!(mv.kind(), kind);
                assert_eq!(mv.is_capture(), kind as u8 & 4 != 0);
                assert_eq!(mv.is_promotion(), kind as u8 & 8 != 0);
            }
        }
    }
}

#[test]
fn piece_encoding_roundtrips_color_and_type() {
    for color in [Color::White, Color::Black] {
        for piece_type in
            [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King]
        {
            let piece = Piece::new(color, piece_type);
            assert_eq!(piece.color(), color);
            assert_eq!(piece.piece_type(), piece_type);
            assert_eq!(Piece::try_from(piece.to_string().chars().next().unwrap()), Ok(piece));
        }
    }
}

#[test]
fn move_list_only_exposes_initialized_entries() {
    let mut moves = MoveList::new();
    assert!(moves.is_empty());
    assert_eq!(moves.iter().count(), 0);
    moves.push(Square::E2, Square::E4, MoveKind::DoublePush);
    moves.push_setwise(Square::G1, Square::F3.to_bb() | Square::H3.to_bb(), MoveKind::Normal);
    assert_eq!(moves.len(), 3);
    assert_eq!(moves[0], Move::new(Square::E2, Square::E4, MoveKind::DoublePush));
    assert_eq!(moves.clone().iter().copied().collect::<Vec<_>>(), moves.iter().copied().collect::<Vec<_>>());
}

#[test]
#[should_panic(expected = "move list capacity exceeded")]
fn move_list_rejects_overflow_instead_of_writing_out_of_bounds() {
    let mut moves = MoveList::new();
    for _ in 0..=asura::types::MAX_MOVES {
        moves.push(Square::E2, Square::E4, MoveKind::DoublePush);
    }
}
