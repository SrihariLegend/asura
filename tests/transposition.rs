use asura::{
    board::Board,
    search::{MATE_SCORE, MATE_THRESHOLD, search_with_tt},
    transposition::{Bound, TranspositionTable, TtDepth},
    types::{Move, MoveKind, Square},
};

#[test]
fn stores_and_reads_entries_across_depth_tiers_while_preserving_incumbent_moves() {
    let tt = TranspositionTable::new(1);
    let board = Board::starting_position();
    let hash = board.hash();
    let mv = Move::new(Square::E2, Square::E4, MoveKind::DoublePush);

    assert!(tt.read(hash, 0, 0).is_none());

    // Verify round-trip ordering across all sub-horizon tiers.
    for tier in [
        TtDepth::EVAL_ONLY,
        TtDepth::QS_RECAPTURES,
        TtDepth::QS_PRE_RECAPTURES,
        TtDepth::QS_NO_CHECKS,
        TtDepth::QS_SHALLOW_CHECKS,
        TtDepth::QS_CHECKS,
    ] {
        tt.write(hash, tier, 45, TtDepth::NONE_SCORE, Bound::None, Move::NULL, 0, false, true);
        assert_eq!(tt.read(hash, 0, 0).unwrap().depth, tier);
    }

    tt.write(hash, TtDepth::EVAL_ONLY, 45, TtDepth::NONE_SCORE, Bound::None, Move::NULL, 0, false, true);
    let eval_hit = tt.read(hash, 0, 0).expect("eval-only entry must be readable");
    assert_eq!(eval_hit.depth, TtDepth::EVAL_ONLY);
    assert_eq!(eval_hit.raw_eval, 45);
    assert_eq!(eval_hit.score, TtDepth::NONE_SCORE);
    assert_eq!(eval_hit.bound, Bound::None);
    assert_eq!(eval_hit.mv, Move::NULL);

    // Deeper entry stores a move and bound.
    tt.write(hash, 6, 45, 120, Bound::Lower, mv, 0, true, false);
    let deep_hit = tt.read(hash, 0, 0).unwrap();
    assert_eq!(deep_hit.depth, 6);
    assert_eq!(deep_hit.score, 120);
    assert_eq!(deep_hit.raw_eval, 45);
    assert_eq!(deep_hit.bound, Bound::Lower);
    assert!(deep_hit.tt_pv);
    assert_eq!(deep_hit.mv, mv);

    // Shallower non-forced write in the same generation is rejected, preserving depth 6.
    tt.write(hash, 1, 10, -50, Bound::Upper, Move::NULL, 0, false, false);
    let preserved = tt.read(hash, 0, 0).unwrap();
    assert_eq!(preserved.depth, 6);
    assert_eq!(preserved.score, 120);
    assert_eq!(preserved.mv, mv);

    // Forced (PV) write overwrites even a deeper incumbent, while Move::NULL keeps the incumbent move.
    tt.write(hash, 1, 30, 80, Bound::Exact, Move::NULL, 0, true, true);
    let forced = tt.read(hash, 0, 0).unwrap();
    assert_eq!(forced.depth, 1);
    assert_eq!(forced.score, 80);
    assert_eq!(forced.raw_eval, 30);
    assert_eq!(forced.bound, Bound::Exact);
    assert_eq!(forced.mv, mv, "incumbent move must survive a Move::NULL update for the same key");
}

#[test]
fn normalizes_mate_scores_by_ply_and_downgrades_fifty_move_invalidated_mates() {
    let tt = TranspositionTable::new(1);
    let board = Board::starting_position();
    let hash = board.hash();

    // Mate in 6 plies from root found at ply = 2 has node-relative score MATE_SCORE - 6.
    // Stored in TT as MATE_SCORE - 4 (plies from this position to mate).
    let node_score = MATE_SCORE - 6;
    tt.write(hash, 5, 0, node_score, Bound::Exact, Move::NULL, 2, true, true);

    // Reading at ply = 2 restores MATE_SCORE - 6; reading at ply = 5 gives MATE_SCORE - 9.
    assert_eq!(tt.read(hash, 0, 2).unwrap().score, MATE_SCORE - 6);
    assert_eq!(tt.read(hash, 0, 5).unwrap().score, MATE_SCORE - 9);

    // When halfmove_clock = 97, only 3 plies remain before the 50-move draw,
    // which is fewer than the 4 plies to mate stored in the entry: downgrade!
    let downgraded = tt.read(hash, 97, 2).unwrap().score;
    assert_eq!(downgraded, MATE_THRESHOLD - 1);
}

#[test]
fn warm_transposition_table_accelerates_repeated_search_without_changing_tactics() {
    let tt = TranspositionTable::new(4);
    let board = Board::from_fen("7k/8/5K2/8/8/8/3Q4/8 w - - 0 1").unwrap();

    let cold = search_with_tt(&board, 3, &tt);
    assert_eq!(cold.score, 31_997);
    assert_eq!(cold.best_move.unwrap().to_uci(&board), "d2h6");

    tt.increment_age();
    let warm = search_with_tt(&board, 3, &tt);
    assert_eq!(warm.score, 31_997);
    assert_eq!(warm.best_move.unwrap().to_uci(&board), "d2h6");
    assert!(warm.nodes < cold.nodes, "warm TT should reduce searched nodes ({} vs {})", warm.nodes, cold.nodes);
}

#[test]
fn cluster_replacement_evicts_lowest_quality_slot_and_tracks_age() {
    let tt = TranspositionTable::new(1);
    // Keys with identical upper bits map to cluster 0, while distinct low 21 bits
    // occupy all 3 slots in that cluster.
    let k1 = 1u64;
    let k2 = 2u64;
    let k3 = 3u64;
    let k4 = 4u64;

    tt.write(k1, 8, 0, 10, Bound::Exact, Move::NULL, 0, true, false);
    tt.write(k2, 2, 0, 20, Bound::Upper, Move::NULL, 0, false, false);
    tt.increment_age();
    tt.write(k3, 5, 0, 30, Bound::Lower, Move::NULL, 0, false, false);

    // k2 has depth 2 and relative_age 1 -> quality = 2 - 4 = -2 (lowest).
    // k1 has depth 8 and relative_age 1 -> quality = 8 - 4 = 4.
    // k3 has depth 5 and relative_age 0 -> quality = 5.
    // Writing k4 must evict k2 while retaining k1 and k3.
    tt.write(k4, 4, 0, 40, Bound::Exact, Move::NULL, 0, true, false);

    assert!(tt.read(k1, 0, 0).is_some(), "deep entry k1 must survive");
    assert!(tt.read(k2, 0, 0).is_none(), "lowest-quality entry k2 must be evicted");
    assert!(tt.read(k3, 0, 0).is_some(), "current-generation entry k3 must survive");
    assert_eq!(tt.read(k4, 0, 0).unwrap().score, 40);
}
