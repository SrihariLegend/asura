use asura::{
    board::Board,
    search::{MATE_SCORE, search_with_tt},
    transposition::{Bound, Entry, TranspositionTable, TtDepth},
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
fn normalizes_mate_scores_by_ply_and_discards_clock_incompatible_bounds() {
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

    // Four plies still fit at clock 96, including a mate on the 100th halfmove.
    assert_eq!(tt.read(hash, 96, 2).unwrap().score, node_score);
    // At 97 the cached line may no longer fit. Without knowing its zeroing
    // moves, discard the bound rather than manufacture a non-mate Exact score.
    for sign in [-1, 1] {
        tt.write(hash, 5, 12, sign * node_score, Bound::Exact, Move::NULL, 2, true, true);
        for ply in [0, 2, 5] {
            let incompatible = tt.read(hash, 97, ply).unwrap();
            assert_eq!(incompatible.score, TtDepth::NONE_SCORE);
            assert_eq!(incompatible.bound, Bound::None);
            assert_eq!(incompatible.raw_eval, 12);
        }
    }
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
fn concurrent_cluster_replacement_never_returns_another_keys_payload() {
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    // Distinct verification signatures, all in cluster zero. Four keys compete
    // for three slots, exercising replacement as well as concurrent updates.
    let keys = [0x11111, 0x22222, 0x33333, 0x44444];
    let expected: [Entry; 4] = std::array::from_fn(|i| Entry {
        mv: Move::new(Square::new(i as u8 + 8), Square::new(i as u8 + 24), MoveKind::Normal),
        score: 1000 + 100 * i as i32,
        raw_eval: -1000 - 100 * i as i32,
        depth: 10 + i as i32,
        bound: [Bound::Exact, Bound::Lower, Bound::Upper, Bound::Exact][i],
        tt_pv: i % 2 == 0,
    });
    let tt = TranspositionTable::new(1);
    let barrier = Barrier::new(4);
    let hits = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for thread in 0..4 {
            let (tt, barrier, hits, expected, keys) = (&tt, &barrier, &hits, &expected, &keys);
            scope.spawn(move || {
                barrier.wait();
                for iteration in 0..20_000 {
                    let i = (iteration + thread) % keys.len();
                    let e = expected[i];
                    tt.write(keys[i], e.depth, e.raw_eval, e.score, e.bound, e.mv, 0, e.tt_pv, true);
                    for j in 0..keys.len() {
                        if let Some(entry) = tt.read(keys[j], 0, 0) {
                            hits.fetch_add(1, Ordering::Relaxed);
                            assert_eq!(entry, expected[j], "wrong payload for key {:x}", keys[j]);
                        }
                    }
                }
            });
        }
    });
    assert!(hits.load(Ordering::Relaxed) > 1000, "the stress test must actually read entries");
}

#[test]
fn cross_clock_mates_and_zeroing_mates_never_become_fake_exact_scores() {
    for zeroing in [false, true] {
        let tt = TranspositionTable::new(1);
        let (from, to) = if zeroing {
            let fen = "7k/8/5K1p/8/8/8/3Q4/8 w - - 99 1";
            (fen, fen)
        } else {
            ("7k/8/5K2/8/8/8/3Q4/8 w - - 96 1", "7k/8/5K2/8/8/8/3Q4/8 w - - 99 1")
        };
        let from = Board::from_fen(from).unwrap();
        let to = Board::from_fen(to).unwrap();
        assert_eq!(from.hash(), to.hash());
        let proven = search_with_tt(&from, 3, &tt);
        assert_eq!(proven.score, MATE_SCORE - 3);
        let entry = tt.read(to.hash(), to.fiftymove_clock(), 0).unwrap();
        if zeroing && entry.score != TtDepth::NONE_SCORE {
            assert_eq!(entry.score, proven.score, "a zeroing move may preserve the proven mate");
        } else {
            assert_eq!(entry.score, TtDepth::NONE_SCORE);
            assert_eq!(entry.bound, Bound::None);
        }
        if !zeroing {
            for &mv in to.generate_all_moves().iter() {
                let mut child = to.clone();
                child.make_move(mv);
                assert!(child.draw_by_fifty_move_rule());
            }
            // The warm QS must not turn the expired mate into a huge cutoff.
            assert!(search_with_tt(&to, 0, &tt).score.abs() < 1000);
        }
    }
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
