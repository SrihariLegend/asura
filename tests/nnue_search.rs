use std::sync::Arc;

use asura::{
    board::Board,
    nnue::{ModelParameters, NetworkState, PSQT_BUCKETS, heavy_regime, psq_row_index},
    search::start_worker_search,
    thread::{PvLine, ThreadData},
    time::{Limits, TimeManager},
    transposition::{Bound, TranspositionTable, TtDepth},
    types::{Color, PieceType, Square},
};

#[test]
fn dense_positions_keep_all_ft_and_pawn_king_features() {
    use asura::nnue::{PK_CACHE_SIZE, PkCacheEntry, collect_active_rows, evaluate_pawn_king_subnet};

    let mut params = ModelParameters::new_zeroed();
    // A counting subnet makes its first beta output equal the active PK count.
    for row in params.pk_weights_1.iter_mut() {
        row[0] = 1;
    }
    params.pk_beta_weights[0][0] = 1;
    let mut cache: Box<[PkCacheEntry; PK_CACHE_SIZE]> =
        vec![PkCacheEntry::default(); PK_CACHE_SIZE].into_boxed_slice().try_into().ok().unwrap();
    let mut last_pk = 0;
    for (fen, ft_count, pk_count) in [
        ("r3b3/7k/1ppppr2/bPnnq1pp/pQNPPpPP/PBP1RN2/1B3P2/R2K4 w - - 0 1", 129, 38),
        ("8/kP1pP1pK/8/pP1pP1pP/8/8/pP1pP1pP/8 w - - 0 1", 58, 100),
    ] {
        let board = Board::from_fen(fen).unwrap();
        for pov in [Color::White, Color::Black] {
            let mut rows = Vec::new();
            collect_active_rows(&board, pov, &mut rows);
            assert_eq!(rows.len(), ft_count);
        }
        let (beta, _) = evaluate_pawn_king_subnet(&board, &params, &mut cache, &mut last_pk);
        assert_eq!(beta[0], pk_count);
    }
}

#[test]
fn tt_hits_do_not_reuse_rule50_scaled_evals_from_another_clock() {
    let mut params = ModelParameters::new_zeroed();
    for stack in &mut params.stacks {
        stack.fc3_biases[0] = 16384; // 600 cp before rule-50 scaling.
    }
    let params = Arc::new(params);
    // Include both directions across the damping threshold, within one hash bucket.
    for (old_clock, clock, cached_eval, expected) in [(24, 31, 570, 517), (23, 16, 577, 600), (16, 23, 600, 577)] {
        let old = Board::from_fen(&format!("7k/8/8/8/8/8/8/R3K3 w - - {old_clock} 1")).unwrap();
        let board = Board::from_fen(&format!("7k/8/8/8/8/8/8/R3K3 w - - {clock} 1")).unwrap();
        assert_eq!(old.hash(), board.hash());
        let mut net = NetworkState::new(Arc::clone(&params));
        assert_eq!(net.evaluate_incremental(&old, 0), cached_eval);
        assert_eq!(net.evaluate_incremental(&board, 0), expected);
        let tt = TranspositionTable::new(1);
        tt.write(
            old.hash(),
            TtDepth::EVAL_ONLY,
            cached_eval,
            TtDepth::NONE_SCORE,
            Bound::None,
            asura::types::Move::NULL,
            0,
            false,
            true,
        );
        let mut td = ThreadData::with_time_manager_and_nnue(
            &board,
            TimeManager::new(Limits::Depth(1), 1, 0),
            Some(&tt),
            None,
            Some(net),
        );
        start_worker_search(&mut td, &mut PvLine::new());
        assert_eq!(td.stack[0].static_eval, expected, "cached eval from clock {old_clock} reused at {clock}");
    }
}

#[test]
fn search_psqt_matches_the_network_component_in_both_perspectives() {
    let board = Board::from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
    let mut params = ModelParameters::new_zeroed();
    // One active white-perspective feature contributes -511 quantized units.
    // The full forward pass uses signed / 32: -15 cp, not floor(-511 / 32)=-16.
    // Ordinary material is +500 here, so a material fallback cannot pass.
    let row = psq_row_index(
        Color::White,
        PieceType::King,
        Square::E1,
        Square::E1,
        heavy_regime(&board, Color::White),
        Color::White,
    );
    params.ft_psqt[row] = [-511; PSQT_BUCKETS];
    let net = NetworkState::new(Arc::new(params));
    let mut td =
        ThreadData::with_time_manager_and_nnue(&board, TimeManager::new(Limits::Depth(1), 1, 0), None, None, Some(net));

    assert_eq!(td.evaluate().1, -15);
    assert_eq!(td.evaluate_psqt_only(), -15, "TT-hit compensation must use learned PSQT");

    td.board.make_null_move();
    assert_eq!(td.evaluate_psqt_only(), 15);
    assert_eq!(td.evaluate().1, 15);
    td.board.undo_null_move();
    assert_eq!(td.evaluate_psqt_only(), -15);

    let mut material_only =
        ThreadData::with_time_manager_and_nnue(&board, TimeManager::new(Limits::Depth(1), 1, 0), None, None, None);
    assert_eq!(material_only.evaluate(), (500, 500));
    assert_eq!(material_only.evaluate_psqt_only(), 500);
}
