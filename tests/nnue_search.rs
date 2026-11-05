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
    // TT stores undamped raw_eval (600 cp) and correct_eval applies (200 - clock) / 200 for the current clock.
    for (old_clock, clock, cached_eval, expected) in [(24, 31, 600, 507), (23, 16, 600, 552), (16, 23, 600, 531)] {
        let old = Board::from_fen(&format!("7k/8/8/8/8/8/8/R3K3 w - - {old_clock} 1")).unwrap();
        let board = Board::from_fen(&format!("7k/8/8/8/8/8/8/R3K3 w - - {clock} 1")).unwrap();
        assert_eq!(old.hash(), board.hash());
        let mut net = NetworkState::new(Arc::clone(&params));
        assert_eq!(net.evaluate_incremental(&old, 0), cached_eval);
        assert_eq!(net.evaluate_incremental(&board, 0), cached_eval);
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

fn flip_fen_colors(fen: &str) -> String {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let ranks: Vec<&str> = parts[0].split('/').collect();
    let flipped_board: Vec<String> = ranks
        .into_iter()
        .rev()
        .map(|rank| {
            rank.chars()
                .map(|c| {
                    if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else {
                        c
                    }
                })
                .collect()
        })
        .collect();
    let stm = if parts[1] == "w" { "b" } else { "w" };
    format!("{} {} - - {} {}", flipped_board.join("/"), stm, parts[4], parts[5])
}

#[test]
fn atn1_evaluation_is_symmetric_under_color_inversion_with_unequal_square_complex_pawns() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("asura_atn1.nnue");
    let params = Arc::new(ModelParameters::load_from_file(path).expect("load bundled ATN-1"));
    let mut net = NetworkState::new(params);

    for fen in [
        "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1BP2/PP2B1PP/R2Q1RK1 w - - 0 9",
        "r2q1rk1/ppp1bppp/2np1n2/4p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 7",
        "8/5pk1/4p1p1/3pP1P1/3P1P2/6K1/8/8 w - - 0 40",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ] {
        let mirrored = flip_fen_colors(fen);
        let b1 = Board::from_fen(fen).unwrap();
        let b2 = Board::from_fen(&mirrored).unwrap();
        assert_eq!(
            net.evaluate_incremental(&b1, 0),
            net.evaluate_incremental(&b2, 0),
            "Color-inversion asymmetry between {fen} and {mirrored}"
        );
        assert_eq!(
            net.evaluate_full_refresh_scalar(&b1, 0),
            net.evaluate_full_refresh_scalar(&b2, 0),
            "Scalar color-inversion asymmetry between {fen} and {mirrored}"
        );
    }
}
