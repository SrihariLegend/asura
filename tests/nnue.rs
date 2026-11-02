use std::sync::Arc;

use asura::{
    board::Board,
    nnue::{ModelParameters, NetworkState},
};

fn verify_nnperft_tree(
    board: &mut Board, depth: u8, inc_net: &mut NetworkState, ref_net: &mut NetworkState, checked_nodes: &mut usize,
) {
    let inc_eval = inc_net.evaluate_incremental(board, 15);
    let ref_eval = ref_net.evaluate_full_refresh_scalar(board, 15);
    assert_eq!(
        inc_eval,
        ref_eval,
        "Gate 1 failure: Incremental vs Full-Refresh mismatch at FEN {}",
        board.to_fen()
    );
    assert_eq!(
        &inc_net.acc[0].values[..],
        &ref_net.acc[0].values[..],
        "White accumulator mismatch at FEN {}",
        board.to_fen()
    );
    assert_eq!(
        &inc_net.acc[1].values[..],
        &ref_net.acc[1].values[..],
        "Black accumulator mismatch at FEN {}",
        board.to_fen()
    );
    assert_eq!(inc_net.acc[0].psqt, ref_net.acc[0].psqt, "White PSQT mismatch at FEN {}", board.to_fen());
    assert_eq!(inc_net.acc[1].psqt, ref_net.acc[1].psqt, "Black PSQT mismatch at FEN {}", board.to_fen());
    *checked_nodes += 1;

    if depth == 0 {
        return;
    }

    let moves = board.generate_all_moves();
    for &mv in moves.iter() {
        board.make_move(mv);
        verify_nnperft_tree(board, depth - 1, inc_net, ref_net, checked_nodes);
        board.undo_move(mv);

        // Verify undo restoration as well!
        let post_undo_inc = inc_net.evaluate_incremental(board, 15);
        assert_eq!(post_undo_inc, ref_eval, "Post-undo evaluation mismatch after undoing {:?}", mv);
    }
}

#[test]
fn gate1_nnperft_incremental_and_avx2_match_scalar_refresh_across_all_move_types() {
    let params = Arc::new(ModelParameters::new_deterministic_random(0xCAFE_BABE_1337_BEEF));
    let mut inc_net = NetworkState::new(params.clone());
    let mut ref_net = NetworkState::new(params);

    // Test suite covering:
    // 1. Starting position
    // 2. Kiwipete (castling, en passant, captures, promotions, king moves across e/d mirror, H3 Queen captures)
    // 3. Promotions & Underpromotions (N, B, R, Q) + Passed Pawns + Pawn Pairs
    // 4. En-Passant rank pin & 3-square mutation
    // 5. Queenless endgame & Percolation pawn-wall fortress
    // 6. Chess960 / FRC castling where King and Rook swap squares!
    let fens = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3),
        ("8/8/4k3/2p1p1p1/2P1P1P1/3K4/8/8 w - - 0 1", 3),
    ];

    let mut total_checked = 0usize;
    for (fen, depth) in fens {
        let mut board = Board::from_fen(fen).unwrap();
        verify_nnperft_tree(&mut board, depth, &mut inc_net, &mut ref_net, &mut total_checked);
    }

    // Also test Chess960 castling (King on f1, Rook on g1 swapping squares!)
    let mut frc_board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1RN w Gkq - 0 1").unwrap();
    verify_nnperft_tree(&mut frc_board, 2, &mut inc_net, &mut ref_net, &mut total_checked);

    assert!(total_checked > 5_000, "Expected >5,000 verified nnperft nodes, got {total_checked}");
}

#[test]
fn gate2_trainer_to_engine_bit_identity_lock_across_10000_positions() {
    use asura::nnue::PackedBoard;
    use std::path::Path;

    let nnue_path = Path::new("/tmp/gate2_oracle.nnue");
    let oracle_path = Path::new("/tmp/gate2_oracle_10k.bin");
    let data_path = Path::new("/home/tom/.cache/asura_spsa/nnue_lab/datasets/asura_bt4_64m.bin");
    if !nnue_path.exists() || !oracle_path.exists() || !data_path.exists() {
        return;
    }

    let params = Arc::new(ModelParameters::load_from_file(nnue_path).expect("Load Gate 2 .nnue checkpoint"));
    let mut inc_net = NetworkState::new(params.clone());
    let mut ref_net = NetworkState::new(params);

    let oracle_bytes = std::fs::read(oracle_path).unwrap();
    assert_eq!(oracle_bytes.len(), 10_000 * 4);

    use std::io::Read;
    let mut data_file = std::fs::File::open(data_path).unwrap();
    let mut raw_boards = vec![0u8; 10_000 * 32];
    data_file.read_exact(&mut raw_boards).unwrap();

    for i in 0..10_000 {
        let pb_bytes: &[u8; 32] = raw_boards[i * 32..(i + 1) * 32].try_into().unwrap();
        let pb = PackedBoard::from_le_bytes(pb_bytes);
        let board = pb.to_board();
        let gpu_eval = i32::from_le_bytes(oracle_bytes[i * 4..(i + 1) * 4].try_into().unwrap());

        let inc_eval = inc_net.evaluate_incremental(&board, 0);
        let ref_eval = ref_net.evaluate_full_refresh_scalar(&board, 0);
        assert_eq!(
            ref_eval, gpu_eval,
            "Gate 2 Mismatch at pos {i} (FEN {}): Rust scalar={ref_eval} vs CUDA GPU={gpu_eval}",
            board.to_fen()
        );
        assert_eq!(
            inc_eval, gpu_eval,
            "Gate 2 Mismatch at pos {i} (FEN {}): Rust AVX2 inc={inc_eval} vs CUDA GPU={gpu_eval}",
            board.to_fen()
        );
    }
}

#[test]
fn gate3_trained_atn1_network_sanity_and_compensation_audit() {
    use std::path::Path;
    let nnue_path = Path::new("/home/tom/dev/personal/asura/asura_atn1.nnue");
    if !nnue_path.exists() {
        return;
    }

    let params = Arc::new(ModelParameters::load_from_file(nnue_path).expect("Load trained ATN-1 .nnue"));
    let mut net = NetworkState::new(params);

    let start_b = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let start_ev = net.evaluate_incremental(&start_b, 0);

    let up_queen_b = Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let up_queen_ev = net.evaluate_incremental(&up_queen_b, 0);

    let down_rook_b = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1").unwrap();
    let down_rook_ev = net.evaluate_incremental(&down_rook_b, 0);

    // Realistic Middlegame Outpost Knight on d5 vs Paralyzed Bad Bishop on c8 (equal material: 2 Rooks + 1 Minor + 8 Pawns each):
    let outpost_vs_paralyzed =
        Board::from_fen("r1b2rk1/pp3ppp/3p4/2pNp3/2P1P3/3P4/PP3PPP/R4RK1 w - - 0 1").unwrap();
    let (outpost_ev, outpost_psqt, _) = net.evaluate_with_components(&outpost_vs_paralyzed, 0);

    println!(
        "Gate 3 Trained ATN-1 Evals: startpos={start_ev} cp, up_queen={up_queen_ev} cp, down_rook={down_rook_ev} cp, outpost_vs_paralyzed={outpost_ev} cp (psqt={outpost_psqt} cp)"
    );

    assert!(start_ev.abs() < 80, "Startpos eval should be near 0, got {start_ev}");
    assert!(up_queen_ev > 500, "Up Queen eval should be > +500 cp, got {up_queen_ev}");
    assert!(down_rook_ev < -250, "Down Rook eval should be < -250 cp, got {down_rook_ev}");
    assert!(
        outpost_ev > 40,
        "ATN-1 must reward dominant d5 Knight vs paralyzed c8 Bishop (> +40 cp on equal material): eval={outpost_ev} (psqt={outpost_psqt})"
    );

    // Evaluate STS 1500 (1-ply + Quiescence) comparing ATN-1 vs Material-Only baseline (`evaluation::evaluate`)
    let sts_path = Path::new("/home/tom/.cache/asura_spsa/nnue_lab/test_suites/sts_1500.epd");
    if sts_path.exists() {
        fn qsearch_atn1(board: &mut Board, net: &mut NetworkState, mut alpha: i32, beta: i32, depth: u8) -> i32 {
            let in_check = board.in_check();
            if !in_check {
                let stand_pat = net.evaluate_incremental(board, 0);
                if depth == 0 || stand_pat >= beta {
                    return stand_pat;
                }
                if stand_pat > alpha {
                    alpha = stand_pat;
                }
            } else if depth == 0 {
                return net.evaluate_incremental(board, 0);
            }
            let mut moves = asura::types::MoveList::new();
            if in_check {
                board.append_all_moves(&mut moves);
            } else {
                board.append_noisy_moves(&mut moves);
            }
            if in_check && moves.is_empty() {
                return -29_000;
            }
            for &mv in moves.iter() {
                if !in_check && !board.see_ge(mv, 0) {
                    continue;
                }
                board.make_move(mv);
                let score = -qsearch_atn1(board, net, -beta, -alpha, depth - 1);
                board.undo_move(mv);
                if score >= beta {
                    return score;
                }
                if score > alpha {
                    alpha = score;
                }
            }
            alpha
        }

        let content = std::fs::read_to_string(sts_path).unwrap();
        let mut atn1_pts = 0i32;
        let mut cat_pts = [0i32; 15];
        let mut total_pos = 0usize;

        for line in content.lines() {
            let Some(bm_pos) = line.find(" bm ") else { continue };
            let fen = format!("{} 0 1", line[..bm_pos].trim());
            let Ok(mut board) = Board::from_fen(&fen) else { continue };

            let extract_tag = |tag: &str| -> Vec<String> {
                let Some(p) = line.find(tag) else { return Vec::new() };
                let rest = &line[p + tag.len()..];
                let Some(end) = rest.find('"') else { return Vec::new() };
                rest[..end].split_whitespace().map(|s| s.to_string()).collect()
            };
            let pts_list = extract_tag("c8 \"");
            let uci_list = extract_tag("c9 \"");
            if pts_list.is_empty() || pts_list.len() != uci_list.len() {
                continue;
            }

            let moves = board.generate_all_moves();
            let mut best_atn1 = (i32::MIN, String::new());
            for &mv in moves.iter() {
                let uci = mv.to_uci(&board);
                board.make_move(mv);
                let s_atn1 = -qsearch_atn1(&mut board, &mut net, -30_000, 30_000, 4);
                board.undo_move(mv);
                if s_atn1 > best_atn1.0 {
                    best_atn1 = (s_atn1, uci);
                }
            }

            let cat_idx = (total_pos / 100).min(14);
            for (u, p_str) in uci_list.iter().zip(pts_list.iter()) {
                let p = p_str.parse::<i32>().unwrap_or(0);
                if u == &best_atn1.1 {
                    atn1_pts += p;
                    cat_pts[cat_idx] += p;
                }
            }
            total_pos += 1;
        }
        println!(
            "Gate 3 STS 1-15 ({total_pos} Grandmaster Positions, 14,990 max): ATN-1 = {atn1_pts} pts (vs Ref-A=7654, Ref-B=7356, Ref-C=6740)"
        );
        for (c, &pts) in cat_pts.iter().enumerate() {
            println!("  STS {:02}: {:4} / 1000", c + 1, pts);
        }

        let fort_path = Path::new("/home/tom/.cache/asura_spsa/nnue_lab/test_suites/fortresses.epd");
        if fort_path.exists() {
            let f_content = std::fs::read_to_string(fort_path).unwrap();
            let mut sum_abs = 0i64;
            let mut n_fort = 0usize;
            for line in f_content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                let fen = format!("{} {} {} {} 0 1", parts[0], parts[1], parts[2], parts[3]);
                if let Ok(b) = Board::from_fen(&fen)
                    && b.occupancies().popcount() <= 32
                {
                    let ev = net.evaluate_incremental(&b, 0);
                    sum_abs += ev.abs() as i64;
                    n_fort += 1;
                }
            }
            if n_fort > 0 {
                println!(
                    "  Fortresses ({n_fort} legal drawn fortresses, lower |eval| = better): ATN-1 mean |eval| = {:.1} cp",
                    sum_abs as f64 / n_fort as f64
                );
            }
        }

        // 4-Network Head-to-Head Audit on 10,000 Unseen Positions (`Ref-A` vs `Ref-B` vs `Ref-C` vs `Asura ATN-1`):
        let unseen_path = Path::new("/tmp/unseen_10k_4net.bin");
        if unseen_path.exists() {
            use asura::nnue::PackedBoard;
            let raw = std::fs::read(unseen_path).unwrap();
            let n_rec = raw.len() / 64;

            // `[0] = Ref-A`, `[1] = Ref-B`, `[2] = Ref-C`, `[3] = Asura ATN-1`
            let mut sq_err_bt4 = [0.0f64; 4];
            let mut sq_err_map = [0.0f64; 4];
            let mut wdl_err_map = [0.0f64; 4];
            let mut cusp_ok_bt4 = [0usize; 4];
            let mut cusp_ok_map = [0usize; 4];
            let mut cusp_total_bt4 = 0usize;
            let mut cusp_total_map = 0usize;
            let mut all_evals: Vec<([i32; 4], i32, i32)> = Vec::with_capacity(n_rec);

            for i in 0..n_rec {
                let base = i * 64;
                let pb = PackedBoard::from_le_bytes(raw[base..base + 32].try_into().unwrap());
                let v_sf = i32::from_le_bytes(raw[base + 32..base + 36].try_into().unwrap());
                let v_rk = i32::from_le_bytes(raw[base + 36..base + 40].try_into().unwrap());
                let v_th = i32::from_le_bytes(raw[base + 40..base + 44].try_into().unwrap());
                let b = pb.to_board();
                let v_atn1 = net.evaluate_incremental(&b, 0);

                let bt4 = pb.score_cp as i32;
                let map_cp = (35 * bt4 + 30 * v_sf + 20 * v_rk + 15 * v_th) / 100;
                let p_map = 1.0 / (1.0 + (-f64::from(map_cp) / 360.0).exp());

                let evs = [v_sf, v_rk, v_th, v_atn1];
                for k in 0..4 {
                    let d_bt4 = f64::from(evs[k] - bt4);
                    let d_map = f64::from(evs[k] - map_cp);
                    sq_err_bt4[k] += d_bt4 * d_bt4;
                    sq_err_map[k] += d_map * d_map;
                    let pk = 1.0 / (1.0 + (-f64::from(evs[k]) / 360.0).exp());
                    wdl_err_map[k] += (pk - p_map) * (pk - p_map);
                }

                if (pb.stm_wdl & 8) != 0 && bt4.abs() >= 35 {
                    cusp_total_bt4 += 1;
                    for k in 0..4 {
                        if (evs[k] >= 0) == (bt4 >= 0) {
                            cusp_ok_bt4[k] += 1;
                        }
                    }
                }
                if (pb.stm_wdl & 8) != 0 && map_cp.abs() >= 35 {
                    cusp_total_map += 1;
                    for k in 0..4 {
                        if (evs[k] >= 0) == (map_cp >= 0) {
                            cusp_ok_map[k] += 1;
                        }
                    }
                }
                all_evals.push((evs, bt4, map_cp));
            }

            let mut pairs_bt4_total = 0usize;
            let mut pairs_map_total = 0usize;
            let mut pairs_ok_bt4 = [0usize; 4];
            let mut pairs_ok_map = [0usize; 4];
            for i in (0..n_rec.saturating_sub(1)).step_by(2) {
                let (e1, b1, m1) = all_evals[i];
                let (e2, b2, m2) = all_evals[i + 1];
                if (b1 - b2).abs() >= 40 {
                    pairs_bt4_total += 1;
                    for k in 0..4 {
                        if (e1[k] > e2[k]) == (b1 > b2) {
                            pairs_ok_bt4[k] += 1;
                        }
                    }
                }
                if (m1 - m2).abs() >= 40 {
                    pairs_map_total += 1;
                    for k in 0..4 {
                        if (e1[k] > e2[k]) == (m1 > m2) {
                            pairs_ok_map[k] += 1;
                        }
                    }
                }
            }

            let names = ["Ref-Net-A ", "Ref-Net-B ", "Ref-Net-C ", "Asura ATN1"];
            println!("=== 4-NETWORK HEAD-TO-HEAD ON {n_rec} UNSEEN POSITIONS (VS REFERENCE ORACLE & VS ENSEMBLE MAP) ===");
            for k in 0..4 {
                println!(
                    "  {} | BT4 RMSE: {:6.2}cp (Rank {:5.2}%, Cusp {:5.2}%) | MAP RMSE: {:5.2}cp, WDL: {:.6} (Rank {:5.2}%, Cusp {:5.2}%)",
                    names[k],
                    (sq_err_bt4[k] / n_rec as f64).sqrt(),
                    100.0 * pairs_ok_bt4[k] as f64 / pairs_bt4_total.max(1) as f64,
                    100.0 * cusp_ok_bt4[k] as f64 / cusp_total_bt4.max(1) as f64,
                    (sq_err_map[k] / n_rec as f64).sqrt(),
                    wdl_err_map[k] / n_rec as f64,
                    100.0 * pairs_ok_map[k] as f64 / pairs_map_total.max(1) as f64,
                    100.0 * cusp_ok_map[k] as f64 / cusp_total_map.max(1) as f64,
                );
            }
        }
        assert!(
            atn1_pts > 7_356,
            "ATN-1 must surpass Ref-C (6740) and Ref-B (7356) on STS 1500, got {atn1_pts}"
        );

        // Gate 4: Full Alpha-Beta Search Integration with ATN-1 + Box 5 Hindsight Compensation Refund
        asura::nnue::load_global_model(nnue_path).unwrap();
        let search_b = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
        let res = asura::search::search(&search_b, 6);
        asura::nnue::clear_global_model();
        assert!(res.best_move.is_some() && search_b.is_legal(res.best_move.unwrap()));
        assert!(res.nodes > 100);
        println!(
            "Gate 4 Full Alpha-Beta Search (Depth 6 with ATN-1 + Box 5 Compensation): bestmove={}, score={} cp, nodes={}",
            res.best_move.unwrap().to_uci(&search_b),
            res.score,
            res.nodes
        );
    }
}
