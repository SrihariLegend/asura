use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use asura::{
    board::Board,
    history::HistoryTables,
    nnue::{ModelParameters, NetworkState, PackedBoard, load_global_model},
    parameters::{set_spsa_active, tunables},
    search::search_with_limits,
    time::Limits,
    transposition::TranspositionTable,
    types::Move,
};

struct MoveOracle {
    mv: Move,
    obj_q_cp: i32,
    min_style_q_cp: i32,
    draw_permille: i32,
    def_difficulty_cp: i32,
}

struct PositionOracle {
    board: Board,
    v_root_cp: i32,
    best_obj_q_cp: i32,
    moves: Vec<MoveOracle>,
}

#[derive(Copy, Clone)]
struct ParamSpec {
    name: &'static str,
    step: i64,
    min: i64,
    max: i64,
}

fn hadamard_16() -> [[f64; 16]; 16] {
    let mut h = [[1.0; 16]; 16];
    let mut n = 1;
    while n < 16 {
        for i in 0..n {
            for j in 0..n {
                let v = h[i][j];
                h[i + n][j] = v;
                h[i][j + n] = v;
                h[i + n][j + n] = -v;
            }
        }
        n *= 2;
    }
    h
}

fn build_oracle_suite(params: Arc<ModelParameters>) -> Vec<PositionOracle> {
    let data_path = Path::new("/home/tom/.cache/asura_spsa/nnue_lab/datasets/asura_bt4_64m.bin");
    use std::io::Read;
    let mut f = std::fs::File::open(data_path).expect("Open dataset");
    let mut raw = vec![0u8; 40_000 * 32];
    f.read_exact(&mut raw).unwrap();

    let mut candidates = Vec::with_capacity(240);
    let (mut n_w, mut n_b, mut n_l) = (0usize, 0usize, 0usize);

    for i in (0..40_000).step_by(17) {
        if n_w >= 80 && n_b >= 80 && n_l >= 80 {
            break;
        }
        let pb = PackedBoard::from_le_bytes(raw[i * 32..(i + 1) * 32].try_into().unwrap());
        let pcs = pb.occ.count_ones();
        if !(10..=28).contains(&pcs) {
            continue;
        }
        let sc = pb.score_cp as i32;
        let b = pb.to_board();
        if b.in_check() || b.generate_all_moves().len() < 8 {
            continue;
        }
        if (90..=280).contains(&sc) && n_w < 80 {
            candidates.push((b, sc));
            n_w += 1;
        } else if (-50..=50).contains(&sc) && n_b < 80 {
            candidates.push((b, sc));
            n_b += 1;
        } else if (-280..=-90).contains(&sc) && n_l < 80 {
            candidates.push((b, sc));
            n_l += 1;
        }
    }

    let total = candidates.len();
    let out: Arc<Mutex<Vec<Option<PositionOracle>>>> = Arc::new(Mutex::new((0..total).map(|_| None).collect()));
    let next_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    std::thread::scope(|s| {
        for _t in 0..20 {
            let params_t = params.clone();
            let cands = &candidates;
            let out_t = out.clone();
            let idx_t = next_idx.clone();
            s.spawn(move || {
                let mut net = NetworkState::new(params_t);
                let mut tt = TranspositionTable::new(1);
                let mut hist = HistoryTables::default();
                loop {
                    let idx = idx_t.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }
                    let (ref board, v_teacher_root) = cands[idx];
                    let root_moves = board.generate_all_moves();
                    let mut move_oracles = Vec::with_capacity(root_moves.len());
                    let mut best_2ply_q = i32::MIN;

                    for &mv in root_moves.iter() {
                        let mut child = board.clone();
                        child.make_move(mv);
                        let opp_moves = child.generate_all_moves();
                        if opp_moves.is_empty() {
                            let q = if child.in_check() { 29_000 } else { 0 };
                            best_2ply_q = best_2ply_q.max(q);
                            move_oracles.push(MoveOracle {
                                mv,
                                obj_q_cp: q,
                                min_style_q_cp: q,
                                draw_permille: if q == 0 { 1000 } else { 0 },
                                def_difficulty_cp: 0,
                            });
                            continue;
                        }

                        let (child_ev, child_psqt, child_draw) = net.evaluate_with_components(&child, 0);
                        let q_static = -child_ev;
                        let q_psqt_style = -child_psqt;

                        let mut opp_qs = Vec::with_capacity(opp_moves.len());
                        let mut shallow_best_idx = 0usize;
                        let mut shallow_best_impulse = i32::MIN;

                        for (oi, &omv) in opp_moves.iter().enumerate() {
                            let cap_gain = if omv.is_capture() {
                                asura::evaluation::evaluate(&child)
                                    + (child.type_on(omv.capture_sq()) as i32 + 1) * 120
                            } else {
                                0
                            };
                            let mut grand = child.clone();
                            grand.make_move(omv);
                            let (g_ev, _, _) = net.evaluate_with_components(&grand, 0);
                            let opp_q = -g_ev;
                            opp_qs.push(opp_q);

                            let impulse = cap_gain + opp_q / 4;
                            if impulse > shallow_best_impulse {
                                shallow_best_impulse = impulse;
                                shallow_best_idx = oi;
                            }
                        }

                        let shallow_top_q = opp_qs[shallow_best_idx];
                        opp_qs.sort_unstable_by(|a, b| b.cmp(a));
                        let opp_best_q = opp_qs[0];
                        let q_minimax = -opp_best_q;
                        let obj_q = (65 * q_minimax + 35 * q_static) / 100;
                        best_2ply_q = best_2ply_q.max(obj_q);

                        let n_viable = opp_qs.iter().filter(|&&oq| oq >= opp_best_q - 35).count();
                        let child_pcs = child.occupancies().popcount() as i32;
                        let draw_permille = (480 - obj_q.abs() * 2 - child_pcs * 6 + child_draw * 2).clamp(25, 920);

                        let def_diff = if (1..=3).contains(&n_viable) && draw_permille < 750 && !mv.is_capture() {
                            let discovery_gap = (opp_best_q - shallow_top_q).max(0);
                            let scarcity = (4 - n_viable as i32).max(1);
                            let bonus = if n_viable == 1 && discovery_gap > 20 { 45 } else { 0 };
                            (scarcity * (discovery_gap + bonus) * (1000 - draw_permille)) / 3000
                        } else {
                            0
                        };

                        let min_style_q = obj_q.min(q_static).min((obj_q + q_psqt_style) / 2);
                        move_oracles.push(MoveOracle {
                            mv,
                            obj_q_cp: obj_q,
                            min_style_q_cp: min_style_q,
                            draw_permille,
                            def_difficulty_cp: def_diff.clamp(0, 400),
                        });
                    }

                    // Theorem 17 (Strict Oracle Dominance `Nodes_Oracle > max(B_r)`):
                    // Refine top candidate moves with a deep 6,000-node alpha-beta search so shallow 2-ply mirages are refuted.
                    let mut order: Vec<usize> = (0..move_oracles.len()).collect();
                    order.sort_unstable_by(|&a, &b| move_oracles[b].obj_q_cp.cmp(&move_oracles[a].obj_q_cp));
                    let mut best_obj_q = i32::MIN;
                    for (rank, &m_idx) in order.iter().enumerate() {
                        let mo = &mut move_oracles[m_idx];
                        if mo.obj_q_cp.abs() < 25_000 && (rank < 8 || mo.obj_q_cp >= best_2ply_q - 160) {
                            let mut child = board.clone();
                            child.make_move(mo.mv);
                            tt.clear(1);
                            hist.clear();
                            let deep_res = search_with_limits(&child, Limits::Nodes(6_000), &tt, &mut hist);
                            let deep_q = (-deep_res.score).clamp(-4_000, 4_000);
                            mo.obj_q_cp = (75 * deep_q + 25 * mo.obj_q_cp) / 100;
                            mo.min_style_q_cp = mo.min_style_q_cp.min(mo.obj_q_cp);
                        }
                        best_obj_q = best_obj_q.max(mo.obj_q_cp);
                    }

                    out_t.lock().unwrap()[idx] = Some(PositionOracle {
                        board: board.clone(),
                        v_root_cp: v_teacher_root,
                        best_obj_q_cp: best_obj_q,
                        moves: move_oracles,
                    });
                }
            });
        }
    });

    let mut guard = out.lock().unwrap();
    guard.drain(..).flatten().collect()
}

fn evaluate_pyramid(
    suite: &[PositionOracle],
    budgets: &[u64; 3],
) -> ([[f64; 6]; 3], Vec<[f64; 6]>) {
    let n_pos = suite.len();
    let per_pos_agg: Arc<Mutex<Vec<[f64; 6]>>> = Arc::new(Mutex::new(vec![[0.0; 6]; n_pos]));
    let per_rung_sums: Arc<Mutex<[[f64; 6]; 3]>> = Arc::new(Mutex::new([[0.0; 6]; 3]));
    let next_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    std::thread::scope(|s| {
        for _t in 0..20 {
            let agg_t = per_pos_agg.clone();
            let rung_t = per_rung_sums.clone();
            let idx_t = next_idx.clone();
            s.spawn(move || {
                let mut tt = TranspositionTable::new(1);
                let mut hist = HistoryTables::default();
                let mut local_rung = [[0.0f64; 6]; 3];

                loop {
                    let idx = idx_t.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= n_pos {
                        break;
                    }
                    let pos = &suite[idx];
                    let mut pos_metrics = [0.0f64; 6];

                    for (r_idx, &b_nodes) in budgets.iter().enumerate() {
                        tt.clear(1);
                        hist.clear();
                        let res = search_with_limits(&pos.board, Limits::Nodes(b_nodes), &tt, &mut hist);
                        let chosen = res.best_move.unwrap_or(pos.moves[0].mv);
                        let mo = pos.moves.iter().find(|m| m.mv == chosen).unwrap_or(&pos.moves[0]);

                        let regret = (pos.best_obj_q_cp - mo.obj_q_cp).max(0) as f64;
                        let draw_p = mo.draw_permille as f64 / 1000.0;
                        let p_nonloss = 1.0 / (1.0 + (-f64::from(mo.obj_q_cp) / 360.0).exp());
                        let p_obj_win = (p_nonloss - 0.5 * draw_p).clamp(0.0, 1.0);

                        let def_diff = mo.def_difficulty_cp as f64;
                        let p_slip = (def_diff / (def_diff + 180.0)).clamp(0.0, 0.55);
                        let p_prac_win = p_obj_win + (1.0 - p_obj_win) * 0.45 * p_slip;
                        let p_prac_draw =
                            draw_p * (1.0 - 0.45 * p_slip) + (1.0 - p_obj_win - draw_p).max(0.0) * 0.55 * p_slip;

                        let excess_regret = (regret - 35.0).max(0.0);
                        let loose_soundness = -(excess_regret * excess_regret) / 25.0;

                        // Universal Win-First Law (Theorem 14):
                        let u_asura = if pos.v_root_cp >= 80 {
                            (4.0 * p_prac_win - 1.0 * p_prac_draw) * 1000.0 + 0.25 * loose_soundness
                        } else {
                            (3.0 * p_prac_win + 1.0 * p_prac_draw) * 1000.0 + 0.25 * loose_soundness
                        };

                        let m_vec = [
                            p_prac_win * 1000.0,
                            (p_prac_win + p_prac_draw) * 1000.0,
                            def_diff,
                            mo.min_style_q_cp as f64,
                            loose_soundness,
                            u_asura,
                        ];
                        for k in 0..6 {
                            local_rung[r_idx][k] += m_vec[k] / n_pos as f64;
                            pos_metrics[k] += m_vec[k] / 3.0;
                        }
                    }
                    agg_t.lock().unwrap()[idx] = pos_metrics;
                }
                let mut rg = rung_t.lock().unwrap();
                for r in 0..3 {
                    for k in 0..6 {
                        rg[r][k] += local_rung[r][k];
                    }
                }
            });
        }
    });

    let rung_means = *per_rung_sums.lock().unwrap();
    let pos_vecs = per_pos_agg.lock().unwrap().clone();
    (rung_means, pos_vecs)
}

fn compute_paired_lcb95(cand: &[[f64; 6]], base: &[[f64; 6]]) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let n = cand.len() as f64;
    let mut means = [0.0; 6];
    let mut ses   = [0.0; 6];
    let mut lcbs  = [0.0; 6];
    for k in 0..6 {
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for i in 0..cand.len() {
            let d = cand[i][k] - base[i][k];
            sum += d;
            sum2 += d * d;
        }
        let mean = sum / n;
        let var = (sum2 / n - mean * mean).max(1e-9);
        let se = (var / n).sqrt();
        means[k] = mean;
        ses[k] = se;
        lcbs[k] = mean - 1.96 * se;
    }
    (means, ses, lcbs)
}

fn main() {
    let t0 = Instant::now();
    let nnue_path = Path::new("/home/tom/dev/personal/asura/asura_atn1.nnue");
    load_global_model(nnue_path).expect("Load ATN-1 model");
    set_spsa_active(true);
    let params = asura::nnue::global_model().unwrap();

    println!("=========================================================================================");
    println!("  ASURA 2-PASS 5-BLOCK SIDON-HADAMARD 3-BUDGET PYRAMID HYPER-SPSA (40 PARAMETERS)");
    println!("=========================================================================================");

    let suite = build_oracle_suite(params);
    let budgets = [500u64, 1_500, 4_500];

    let blocks: [(&str, [ParamSpec; 8]); 6] = [
        (
            "Block 1: ATN-1 Compensation, Shield & Tactical Ordering (Box 5 Only)",
            [
                ParamSpec { name: "HINDSIGHT_SAC_COMP_MIN",       step: 10,  min: 50,   max: 180 },
                ParamSpec { name: "HINDSIGHT_EXT_MIN_REDUCTION",  step: 200, min: 1000, max: 3500 },
                ParamSpec { name: "SEE_KING_RING_SAC_SHIELD",     step: 20,  min: 60,   max: 350 },
                ParamSpec { name: "QUIET_GREEDOID_UNBLOCK_BONUS", step: 90,  min: 100,  max: 2000 },
                ParamSpec { name: "QUIET_PHYSARUM_ARTERY_BONUS",  step: 90,  min: 100,  max: 2000 },
                ParamSpec { name: "QUIET_OUTPOST_HOLE_BONUS",     step: 90,  min: 100,  max: 2200 },
                ParamSpec { name: "QUIET_PETRI_OVERLOAD_BONUS",   step: 90,  min: 200,  max: 2200 },
                ParamSpec { name: "LMR_CAUSAL_RED_PENALTY",       step: 35,  min: 50,   max: 600 },
            ],
        ),
        (
            "Block 2: Coupled LMR & Singular Extensions Core",
            [
                ParamSpec { name: "LMR_TABLE_SCALE",              step: 14,  min: 300,  max: 560 },
                ParamSpec { name: "SE_BETA_MARGIN",               step: 4,   min: 10,   max: 55 },
                ParamSpec { name: "LMR_CUT_BIAS",                 step: 90,  min: 1200, max: 3000 },
                ParamSpec { name: "LMR_TT_PV_BONUS",              step: 80,  min: 500,  max: 1800 },
                ParamSpec { name: "LMR_NON_IMPROVING_PENALTY",    step: 80,  min: 500,  max: 1800 },
                ParamSpec { name: "LMR_PRIORITY_QUIET_BONUS",     step: 80,  min: 400,  max: 1600 },
                ParamSpec { name: "LMR_CHECK_BONUS",              step: 70,  min: 300,  max: 1400 },
                ParamSpec { name: "LMR_THREAT_BONUS",             step: 70,  min: 300,  max: 1400 },
            ],
        ),
        (
            "Block 3: Forward Pruning (RFP, NMP, Razoring & Hindsight)",
            [
                ParamSpec { name: "RFP_DEPTH_MARGIN",             step: 5,   min: 40,   max: 105 },
                ParamSpec { name: "RFP_IMPROVING_BONUS",          step: 5,   min: 20,   max: 85 },
                ParamSpec { name: "RFP_NO_THREATS_BONUS",         step: 5,   min: 10,   max: 75 },
                ParamSpec { name: "HINDSIGHT_REDUCTION_MARGIN",   step: 8,   min: 45,   max: 160 },
                ParamSpec { name: "NMP_DEEP_FLOOR",               step: 12,  min: -200, max: -20 },
                ParamSpec { name: "NMP_TT_SEE_MARGIN",            step: 20,  min: 80,   max: 350 },
                ParamSpec { name: "RAZOR_BASE",                   step: 18,  min: 100,  max: 340 },
                ParamSpec { name: "RAZOR_MARGIN",                 step: 18,  min: 120,  max: 350 },
            ],
        ),
        (
            "Block 4: ProbCut, Futility, SEE & History Pruning",
            [
                ParamSpec { name: "PROBCUT_BETA_MARGIN",          step: 10,  min: 80,   max: 210 },
                ParamSpec { name: "PROBCUT_IMPROVING_BONUS",      step: 6,   min: 15,   max: 85 },
                ParamSpec { name: "FP_BASE_MARGIN",               step: 10,  min: 60,   max: 200 },
                ParamSpec { name: "FP_DEPTH_MARGIN",              step: 8,   min: 55,   max: 165 },
                ParamSpec { name: "SEE_QUIET_QUAD_COEFF",         step: 3,   min: -40,  max: -6 },
                ParamSpec { name: "SEE_NOISY_DEPTH_COEFF",        step: 8,   min: -160, max: -40 },
                ParamSpec { name: "HP_IMPROVING_OFFSET",          step: 20,  min: -350, max: -80 },
                ParamSpec { name: "SE_DOUBLE_GAP",                step: 4,   min: 10,   max: 55 },
            ],
        ),
        (
            "Block 5: Move Ordering & Continuation History Weights",
            [
                ParamSpec { name: "QUIET_HIST_WEIGHT",            step: 100, min: 800,  max: 2400 },
                ParamSpec { name: "PAWN_HIST_WEIGHT",             step: 90,  min: 500,  max: 1800 },
                ParamSpec { name: "CONT_HIST_1_WEIGHT",           step: 100, min: 800,  max: 2400 },
                ParamSpec { name: "CONT_HIST_2_WEIGHT",           step: 90,  min: 500,  max: 1800 },
                ParamSpec { name: "QUIET_THREATENED_DEST_PENALTY",step: 500, min: 4000, max: 12000 },
                ParamSpec { name: "QUIET_CREATE_THREAT_BONUS",    step: 300, min: 1500, max: 6500 },
                ParamSpec { name: "QUIET_WALL_PAWN_PENALTY",      step: 350, min: 1500, max: 7000 },
                ParamSpec { name: "QUIET_PETRI_OVERLOAD_BONUS",   step: 90,  min: 200,  max: 2000 },
            ],
        ),
        (
            "Block 6: Quiescence Pruning & Late-Quiet Closure",
            [
                ParamSpec { name: "QS_FUTILITY_MARGIN",           step: 15,  min: 100,  max: 320 },
                ParamSpec { name: "QS_SEE_FUTILITY_MARGIN",       step: 20,  min: 150,  max: 450 },
                ParamSpec { name: "QS_EARLY_CUTOFF_MARGIN",       step: 18,  min: 120,  max: 350 },
                ParamSpec { name: "QS_SEE_BASE",                  step: 8,   min: 30,   max: 130 },
                ParamSpec { name: "QS_QUIET_HIST_THRESHOLD",      step: 40,  min: -600, max: -100 },
                ParamSpec { name: "LATE_QUIET_BASE_MARGIN",       step: 6,   min: 20,   max: 90 },
                ParamSpec { name: "LATE_QUIET_DEPTH_MARGIN",      step: 6,   min: 40,   max: 120 },
                ParamSpec { name: "DEFER_QUIET_THRESHOLD",        step: 150, min: -2500,max: -300 },
            ],
        ),
    ];

    const SIDON_COLS: [usize; 8] = [1, 2, 4, 8, 5, 10, 6, 11];
    let h16 = hadamard_16();

    let (mut cur_rungs, mut cur_pos) = evaluate_pyramid(&suite, &budgets);
    println!(
        "  Initial 3-Pyramid Utility : STC={:.2}, MTC={:.2}, LTC={:.2} (Mean={:.2})\n",
        cur_rungs[0][5],
        cur_rungs[1][5],
        cur_rungs[2][5],
        (cur_rungs[0][5] + cur_rungs[1][5] + cur_rungs[2][5]) / 3.0
    );

    for pass in 1..=2 {
        println!("--- PASS {}/2 ACROSS ALL 5 CAUSAL BLOCKS ---", pass);
        let step_mult = if pass == 1 { 1.0 } else { 0.5 };

        for (b_name, specs) in &blocks {
            let t_b = Instant::now();
            let base_vals: [i64; 8] = std::array::from_fn(|i| tunables::get_by_name(specs[i].name).unwrap());
            // Theorem 20: Deep-Weighted Pyramid Objective (0.15 STC + 0.30 MTC + 0.55 LTC)
            let u0 = 0.15 * cur_rungs[0][5] + 0.30 * cur_rungs[1][5] + 0.55 * cur_rungs[2][5];

            let mut grad_u = [0.0f64; 8];
            let mut grad_sq = [0.0f64; 8];
            let mut hess_pairs = [0.0f64; 4];

            for m in 0..16 {
                for i in 0..8 {
                    let sign = h16[m][SIDON_COLS[i]] as i64;
                    let delta = ((specs[i].step as f64 * step_mult).round() as i64).max(1);
                    let vp = (base_vals[i] + sign * delta).clamp(specs[i].min, specs[i].max);
                    tunables::set_by_name(specs[i].name, vp);
                }
                let (r_p, _) = evaluate_pyramid(&suite, &budgets);

                for i in 0..8 {
                    let sign = h16[m][SIDON_COLS[i]] as i64;
                    let delta = ((specs[i].step as f64 * step_mult).round() as i64).max(1);
                    let vm = (base_vals[i] - sign * delta).clamp(specs[i].min, specs[i].max);
                    tunables::set_by_name(specs[i].name, vm);
                }
                let (r_m, _) = evaluate_pyramid(&suite, &budgets);

                let up = 0.15 * r_p[0][5] + 0.30 * r_p[1][5] + 0.55 * r_p[2][5];
                let um = 0.15 * r_m[0][5] + 0.30 * r_m[1][5] + 0.55 * r_m[2][5];
                let diff = up - um;
                let sum = up + um - 2.0 * u0;

                for i in 0..8 {
                    let gi = (diff * h16[m][SIDON_COLS[i]]) / 2.0;
                    grad_u[i] += gi / 16.0;
                    grad_sq[i] += (gi * gi) / 16.0;
                }
                for p in 0..4 {
                    let s_ab = h16[m][SIDON_COLS[2 * p]] * h16[m][SIDON_COLS[2 * p + 1]];
                    hess_pairs[p] += (sum * s_ab) / 32.0;
                }
            }

            // Restore base values before testing line-search candidates
            for i in 0..8 {
                tunables::set_by_name(specs[i].name, base_vals[i]);
            }

            // Theorem 19: SNR-Filtered Proportional Steps (zero out passenger coordinates with |g_i| < 1.5 * SE_i)
            let mut adj_g = [0.0f64; 8];
            let mut max_g = 1e-9f64;
            for p in 0..4 {
                let i = 2 * p;
                let j = 2 * p + 1;
                let se_i = ((grad_sq[i] - grad_u[i] * grad_u[i]).max(1e-9) / 16.0).sqrt();
                let se_j = ((grad_sq[j] - grad_u[j] * grad_u[j]).max(1e-9) / 16.0).sqrt();
                let raw_i = grad_u[i] - 0.15 * hess_pairs[p].clamp(-2.0, 2.0) * grad_u[j];
                let raw_j = grad_u[j] - 0.15 * hess_pairs[p].clamp(-2.0, 2.0) * grad_u[i];
                adj_g[i] = if raw_i.abs() * 2.0 >= 3.0 * se_i { raw_i } else { 0.0 };
                adj_g[j] = if raw_j.abs() * 2.0 >= 3.0 * se_j { raw_j } else { 0.0 };
                max_g = max_g.max(adj_g[i].abs()).max(adj_g[j].abs());
            }

            let mut accepted = false;
            for scale in [1.0, 0.5] {
                let mut cand_vals = base_vals;
                for i in 0..8 {
                    let si = ((adj_g[i] / max_g) * scale * step_mult * (specs[i].step as f64)).round() as i64;
                    cand_vals[i] = (base_vals[i] + si).clamp(specs[i].min, specs[i].max);
                    tunables::set_by_name(specs[i].name, cand_vals[i]);
                }
                if cand_vals == base_vals {
                    break;
                }

                let (cand_rungs, cand_pos) = evaluate_pyramid(&suite, &budgets);
                let (means, ses, lcbs) = compute_paired_lcb95(&cand_pos, &cur_pos);
                let stc_g = cand_rungs[0][5] - cur_rungs[0][5];
                let mtc_g = cand_rungs[1][5] - cur_rungs[1][5];
                let ltc_g = cand_rungs[2][5] - cur_rungs[2][5];

                // Theorem 20: Asymptotic LTC-Scaling Gate (`15*d1 + 30*d2 + 55*d3 > 0 && lcb > 0 && d3 >= d2 && 2*d3 >= d1`)
                let pyr_g = 0.15 * stc_g + 0.30 * mtc_g + 0.55 * ltc_g;
                let asymptotic_ltc_ok = pyr_g > 0.0 && lcbs[5] > 0.0 && ltc_g >= mtc_g && 2.0 * ltc_g >= stc_g;
                let no_regression = lcbs[0] >= -5.0 && lcbs[1] >= -5.0 && lcbs[3] >= -3.0 && lcbs[4] >= -3.0;
                if no_regression && asymptotic_ltc_ok {
                    cur_rungs = cand_rungs;
                    cur_pos = cand_pos;
                    accepted = true;
                    println!(
                        "  [{}] ({:.1}s) ACCEPTED: dUtil={:+.2} ± {:.2} (LCB={:+.2}) | dSTC={:+.2}, dMTC={:+.2}, dLTC={:+.2} | {:?}",
                        b_name,
                        t_b.elapsed().as_secs_f64(),
                        means[5],
                        1.96 * ses[5],
                        lcbs[5],
                        stc_g,
                        mtc_g,
                        ltc_g,
                        cand_vals
                    );
                    break;
                }
            }
            if !accepted {
                for i in 0..8 {
                    tunables::set_by_name(specs[i].name, base_vals[i]);
                }
                println!(
                    "  [{}] ({:.1}s) HELD BASELINE (no candidate cleared LCB_95% + LTC gate): {:?}",
                    b_name,
                    t_b.elapsed().as_secs_f64(),
                    base_vals
                );
            }
        }
    }

    println!("\n=== FINAL 2-PASS VALIDATED PARAMETERS FOR src/parameters.rs ===");
    for (_, specs) in &blocks {
        for sp in specs {
            println!("    {:32}: {};", sp.name, tunables::get_by_name(sp.name).unwrap());
        }
    }
    println!(
        "  Final 3-Pyramid Utility : STC={:.2}, MTC={:.2}, LTC={:.2} (Mean={:.2}) in {:.2?}",
        cur_rungs[0][5],
        cur_rungs[1][5],
        cur_rungs[2][5],
        (cur_rungs[0][5] + cur_rungs[1][5] + cur_rungs[2][5]) / 3.0,
        t0.elapsed()
    );
}
