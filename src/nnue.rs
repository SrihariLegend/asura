pub mod accumulator;

pub mod forward {
    pub mod scalar;
    #[cfg(target_feature = "avx2")]
    pub mod vectorized;

    pub use scalar::forward_scalar;
    #[cfg(target_feature = "avx2")]
    pub use vectorized::forward_avx2;
}

pub mod simd {
    #[cfg(target_feature = "avx2")]
    pub mod avx2;
    #[cfg(target_feature = "avx2")]
    pub use avx2::*;
}

pub use accumulator::{
    AccumulatorSide, BoardSnapshot, FinnyEntry, PkCacheEntry, collect_active_rows, collect_active_rows_snap,
    collect_pawn_rows, collect_psq_rows, collect_threat_rows, collect_threat_rows_both, diff_pawn_rows_both,
    diff_pawn_rows_one, diff_threat_rows_both, diff_threat_rows_one, evaluate_pawn_king_subnet, heavy_regime,
    orient_flip, passed_pawn_row_index, pawn_pair_row_index, psq_row_index, threat_row_index,
};
pub use forward::forward_scalar;
#[cfg(target_feature = "avx2")]
pub use forward::forward_avx2;

use std::sync::{
    Arc, OnceLock, RwLock,
    atomic::{AtomicBool, Ordering},
};

use crate::{
    board::Board,
    param,
    types::{Color, PieceType},
};

pub const L1_SIZE: usize = 768;
pub const L1_HALF: usize = L1_SIZE / 2;
pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 32;
pub const FC3_INPUTS: usize = L2_SIZE * 2 + L3_SIZE * 2;

pub const KING_BUCKETS: usize = 12;
pub const HEAVY_REGIMES: usize = 3;
pub const PSQT_BUCKETS: usize = 8;
pub const HYBRID_STACKS: usize = 16;

pub const PSQ_ROWS: usize = KING_BUCKETS * HEAVY_REGIMES * 768; // 27,648
pub const THREAT_ROWS: usize = 55_776;
pub const PAWN_PAIR_ROWS: usize = 4_560;
pub const PASSED_PAWN_ROWS: usize = 96;
pub const TOTAL_FT_ROWS: usize = PSQ_ROWS + THREAT_ROWS + PAWN_PAIR_ROWS + PASSED_PAWN_ROWS; // 88,080

pub const PK_INPUTS: usize = 368;
pub const PK_HIDDEN: usize = 128;
pub const PK_CACHE_SIZE: usize = 4096;

#[rustfmt::skip]
pub const KING_BUCKET_LAYOUT: [u8; 64] = [
    0,  1,  2,  3,  3,  2,  1,  0,
    4,  5,  6,  7,  7,  6,  5,  4,
    8,  8,  9,  9,  9,  9,  8,  8,
   10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10,
   11, 11, 11, 11, 11, 11, 11, 11,
   11, 11, 11, 11, 11, 11, 11, 11,
   11, 11, 11, 11, 11, 11, 11, 11,
];

#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct Aligned<T> {
    pub data: T,
}

impl<T> Aligned<T> {
    pub const fn new(data: T) -> Self {
        Self { data }
    }
}

impl<T> std::ops::Deref for Aligned<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for Aligned<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct FeatureRow {
    pub weights: [i8; L1_SIZE],
    pub psqt: [i32; PSQT_BUCKETS],
    pub _pad: [u8; 32],
}

impl Default for FeatureRow {
    fn default() -> Self {
        Self { weights: [0; L1_SIZE], psqt: [0; PSQT_BUCKETS], _pad: [0; 32] }
    }
}

#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct LayerStack {
    pub fc1_biases: [i32; L2_SIZE],
    pub fc1_weights: [i8; L1_SIZE * L2_SIZE],
    pub quad_skip_weights: [i8; L1_SIZE],
    pub fc2_biases: [i32; L3_SIZE],
    pub fc2_weights: [i8; (L2_SIZE * 2) * L3_SIZE],
    pub fc3_biases: [i32; 2],
    pub fc3_weights: [[i8; FC3_INPUTS]; 2],
}

impl Default for LayerStack {
    fn default() -> Self {
        Self {
            fc1_biases: [0; L2_SIZE],
            fc1_weights: [0; L1_SIZE * L2_SIZE],
            quad_skip_weights: [0; L1_SIZE],
            fc2_biases: [0; L3_SIZE],
            fc2_weights: [0; (L2_SIZE * 2) * L3_SIZE],
            fc3_biases: [0; 2],
            fc3_weights: [[0; FC3_INPUTS]; 2],
        }
    }
}

#[repr(C, align(32))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedBoard {
    pub occ: u64,
    pub pieces_lo: u64,
    pub pieces_hi: u64,
    pub score_cp: i16,
    pub psqt_cp: i16,
    pub stm_wdl: u8,
    pub ksq_w: u8,
    pub ksq_b: u8,
    pub ply_clamped: u8,
}

impl PackedBoard {
    pub fn from_le_bytes(b: &[u8; 32]) -> Self {
        Self {
            occ: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            pieces_lo: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            pieces_hi: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            score_cp: i16::from_le_bytes(b[24..26].try_into().unwrap()),
            psqt_cp: i16::from_le_bytes(b[26..28].try_into().unwrap()),
            stm_wdl: b[28],
            ksq_w: b[29],
            ksq_b: b[30],
            ply_clamped: b[31],
        }
    }

    pub fn to_board(&self) -> Board {
        const CHARS: [char; 12] = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
        let mut grid = [None; 64];
        let mut temp = self.occ;
        let mut k = 0usize;
        while temp != 0 {
            let sq = temp.trailing_zeros() as usize;
            temp &= temp - 1;
            let code = if k < 16 { (self.pieces_lo >> (k * 4)) & 0xF } else { (self.pieces_hi >> ((k - 16) * 4)) & 0xF }
                as usize;
            grid[sq] = Some(CHARS[code.min(11)]);
            k += 1;
        }
        let mut fen = String::with_capacity(72);
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                if let Some(ch) = grid[r * 8 + f] {
                    if empty > 0 {
                        fen.push(char::from_digit(empty, 10).unwrap());
                        empty = 0;
                    }
                    fen.push(ch);
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                fen.push(char::from_digit(empty, 10).unwrap());
            }
            if r > 0 {
                fen.push('/');
            }
        }
        if (self.stm_wdl & 1) == 0 {
            fen.push_str(" w - - 0 1");
        } else {
            fen.push_str(" b - - 0 1");
        }
        Board::from_fen(&fen).expect("Valid PackedBoard FEN")
    }
}

#[repr(C, align(64))]
pub struct ModelParameters {
    pub ft_biases: Aligned<[i16; L1_SIZE]>,
    pub ft_weights: Box<[Aligned<[i8; L1_SIZE]>]>,
    pub ft_psqt: Box<[[i32; PSQT_BUCKETS]]>,
    pub pk_biases_1: Aligned<[i16; PK_HIDDEN]>,
    pub pk_weights_1: Box<[[i16; PK_HIDDEN]; PK_INPUTS]>,
    pub pk_beta_biases: [i32; L2_SIZE],
    pub pk_gamma_biases: [i32; L2_SIZE],
    pub pk_beta_weights: [[i8; PK_HIDDEN]; L2_SIZE],
    pub pk_gamma_weights: [[i8; PK_HIDDEN]; L2_SIZE],
    pub stacks: [LayerStack; HYBRID_STACKS],
    pub phase_table: [i32; 16],
}

impl ModelParameters {
    pub fn new_zeroed() -> Self {
        Self {
            ft_biases: Aligned::new([0; L1_SIZE]),
            ft_weights: vec![Aligned::new([0i8; L1_SIZE]); TOTAL_FT_ROWS].into_boxed_slice(),
            ft_psqt: vec![[0i32; PSQT_BUCKETS]; TOTAL_FT_ROWS].into_boxed_slice(),
            pk_biases_1: Aligned::new([0; PK_HIDDEN]),
            pk_weights_1: vec![[0i16; PK_HIDDEN]; PK_INPUTS].into_boxed_slice().try_into().unwrap(),
            pk_beta_biases: [0; L2_SIZE],
            pk_gamma_biases: [0; L2_SIZE],
            pk_beta_weights: [[0; PK_HIDDEN]; L2_SIZE],
            pk_gamma_weights: [[0; PK_HIDDEN]; L2_SIZE],
            stacks: [LayerStack::default(); HYBRID_STACKS],
            phase_table: [1024; 16],
        }
    }

    pub fn new_deterministic_random(seed: u64) -> Self {
        let mut p = Self::new_zeroed();
        let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut next = || -> u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng
        };

        for b in p.ft_biases.iter_mut() {
            *b = ((next() >> 32) as i16 % 160) - 60;
        }
        for r in 0..TOTAL_FT_ROWS {
            for w in p.ft_weights[r].iter_mut() {
                *w = ((next() >> 40) as i8) % 24;
            }
            for psq in p.ft_psqt[r].iter_mut() {
                *psq = ((next() >> 32) as i32 % 400) - 200;
            }
        }
        for b in p.pk_biases_1.iter_mut() {
            *b = ((next() >> 32) as i16 % 40) + 10;
        }
        for row in p.pk_weights_1.iter_mut() {
            for w in row.iter_mut() {
                *w = ((next() >> 32) as i16 % 32) - 12;
            }
        }
        for j in 0..L2_SIZE {
            p.pk_beta_biases[j] = ((next() >> 32) as i32 % 200) - 100;
            p.pk_gamma_biases[j] = ((next() >> 32) as i32 % 32) - 16;
            for k in 0..PK_HIDDEN {
                p.pk_beta_weights[j][k] = ((next() >> 40) as i8) % 16;
                p.pk_gamma_weights[j][k] = ((next() >> 40) as i8) % 8;
            }
        }
        for st in p.stacks.iter_mut() {
            for b in st.fc1_biases.iter_mut() {
                *b = ((next() >> 32) as i32 % 500) - 250;
            }
            for w in st.fc1_weights.iter_mut() {
                *w = ((next() >> 40) as i8) % 32;
            }
            for w in st.quad_skip_weights.iter_mut() {
                *w = ((next() >> 40) as i8) % 24;
            }
            for b in st.fc2_biases.iter_mut() {
                *b = ((next() >> 32) as i32 % 500) - 250;
            }
            for w in st.fc2_weights.iter_mut() {
                *w = ((next() >> 40) as i8) % 32;
            }
            st.fc3_biases[0] = ((next() >> 32) as i32 % 400) - 200;
            st.fc3_biases[1] = ((next() >> 32) as i32 % 200) - 100;
            for c in 0..2 {
                for w in st.fc3_weights[c].iter_mut() {
                    *w = ((next() >> 40) as i8) % 32;
                }
            }
        }
        for (idx, pt) in p.phase_table.iter_mut().enumerate() {
            *pt = 1024 + ((idx as i32 - 8) * 12);
        }
        p
    }

    pub const MAGIC: &'static [u8; 16] = b"ASURA_ATN1_V001\0";

    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(Self::MAGIC)?;
        for &b in self.ft_biases.iter() {
            w.write_all(&b.to_le_bytes())?;
        }
        let pad32 = [0u8; 32];
        for r in 0..TOTAL_FT_ROWS {
            let weights_u8: &[u8; L1_SIZE] = unsafe { std::mem::transmute(&self.ft_weights[r].data) };
            w.write_all(weights_u8)?;
            for &p in self.ft_psqt[r].iter() {
                w.write_all(&p.to_le_bytes())?;
            }
            w.write_all(&pad32)?;
        }
        for &b in self.pk_biases_1.iter() {
            w.write_all(&b.to_le_bytes())?;
        }
        for row in self.pk_weights_1.iter() {
            for &v in row.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        for &b in self.pk_beta_biases.iter() {
            w.write_all(&b.to_le_bytes())?;
        }
        for &b in self.pk_gamma_biases.iter() {
            w.write_all(&b.to_le_bytes())?;
        }
        for row in self.pk_beta_weights.iter() {
            let u: &[u8; PK_HIDDEN] = unsafe { std::mem::transmute(row) };
            w.write_all(u)?;
        }
        for row in self.pk_gamma_weights.iter() {
            let u: &[u8; PK_HIDDEN] = unsafe { std::mem::transmute(row) };
            w.write_all(u)?;
        }
        for st in self.stacks.iter() {
            for &b in st.fc1_biases.iter() {
                w.write_all(&b.to_le_bytes())?;
            }
            let fc1_u8: &[u8; L1_SIZE * L2_SIZE] = unsafe { std::mem::transmute(&st.fc1_weights) };
            w.write_all(fc1_u8)?;
            let qs_u8: &[u8; L1_SIZE] = unsafe { std::mem::transmute(&st.quad_skip_weights) };
            w.write_all(qs_u8)?;
            for &b in st.fc2_biases.iter() {
                w.write_all(&b.to_le_bytes())?;
            }
            let fc2_u8: &[u8; (L2_SIZE * 2) * L3_SIZE] = unsafe { std::mem::transmute(&st.fc2_weights) };
            w.write_all(fc2_u8)?;
            for &b in st.fc3_biases.iter() {
                w.write_all(&b.to_le_bytes())?;
            }
            for c in 0..2 {
                let fc3_u8: &[u8; FC3_INPUTS] = unsafe { std::mem::transmute(&st.fc3_weights[c]) };
                w.write_all(fc3_u8)?;
            }
        }
        for &pt in self.phase_table.iter() {
            w.write_all(&pt.to_le_bytes())?;
        }
        w.flush()
    }

    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        use std::io::{BufReader, Error, ErrorKind, Read};
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 16];
        r.read_exact(&mut magic)?;
        if &magic != Self::MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid ATN-1 magic header"));
        }

        let mut p = Self::new_zeroed();
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];

        for b in p.ft_biases.iter_mut() {
            r.read_exact(&mut buf2)?;
            *b = i16::from_le_bytes(buf2);
        }
        let mut pad32 = [0u8; 32];
        for idx in 0..TOTAL_FT_ROWS {
            let weights_u8: &mut [u8; L1_SIZE] = unsafe { std::mem::transmute(&mut p.ft_weights[idx].data) };
            r.read_exact(weights_u8)?;
            for psq in p.ft_psqt[idx].iter_mut() {
                r.read_exact(&mut buf4)?;
                *psq = i32::from_le_bytes(buf4);
            }
            r.read_exact(&mut pad32)?;
        }
        for b in p.pk_biases_1.iter_mut() {
            r.read_exact(&mut buf2)?;
            *b = i16::from_le_bytes(buf2);
        }
        for row in p.pk_weights_1.iter_mut() {
            for v in row.iter_mut() {
                r.read_exact(&mut buf2)?;
                *v = i16::from_le_bytes(buf2);
            }
        }
        for b in p.pk_beta_biases.iter_mut() {
            r.read_exact(&mut buf4)?;
            *b = i32::from_le_bytes(buf4);
        }
        for b in p.pk_gamma_biases.iter_mut() {
            r.read_exact(&mut buf4)?;
            *b = i32::from_le_bytes(buf4);
        }
        for row in p.pk_beta_weights.iter_mut() {
            let u: &mut [u8; PK_HIDDEN] = unsafe { std::mem::transmute(row) };
            r.read_exact(u)?;
        }
        for row in p.pk_gamma_weights.iter_mut() {
            let u: &mut [u8; PK_HIDDEN] = unsafe { std::mem::transmute(row) };
            r.read_exact(u)?;
        }
        for st in p.stacks.iter_mut() {
            for b in st.fc1_biases.iter_mut() {
                r.read_exact(&mut buf4)?;
                *b = i32::from_le_bytes(buf4);
            }
            let fc1_u8: &mut [u8; L1_SIZE * L2_SIZE] = unsafe { std::mem::transmute(&mut st.fc1_weights) };
            r.read_exact(fc1_u8)?;
            let qs_u8: &mut [u8; L1_SIZE] = unsafe { std::mem::transmute(&mut st.quad_skip_weights) };
            r.read_exact(qs_u8)?;
            for b in st.fc2_biases.iter_mut() {
                r.read_exact(&mut buf4)?;
                *b = i32::from_le_bytes(buf4);
            }
            let fc2_u8: &mut [u8; (L2_SIZE * 2) * L3_SIZE] = unsafe { std::mem::transmute(&mut st.fc2_weights) };
            r.read_exact(fc2_u8)?;
            for b in st.fc3_biases.iter_mut() {
                r.read_exact(&mut buf4)?;
                *b = i32::from_le_bytes(buf4);
            }
            for c in 0..2 {
                let fc3_u8: &mut [u8; FC3_INPUTS] = unsafe { std::mem::transmute(&mut st.fc3_weights[c]) };
                r.read_exact(fc3_u8)?;
            }
        }
        for pt in p.phase_table.iter_mut() {
            r.read_exact(&mut buf4)?;
            *pt = i32::from_le_bytes(buf4);
        }
        Ok(p)
    }
}

#[inline]
pub fn hybrid_stack_index(board: &Board) -> (usize, usize) {
    let pcs = board.occupancies().popcount().clamp(1, 32);
    let psqt_b = (pcs - 1) / 4;
    let mat_band = (pcs - 1) / 8;
    let stm = board.side_to_move();
    let us_q = !board.colored_pieces(stm, PieceType::Queen).is_empty() as usize;
    let th_q = !board.colored_pieces(!stm, PieceType::Queen).is_empty() as usize;
    let q_regime = (us_q << 1) | th_q;
    (psqt_b, mat_band * 4 + q_regime)
}

#[inline]
pub fn check_invasion_percolation(board: &Board, att_side: Color) -> bool {
    let def_side = !att_side;
    let wp = board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp = board.colored_pieces(Color::Black, PieceType::Pawn).0;
    let def_pawns = if def_side == Color::White { wp } else { bp };
    let def_pawn_att = if def_side == Color::White {
        ((def_pawns << 7) & 0x7f7f_7f7f_7f7f_7f7f) | ((def_pawns << 9) & 0xfefe_fefe_fefe_fefe)
    } else {
        ((def_pawns >> 9) & 0x7f7f_7f7f_7f7f_7f7f) | ((def_pawns >> 7) & 0xfefe_fefe_fefe_fefe)
    };
    let open_sites = !(wp | bp | def_pawn_att);
    let ksq = board.king_square(att_side) as usize;
    let back_rank = if att_side == Color::White { 0xFFu64 } else { 0xFF00_0000_0000_0000u64 };
    let target_camp = if att_side == Color::White { 0xFFFF_FF00_0000_0000u64 } else { 0x0000_0000_00FF_FFFFu64 };

    let mut cluster = (1u64 << ksq) | back_rank;
    for _ in 0..8 {
        let next = (accumulator::dilate_moore(cluster) & open_sites) | cluster;
        if next == cluster {
            break;
        }
        cluster = next;
    }
    (cluster & target_camp) != 0
}

pub fn scale_atn1_evaluation(
    board: &Board, psqt: i32, positional: i32, draw_logit: i32, optimism: i32, params: &ModelParameters,
) -> i32 {
    let sac_thresh: i32 = param!(EVAL_SACRIFICE_PSQT_THRESHOLD);
    let sac_div: i32 = param!(EVAL_SACRIFICE_BOOST_DIV).max(4);
    let fort_min: i32 = param!(EVAL_FORTRESS_DISAGREE_MIN).max(20);

    let v_total = psqt + positional;
    let mut v = if psqt <= sac_thresh && v_total > 0 {
        v_total + positional / sac_div
    } else if psqt >= -sac_thresh && v_total < 0 {
        v_total + positional / sac_div
    } else if psqt.abs() >= 200 && (psqt > 0) != (positional > 0) && positional.abs() >= fort_min {
        let damp = positional.abs().min(256);
        v_total - (v_total * damp) / 2048
    } else {
        v_total
    };

    let sharpness = (128 - draw_logit.clamp(-64, 96)) as i64;
    let comp_boost = (positional.abs().min(400) as i64) + 256;
    v += ((optimism as i64) * sharpness * comp_boost / (128 * 384)) as i32;

    let stm = board.side_to_move();
    let q_us = board.colored_pieces(stm, PieceType::Queen).popcount();
    let q_th = board.colored_pieces(!stm, PieceType::Queen).popcount();
    let r_us = board.colored_pieces(stm, PieceType::Rook).popcount();
    let r_th = board.colored_pieces(!stm, PieceType::Rook).popcount();
    let b_us = board.colored_pieces(stm, PieceType::Bishop).popcount();
    let b_th = board.colored_pieces(!stm, PieceType::Bishop).popcount();
    let n_us = board.colored_pieces(stm, PieceType::Knight).popcount();
    let n_th = board.colored_pieces(!stm, PieceType::Knight).popcount();

    let imbalance_idx = ((q_us > q_th) as usize)
        | (((r_us > r_th) as usize) << 1)
        | (((b_us > b_th) as usize) << 2)
        | (((n_us > n_th) as usize) << 3);
    v = ((v as i64) * (params.phase_table[imbalance_idx] as i64) / 1024) as i32;

    // Theorem 29: Only apply invasion percolation fortress damping on locked minor/pawn endgames
    // (`q_us + q_th + r_us + r_th == 0` with zero pawn levers), preventing the losing side from
    // trading Queens into a lost endgame just to collect an artificial `3/4` damping cliff.
    if (q_us + q_th + r_us + r_th) == 0 && v.abs() > 80 {
        let wp = board.colored_pieces(Color::White, PieceType::Pawn).0;
        let bp = board.colored_pieces(Color::Black, PieceType::Pawn).0;
        let w_levers = ((wp << 7) & 0x7f7f_7f7f_7f7f_7f7f) | ((wp << 9) & 0xfefe_fefe_fefe_fefe);
        if (w_levers & bp) == 0 {
            let winning_side = if v > 0 { stm } else { !stm };
            if !check_invasion_percolation(board, winning_side) {
                v = (v * 3) / 4;
            }
        }
    }

    let r50 = board.fiftymove_clock().min(100) as i32;
    if r50 > 20 {
        v = ((v as i64) * ((100 - r50) as i64) / 80) as i32;
    }

    v.clamp(-30_000, 30_000)
}

const ACC_HISTORY_LEN: usize = 208;

#[derive(Copy, Clone)]
pub struct AccHistoryEntry {
    pub snap: BoardSnapshot,
    pub acc: [AccumulatorSide; 2],
    pub b_id_w: u8,
    pub b_id_b: u8,
    pub valid: bool,
}

#[derive(Clone)]
pub struct NetworkState {
    pub params: Arc<ModelParameters>,
    pub acc: [AccumulatorSide; 2],
    pub pk_cache: Box<[PkCacheEntry; PK_CACHE_SIZE]>,
    pub cached_board: Option<u64>,
    pub history: Vec<AccHistoryEntry>,
    pub pending_snaps: Box<[(u64, BoardSnapshot); 64]>,
    pub bucket_last_slot: [[u8; 72]; 2],
    pub history_cursor: usize,
    pub last_slot: usize,
    pub last_pk_idx: usize,
    pub scratch_adds: Vec<usize>,
    pub scratch_subs: Vec<usize>,
    pub scratch_prev: Vec<usize>,
    pub scratch_curr: Vec<usize>,
}

impl NetworkState {
    pub fn new(params: Arc<ModelParameters>) -> Self {
        let mut history = Vec::with_capacity(ACC_HISTORY_LEN);
        for _ in 0..ACC_HISTORY_LEN {
            history.push(AccHistoryEntry {
                snap: BoardSnapshot::default(),
                acc: [AccumulatorSide::new(&params), AccumulatorSide::new(&params)],
                b_id_w: u8::MAX,
                b_id_b: u8::MAX,
                valid: false,
            });
        }
        Self {
            acc: [AccumulatorSide::new(&params), AccumulatorSide::new(&params)],
            pk_cache: Box::new([PkCacheEntry::default(); PK_CACHE_SIZE]),
            params,
            cached_board: None,
            history,
            pending_snaps: Box::new([(u64::MAX, BoardSnapshot::default()); 64]),
            bucket_last_slot: [[u8::MAX; 72]; 2],
            history_cursor: 0,
            last_slot: 0,
            last_pk_idx: 0,
            scratch_adds: Vec::with_capacity(128),
            scratch_subs: Vec::with_capacity(128),
            scratch_prev: Vec::with_capacity(128),
            scratch_curr: Vec::with_capacity(128),
        }
    }

    #[inline]
    pub fn reset_history(&mut self) {
        for entry in &mut self.history {
            entry.valid = false;
        }
        self.pending_snaps.fill((u64::MAX, BoardSnapshot::default()));
        self.bucket_last_slot = [[u8::MAX; 72]; 2];
        self.cached_board = None;
        self.history_cursor = 0;
    }

    #[inline]
    pub fn clear_pending_snaps(&mut self) {
        self.pending_snaps.fill((u64::MAX, BoardSnapshot::default()));
    }

    #[inline(always)]
    pub fn on_make_move(&mut self, parent_board: &Board) {
        let ply = parent_board.halfmove_number();
        let slot = ply & 63;
        let h = parent_board.hash();
        if self.history[slot].valid && self.history[slot].snap.hash == h {
            self.pending_snaps[slot].0 = u64::MAX;
        } else if (self.pending_snaps[slot].0 & 0x7FFF_FFFF_FFFF_FFFF) != ply as u64
            || self.pending_snaps[slot].1.hash != h
        {
            self.pending_snaps[slot] = (ply as u64, BoardSnapshot::from_board(parent_board));
        }
    }

    pub fn evaluate_full_refresh_scalar(&mut self, board: &Board, optimism: i32) -> i32 {
        let mut rows = Vec::with_capacity(128);
        for pov in [Color::White, Color::Black] {
            let mut side_acc = AccumulatorSide::new(&self.params);
            collect_active_rows(board, pov, &mut rows);
            for &r in &rows {
                side_acc.add_row(&self.params.ft_weights[r].data, &self.params.ft_psqt[r]);
            }
            self.acc[pov as usize] = side_acc;
        }
        let stm = board.side_to_move() as usize;
        let (pk_beta, pk_gamma) = evaluate_pawn_king_subnet(board, &self.params, &mut self.pk_cache, &mut self.last_pk_idx);
        let (psqt_b, stack_b) = hybrid_stack_index(board);
        let (psqt, pos, draw_l) = forward_scalar(
            &self.acc[stm],
            &self.acc[stm ^ 1],
            pk_beta,
            pk_gamma,
            psqt_b,
            stack_b,
            &self.params,
        );
        scale_atn1_evaluation(board, psqt, pos, draw_l, optimism, &self.params)
    }

    pub fn evaluate_incremental(&mut self, board: &Board, optimism: i32) -> i32 {
        let ev = self.evaluate_with_components(board, optimism).0;
        self.acc = self.history[self.last_slot].acc;
        ev
    }

    fn update_accumulator(&mut self, board: &Board) -> usize {
        if self.cached_board.is_none() && self.history_cursor != 0 {
            for entry in &mut self.history {
                entry.valid = false;
            }
            self.pending_snaps.fill((u64::MAX, BoardSnapshot::default()));
            self.bucket_last_slot = [[u8::MAX; 72]; 2];
            self.history_cursor = 0;
        }

        let curr_ply = board.halfmove_number();
        let b_hash = board.hash();
        let b_colors = *board.colors_array();
        let b_pieces = *board.pieces_array();
        let ply_slot = curr_ply & 63;
        let parent_ply_slot = curr_ply.wrapping_sub(1) & 63;

        // Fast O(1) exact piece-placement check BEFORE taking BoardSnapshot or materializing ancestors!
        for &chk in &[ply_slot, parent_ply_slot] {
            let entry = &self.history[chk];
            if entry.valid && entry.snap.colors == b_colors && entry.snap.pieces == b_pieces {
                self.pending_snaps[ply_slot].0 = u64::MAX;
                self.cached_board = Some(b_hash);
                self.last_slot = chk;
                return chk;
            }
        }

        // Theorem 37 (Lazy Ancestor-Path Materialization): walk back along contiguous unmaterialized ancestors
        // so `parent_ply_slot` is guaranteed to be the exact 1-move parent!
        if self.cached_board.is_some() {
            let mut max_back = 0usize;
            while max_back < curr_ply && max_back < 16 {
                let anc_ply = curr_ply - (max_back + 1);
                if self.pending_snaps[anc_ply & 63].0 == anc_ply as u64 {
                    max_back += 1;
                } else {
                    break;
                }
            }
            for back in (1..=max_back).rev() {
                let anc_ply = curr_ply - back;
                let anc_slot = anc_ply & 63;
                let pend_snap = self.pending_snaps[anc_slot].1;
                self.pending_snaps[anc_slot].0 = u64::MAX;
                if !self.history[anc_slot].valid
                    || self.history[anc_slot].snap.colors != pend_snap.colors
                    || self.history[anc_slot].snap.pieces != pend_snap.pieces
                {
                    self.update_accumulator_snap(anc_ply, &pend_snap);
                }
            }
        }

        let snap_now = BoardSnapshot::from_board(board);
        self.update_accumulator_snap(curr_ply, &snap_now)
    }

    fn update_accumulator_snap(&mut self, curr_ply: usize, board: &BoardSnapshot) -> usize {
        let b_hash = board.hash;
        let b_occ = board.occupancies();
        let wp = board.colored_pieces(Color::White, PieceType::Pawn);
        let bp = board.colored_pieces(Color::Black, PieceType::Pawn);

        let curr_ksq_w = board.king_square(Color::White);
        let curr_ksq_b = board.king_square(Color::Black);
        let curr_flip_w = orient_flip(Color::White, curr_ksq_w);
        let curr_flip_b = orient_flip(Color::Black, curr_ksq_b);
        let curr_reg_w = board.heavy_regime(Color::White);
        let curr_reg_b = board.heavy_regime(Color::Black);
        let curr_kb_w = KING_BUCKET_LAYOUT[(curr_ksq_w as u8 ^ curr_flip_w) as usize];
        let curr_kb_b = KING_BUCKET_LAYOUT[(curr_ksq_b as u8 ^ curr_flip_b) as usize];

        let ply_slot = curr_ply & 63;
        let parent_ply_slot = curr_ply.wrapping_sub(1) & 63;
        // Theorem 35: Bijective 72-state per-perspective `(kb, reg, mirror)` index in `0..72`!
        let b_id_w = ((curr_kb_w as usize) * 3 + curr_reg_w) * 2 + usize::from(curr_flip_w != 0);
        let b_id_b = ((curr_kb_b as usize) * 3 + curr_reg_b) * 2 + usize::from((curr_flip_b & 7) != 0);
        let b_slot_w = 64 + b_id_w;
        let b_slot_b = 136 + b_id_b;

        // O(1) exact piece-placement check at ply_slot, parent_ply_slot (instant hit on Null Moves!), and bucket slots (`0` bytes copied!)
        for &chk in &[ply_slot, parent_ply_slot, b_slot_w, b_slot_b] {
            let entry = &self.history[chk];
            if entry.valid && entry.snap.colors == board.colors && entry.snap.pieces == board.pieces {
                self.cached_board = Some(b_hash);
                self.last_slot = chk;
                return chk;
            }
        }

        let mut prev_rows = std::mem::take(&mut self.scratch_prev);
        let mut curr_rows = std::mem::take(&mut self.scratch_curr);
        let mut adds = std::mem::take(&mut self.scratch_adds);
        let mut subs = std::mem::take(&mut self.scratch_subs);

        let bw_u8 = b_id_w as u8;
        let bb_u8 = b_id_b as u8;
        let same_w = |entry: &AccHistoryEntry| -> bool { entry.b_id_w == bw_u8 };
        let same_b = |entry: &AccHistoryEntry| -> bool { entry.b_id_b == bb_u8 };

        let mut shared_idx = usize::MAX;
        let mut shared_diff = usize::MAX;
        if self.cached_board.is_some() {
            for &idx in &[
                parent_ply_slot,
                ply_slot,
                curr_ply.wrapping_sub(2) & 63,
                curr_ply.wrapping_sub(3) & 63,
                curr_ply.wrapping_sub(4) & 63,
                (curr_ply + 1) & 63,
                b_slot_w,
                b_slot_b,
            ] {
                let entry = &self.history[idx];
                if entry.valid && same_w(entry) && same_b(entry) {
                    let occ_d = (entry.snap.occupancies() ^ b_occ).popcount();
                    if occ_d <= 2 {
                        shared_diff = occ_d;
                        shared_idx = idx;
                        break;
                    }
                    let p_pen = if entry.snap.colored_pieces(Color::White, PieceType::Pawn) != wp
                        || entry.snap.colored_pieces(Color::Black, PieceType::Pawn) != bp
                    {
                        3
                    } else {
                        0
                    };
                    let d = occ_d + p_pen;
                    if d < shared_diff {
                        shared_diff = d;
                        shared_idx = idx;
                    }
                }
            }
        }

        let slot = ply_slot;

        if shared_idx != usize::MAX && shared_diff <= 10 {
            // Fast Shared Path: both perspectives share `shared_idx` with 0 bucket/flip change!
            let prev_snap = &self.history[shared_idx].snap;
            let pawns_changed = prev_snap.colored_pieces(Color::White, PieceType::Pawn) != wp
                || prev_snap.colored_pieces(Color::Black, PieceType::Pawn) != bp;

            subs.clear();
            adds.clear();
            prev_rows.clear();
            curr_rows.clear();
            let changed_sq = diff_threat_rows_both(
                prev_snap,
                board,
                curr_flip_w,
                curr_flip_b,
                &mut subs,
                &mut adds,
                &mut prev_rows,
                &mut curr_rows,
            );
            if pawns_changed {
                diff_pawn_rows_both(
                    prev_snap,
                    board,
                    curr_flip_w,
                    curr_flip_b,
                    &mut subs,
                    &mut adds,
                    &mut prev_rows,
                    &mut curr_rows,
                );
            }

            let prev_occ = prev_snap.occupancies();
            for sq in changed_sq & prev_occ {
                let old_pc = prev_snap.piece_on(sq);
                subs.push(psq_row_index(old_pc.color(), old_pc.piece_type(), sq, curr_ksq_w, curr_reg_w, Color::White));
                prev_rows.push(psq_row_index(old_pc.color(), old_pc.piece_type(), sq, curr_ksq_b, curr_reg_b, Color::Black));
            }
            for sq in changed_sq & b_occ {
                let new_pc = board.piece_on(sq);
                adds.push(psq_row_index(new_pc.color(), new_pc.piece_type(), sq, curr_ksq_w, curr_reg_w, Color::White));
                curr_rows.push(psq_row_index(new_pc.color(), new_pc.piece_type(), sq, curr_ksq_b, curr_reg_b, Color::Black));
            }

            #[cfg(target_feature = "avx2")]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                for &r in &curr_rows {
                    _mm_prefetch(self.params.ft_weights.get_unchecked(r).as_ptr().cast(), _MM_HINT_T0);
                }
                for &r in &prev_rows {
                    _mm_prefetch(self.params.ft_weights.get_unchecked(r).as_ptr().cast(), _MM_HINT_T0);
                }
            }

            if shared_idx != slot {
                let (src_acc, dst_entry) = if shared_idx < slot {
                    let (l, r) = self.history.split_at_mut(slot);
                    (&l[shared_idx].acc, &mut r[0])
                } else {
                    let (l, r) = self.history.split_at_mut(shared_idx);
                    (&r[0].acc, &mut l[slot])
                };
                dst_entry.acc[0].apply_row_list(&src_acc[0], &adds, &subs, &self.params);
                dst_entry.acc[1].apply_row_list(&src_acc[1], &curr_rows, &prev_rows, &self.params);
            } else {
                let prev_a0 = self.history[slot].acc[0];
                let prev_a1 = self.history[slot].acc[1];
                self.history[slot].acc[0].apply_row_list(&prev_a0, &adds, &subs, &self.params);
                self.history[slot].acc[1].apply_row_list(&prev_a1, &curr_rows, &prev_rows, &self.params);
            }
        } else {
            // Split-Perspective Path (Theorem 35 & 45): update White (`p=0`) and Black (`p=1`) independently!
            for pov in [Color::White, Color::Black] {
                let p = pov as usize;
                let b_slot_p = if p == 0 { b_slot_w } else { b_slot_b };
                let mru_p = (self.bucket_last_slot[p][if p == 0 { b_id_w } else { b_id_b }] as usize).min(b_slot_p);
                let curr_ksq = if p == 0 { curr_ksq_w } else { curr_ksq_b };
                let curr_reg = if p == 0 { curr_reg_w } else { curr_reg_b };
                let curr_flip = if p == 0 { curr_flip_w } else { curr_flip_b };

                let mut best_p = usize::MAX;
                let mut best_d = usize::MAX;
                if self.cached_board.is_some() {
                    for &idx in &[
                        parent_ply_slot,
                        mru_p,
                        b_slot_p,
                        ply_slot,
                        curr_ply.wrapping_sub(2) & 63,
                        (curr_ply + 1) & 63,
                        curr_ply.wrapping_sub(3) & 63,
                    ] {
                        let entry = &self.history[idx];
                        if entry.valid && (if p == 0 { same_w(entry) } else { same_b(entry) }) {
                            let d = (entry.snap.occupancies() ^ b_occ).popcount();
                            if d < best_d {
                                best_d = d;
                                best_p = idx;
                                if d <= 2 {
                                    break;
                                }
                            }
                        }
                    }
                }

                if best_p != usize::MAX && best_d <= 8 {
                    subs.clear();
                    adds.clear();
                    let prev_snap = &self.history[best_p].snap;
                    let pawns_changed = prev_snap.colored_pieces(Color::White, PieceType::Pawn) != wp
                        || prev_snap.colored_pieces(Color::Black, PieceType::Pawn) != bp;
                    let changed_sq = diff_threat_rows_one(prev_snap, board, pov, curr_flip, &mut subs, &mut adds);
                    if pawns_changed {
                        diff_pawn_rows_one(prev_snap, board, pov, curr_flip, &mut subs, &mut adds);
                    }
                    let prev_occ = prev_snap.occupancies();
                    for sq in changed_sq & prev_occ {
                        let old_pc = prev_snap.piece_on(sq);
                        subs.push(psq_row_index(old_pc.color(), old_pc.piece_type(), sq, curr_ksq, curr_reg, pov));
                    }
                    for sq in changed_sq & b_occ {
                        let new_pc = board.piece_on(sq);
                        adds.push(psq_row_index(new_pc.color(), new_pc.piece_type(), sq, curr_ksq, curr_reg, pov));
                    }
                    let src_acc_p = self.history[best_p].acc[p];
                    self.history[slot].acc[p].apply_row_list(&src_acc_p, &adds, &subs, &self.params);
                } else if self.cached_board.is_some()
                    && self.history[parent_ply_slot].valid
                    && orient_flip(pov, self.history[parent_ply_slot].snap.king_square(pov)) == curr_flip
                {
                    // Same mirror orientation at `parent_ply_slot` (1 move away!): diff Threats+Pawns and rebuild only 30 PSQ rows!
                    subs.clear();
                    adds.clear();
                    let prev_snap = &self.history[parent_ply_slot].snap;
                    let pawns_changed = prev_snap.colored_pieces(Color::White, PieceType::Pawn) != wp
                        || prev_snap.colored_pieces(Color::Black, PieceType::Pawn) != bp;
                    let _ = diff_threat_rows_one(prev_snap, board, pov, curr_flip, &mut subs, &mut adds);
                    if pawns_changed {
                        diff_pawn_rows_one(prev_snap, board, pov, curr_flip, &mut subs, &mut adds);
                    }
                    let prev_ksq = prev_snap.king_square(pov);
                    let prev_reg = prev_snap.heavy_regime(pov);
                    for sq in prev_snap.occupancies() {
                        let old_pc = prev_snap.piece_on(sq);
                        subs.push(psq_row_index(old_pc.color(), old_pc.piece_type(), sq, prev_ksq, prev_reg, pov));
                    }
                    for sq in b_occ {
                        let new_pc = board.piece_on(sq);
                        adds.push(psq_row_index(new_pc.color(), new_pc.piece_type(), sq, curr_ksq, curr_reg, pov));
                    }
                    let src_acc_p = self.history[parent_ply_slot].acc[p];
                    self.history[slot].acc[p].apply_row_list(&src_acc_p, &adds, &subs, &self.params);
                } else {
                    let fresh = AccumulatorSide::new(&self.params);
                    curr_rows.clear();
                    collect_active_rows_snap(board, pov, &mut curr_rows);
                    self.history[slot].acc[p].apply_row_list(&fresh, &curr_rows, &[], &self.params);
                }
            }
        }

        self.pending_snaps[slot].0 = u64::MAX;
        self.history[slot].snap = *board;
        self.history[slot].b_id_w = bw_u8;
        self.history[slot].b_id_b = bb_u8;
        self.history[slot].valid = true;
        self.bucket_last_slot[0][b_id_w] = slot as u8;
        self.bucket_last_slot[1][b_id_b] = slot as u8;
        if shared_idx == usize::MAX || shared_diff > 6 || !self.history[b_slot_w].valid || !self.history[b_slot_b].valid {
            self.history[b_slot_w] = self.history[slot];
            self.history[b_slot_b] = self.history[slot];
        }
        self.history_cursor += 1;

        self.cached_board = Some(b_hash);
        self.last_slot = slot;
        self.scratch_prev = prev_rows;
        self.scratch_curr = curr_rows;
        self.scratch_adds = adds;
        self.scratch_subs = subs;
        slot
    }

    pub fn evaluate_psqt_only(&mut self, board: &Board) -> i32 {
        let slot = self.update_accumulator(board);
        let stm = board.side_to_move() as usize;
        let (psqt_b, _) = hybrid_stack_index(board);
        let acc = &self.history[slot].acc;
        (acc[stm].psqt[psqt_b] - acc[stm ^ 1].psqt[psqt_b]) >> 5
    }

    pub fn evaluate_with_components(&mut self, board: &Board, optimism: i32) -> (i32, i32, i32) {
        let slot = self.update_accumulator(board);

        let stm = board.side_to_move() as usize;
        let (pk_beta, pk_gamma) = evaluate_pawn_king_subnet(board, &self.params, &mut self.pk_cache, &mut self.last_pk_idx);
        let (psqt_b, stack_b) = hybrid_stack_index(board);
        let acc = &self.history[slot].acc;

        #[cfg(target_feature = "avx2")]
        let (psqt, pos, draw_l) = unsafe {
            forward_avx2(&acc[stm], &acc[stm ^ 1], pk_beta, pk_gamma, psqt_b, stack_b, &self.params)
        };
        #[cfg(not(target_feature = "avx2"))]
        let (psqt, pos, draw_l) =
            forward_scalar(&acc[stm], &acc[stm ^ 1], pk_beta, pk_gamma, psqt_b, stack_b, &self.params);

        let scaled = scale_atn1_evaluation(board, psqt, pos, draw_l, optimism, &self.params);
        (scaled, psqt, draw_l)
    }
}

static GLOBAL_MODEL: OnceLock<RwLock<Option<Arc<ModelParameters>>>> = OnceLock::new();
static NNUE_ACTIVE: AtomicBool = AtomicBool::new(false);

fn global_model_slot() -> &'static RwLock<Option<Arc<ModelParameters>>> {
    GLOBAL_MODEL.get_or_init(|| {
        let initial = std::env::var("ASURA_NNUE_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|p| ModelParameters::load_from_file(p).ok().map(Arc::new));
        NNUE_ACTIVE.store(initial.is_some(), Ordering::Relaxed);
        RwLock::new(initial)
    })
}

#[inline]
pub fn is_nnue_active() -> bool {
    if GLOBAL_MODEL.get().is_none() {
        let _ = global_model_slot();
    }
    NNUE_ACTIVE.load(Ordering::Relaxed)
}

pub fn load_global_model(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    let params = Arc::new(ModelParameters::load_from_file(path)?);
    *global_model_slot().write().unwrap() = Some(params);
    NNUE_ACTIVE.store(true, Ordering::Relaxed);
    Ok(())
}

pub fn clear_global_model() {
    *global_model_slot().write().unwrap() = None;
    NNUE_ACTIVE.store(false, Ordering::Relaxed);
    TLS_NNUE.with(|slot| *slot.borrow_mut() = None);
}

pub fn global_model() -> Option<Arc<ModelParameters>> {
    global_model_slot().read().unwrap().clone()
}

thread_local! {
    static TLS_NNUE: std::cell::RefCell<Option<NetworkState>> = const { std::cell::RefCell::new(None) };
}

pub fn take_tls_nnue() -> Option<NetworkState> {
    let params = global_model()?;
    if let Some(mut net) = TLS_NNUE.with(|slot| slot.borrow_mut().take())
        && Arc::ptr_eq(&net.params, &params)
    {
        net.cached_board = None;
        return Some(net);
    }
    Some(NetworkState::new(params))
}

pub fn put_tls_nnue(net: NetworkState) {
    TLS_NNUE.with(|slot| {
        let mut guard = slot.borrow_mut();
        if guard.is_none() {
            *guard = Some(net);
        }
    });
}
