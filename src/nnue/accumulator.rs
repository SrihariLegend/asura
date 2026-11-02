pub mod psq;
pub mod threats;

pub use psq::{collect_psq_rows, heavy_regime, orient_flip, psq_row_index};
pub use threats::{
    BoardSnapshot, collect_active_rows_snap, collect_pawn_rows, collect_threat_rows, collect_threat_rows_both,
    diff_pawn_rows_both, diff_pawn_rows_one, diff_threat_rows_both, diff_threat_rows_one, passed_pawn_row_index,
    pawn_pair_row_index, tables, threat_row_index,
};

use crate::{
    board::Board,
    nnue::{Aligned, L1_SIZE, L2_SIZE, ModelParameters, PK_CACHE_SIZE, PSQT_BUCKETS},
    types::{Bitboard, Color, PieceType},
};

#[derive(Copy, Clone)]
pub struct AccumulatorSide {
    pub values: Aligned<[i16; L1_SIZE]>,
    pub psqt: [i32; PSQT_BUCKETS],
}

impl AccumulatorSide {
    pub fn new(params: &ModelParameters) -> Self {
        Self { values: params.ft_biases, psqt: [0; PSQT_BUCKETS] }
    }

    #[inline]
    pub fn add_row(&mut self, weights: &[i8; L1_SIZE], psqt: &[i32; PSQT_BUCKETS]) {
        for i in 0..L1_SIZE {
            self.values[i] = self.values[i].wrapping_add(weights[i] as i16);
        }
        for b in 0..PSQT_BUCKETS {
            self.psqt[b] = self.psqt[b].wrapping_add(psqt[b]);
        }
    }

    #[inline]
    pub fn sub_row(&mut self, weights: &[i8; L1_SIZE], psqt: &[i32; PSQT_BUCKETS]) {
        for i in 0..L1_SIZE {
            self.values[i] = self.values[i].wrapping_sub(weights[i] as i16);
        }
        for b in 0..PSQT_BUCKETS {
            self.psqt[b] = self.psqt[b].wrapping_sub(psqt[b]);
        }
    }

    pub fn apply_row_list(&mut self, prev: &Self, adds: &[usize], subs: &[usize], params: &ModelParameters) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            self.apply_row_list_avx2(prev, adds, subs, params);
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            self.values = prev.values;
            self.psqt = prev.psqt;
            for &a in adds {
                self.add_row(&params.ft_weights[a].data, &params.ft_psqt[a]);
            }
            for &s in subs {
                self.sub_row(&params.ft_weights[s].data, &params.ft_psqt[s]);
            }
        }
    }

    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn apply_row_list_avx2(&mut self, prev: &Self, adds: &[usize], subs: &[usize], params: &ModelParameters) {
        use std::arch::x86_64::*;

        let mut psqt_vec = _mm256_loadu_si256(prev.psqt.as_ptr().cast());
        for &a in adds {
            let r_psqt = _mm256_loadu_si256(params.ft_psqt.get_unchecked(a).as_ptr().cast());
            psqt_vec = _mm256_add_epi32(psqt_vec, r_psqt);
        }
        for &s in subs {
            let r_psqt = _mm256_loadu_si256(params.ft_psqt.get_unchecked(s).as_ptr().cast());
            psqt_vec = _mm256_sub_epi32(psqt_vec, r_psqt);
        }
        _mm256_storeu_si256(self.psqt.as_mut_ptr().cast(), psqt_vec);

        for tile_off in (0..L1_SIZE).step_by(128) {
            let src_ptr = prev.values.as_ptr().add(tile_off);
            let dst_ptr = self.values.as_mut_ptr().add(tile_off);
            let mut r0 = _mm256_load_si256(src_ptr.add(0).cast());
            let mut r1 = _mm256_load_si256(src_ptr.add(16).cast());
            let mut r2 = _mm256_load_si256(src_ptr.add(32).cast());
            let mut r3 = _mm256_load_si256(src_ptr.add(48).cast());
            let mut r4 = _mm256_load_si256(src_ptr.add(64).cast());
            let mut r5 = _mm256_load_si256(src_ptr.add(80).cast());
            let mut r6 = _mm256_load_si256(src_ptr.add(96).cast());
            let mut r7 = _mm256_load_si256(src_ptr.add(112).cast());

            let mut ai = 0usize;
            let mut si = 0usize;
            while ai < adds.len() && si < subs.len() {
                let wa = params.ft_weights.get_unchecked(*adds.get_unchecked(ai)).as_ptr().add(tile_off);
                let ws = params.ft_weights.get_unchecked(*subs.get_unchecked(si)).as_ptr().add(tile_off);
                r0 = _mm256_add_epi16(r0, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(0).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(0).cast()))));
                r1 = _mm256_add_epi16(r1, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(16).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(16).cast()))));
                r2 = _mm256_add_epi16(r2, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(32).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(32).cast()))));
                r3 = _mm256_add_epi16(r3, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(48).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(48).cast()))));
                r4 = _mm256_add_epi16(r4, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(64).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(64).cast()))));
                r5 = _mm256_add_epi16(r5, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(80).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(80).cast()))));
                r6 = _mm256_add_epi16(r6, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(96).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(96).cast()))));
                r7 = _mm256_add_epi16(r7, _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(wa.add(112).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(ws.add(112).cast()))));
                ai += 1;
                si += 1;
            }

            while ai + 1 < adds.len() {
                let w1 = params.ft_weights.get_unchecked(*adds.get_unchecked(ai)).as_ptr().add(tile_off);
                let w2 = params.ft_weights.get_unchecked(*adds.get_unchecked(ai + 1)).as_ptr().add(tile_off);
                r0 = _mm256_add_epi16(r0, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(0).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(0).cast()))));
                r1 = _mm256_add_epi16(r1, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(16).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(16).cast()))));
                r2 = _mm256_add_epi16(r2, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(32).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(32).cast()))));
                r3 = _mm256_add_epi16(r3, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(48).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(48).cast()))));
                r4 = _mm256_add_epi16(r4, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(64).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(64).cast()))));
                r5 = _mm256_add_epi16(r5, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(80).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(80).cast()))));
                r6 = _mm256_add_epi16(r6, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(96).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(96).cast()))));
                r7 = _mm256_add_epi16(r7, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(112).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(112).cast()))));
                ai += 2;
            }
            while ai < adds.len() {
                let w_ptr = params.ft_weights.get_unchecked(*adds.get_unchecked(ai)).as_ptr().add(tile_off);
                r0 = _mm256_add_epi16(r0, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(0).cast())));
                r1 = _mm256_add_epi16(r1, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(16).cast())));
                r2 = _mm256_add_epi16(r2, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(32).cast())));
                r3 = _mm256_add_epi16(r3, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(48).cast())));
                r4 = _mm256_add_epi16(r4, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(64).cast())));
                r5 = _mm256_add_epi16(r5, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(80).cast())));
                r6 = _mm256_add_epi16(r6, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(96).cast())));
                r7 = _mm256_add_epi16(r7, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(112).cast())));
                ai += 1;
            }

            while si + 1 < subs.len() {
                let w1 = params.ft_weights.get_unchecked(*subs.get_unchecked(si)).as_ptr().add(tile_off);
                let w2 = params.ft_weights.get_unchecked(*subs.get_unchecked(si + 1)).as_ptr().add(tile_off);
                r0 = _mm256_sub_epi16(r0, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(0).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(0).cast()))));
                r1 = _mm256_sub_epi16(r1, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(16).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(16).cast()))));
                r2 = _mm256_sub_epi16(r2, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(32).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(32).cast()))));
                r3 = _mm256_sub_epi16(r3, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(48).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(48).cast()))));
                r4 = _mm256_sub_epi16(r4, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(64).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(64).cast()))));
                r5 = _mm256_sub_epi16(r5, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(80).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(80).cast()))));
                r6 = _mm256_sub_epi16(r6, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(96).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(96).cast()))));
                r7 = _mm256_sub_epi16(r7, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(112).cast())), _mm256_cvtepi8_epi16(_mm_loadu_si128(w2.add(112).cast()))));
                si += 2;
            }
            while si < subs.len() {
                let w_ptr = params.ft_weights.get_unchecked(*subs.get_unchecked(si)).as_ptr().add(tile_off);
                r0 = _mm256_sub_epi16(r0, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(0).cast())));
                r1 = _mm256_sub_epi16(r1, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(16).cast())));
                r2 = _mm256_sub_epi16(r2, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(32).cast())));
                r3 = _mm256_sub_epi16(r3, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(48).cast())));
                r4 = _mm256_sub_epi16(r4, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(64).cast())));
                r5 = _mm256_sub_epi16(r5, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(80).cast())));
                r6 = _mm256_sub_epi16(r6, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(96).cast())));
                r7 = _mm256_sub_epi16(r7, _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(112).cast())));
                si += 1;
            }

            _mm256_store_si256(dst_ptr.add(0).cast(), r0);
            _mm256_store_si256(dst_ptr.add(16).cast(), r1);
            _mm256_store_si256(dst_ptr.add(32).cast(), r2);
            _mm256_store_si256(dst_ptr.add(48).cast(), r3);
            _mm256_store_si256(dst_ptr.add(64).cast(), r4);
            _mm256_store_si256(dst_ptr.add(80).cast(), r5);
            _mm256_store_si256(dst_ptr.add(96).cast(), r6);
            _mm256_store_si256(dst_ptr.add(112).cast(), r7);
        }
    }
}

pub fn collect_active_rows(board: &Board, pov: Color, out: &mut Vec<usize>) {
    out.clear();
    collect_psq_rows(board, pov, out);
    collect_threat_rows(board, pov, out);
    collect_pawn_rows(board, pov, out);
}

#[derive(Copy, Clone)]
pub struct FinnyEntry {
    pub psq_acc: AccumulatorSide,
    pub pieces: [Bitboard; PieceType::NUM],
    pub colors: [Bitboard; Color::NUM],
    pub valid: bool,
}

impl FinnyEntry {
    pub fn new(params: &ModelParameters) -> Self {
        Self {
            psq_acc: AccumulatorSide::new(params),
            pieces: [Bitboard(0); PieceType::NUM],
            colors: [Bitboard(0); Color::NUM],
            valid: false,
        }
    }
}

#[inline]
pub(crate) fn dilate_moore(x: u64) -> u64 {
    let l = (x >> 1) & 0x7f7f_7f7f_7f7f_7f7f;
    let r = (x << 1) & 0xfefe_fefe_fefe_fefe;
    let h = x | l | r;
    h | (h << 8) | (h >> 8)
}

#[inline]
fn morph_black_tophat(x: u64) -> u64 {
    let dilated = dilate_moore(x);
    let closed = !dilate_moore(!dilated);
    closed & !x
}

#[inline]
fn betti0_pawn_islands(pawns: u64) -> usize {
    let r4 = pawns | (pawns >> 32);
    let r2 = r4 | (r4 >> 16);
    let fmask = ((r2 | (r2 >> 8)) & 0xFF) as u8;
    (fmask & !(fmask << 1)).count_ones() as usize
}

#[derive(Copy, Clone)]
pub struct PkCacheEntry {
    pub wp: u64,
    pub bp: u64,
    pub kk: u16,
    pub valid_mask: u8,
    pub beta: [[i32; L2_SIZE]; 2],
    pub gamma: [[i32; L2_SIZE]; 2],
}

impl Default for PkCacheEntry {
    fn default() -> Self {
        Self {
            wp: u64::MAX,
            bp: u64::MAX,
            kk: u16::MAX,
            valid_mask: 0,
            beta: [[0; L2_SIZE]; 2],
            gamma: [[0; L2_SIZE]; 2],
        }
    }
}

pub fn evaluate_pawn_king_subnet<'a>(
    board: &Board, params: &ModelParameters, cache: &'a mut [PkCacheEntry; PK_CACHE_SIZE], last_idx: &mut usize,
) -> (&'a [i32; L2_SIZE], &'a [i32; L2_SIZE]) {
    let stm = board.side_to_move();
    let s_idx = stm as usize;
    let s_bit = 1u8 << s_idx;
    let wp = board.colored_pieces(Color::White, PieceType::Pawn).0;
    let bp = board.colored_pieces(Color::Black, PieceType::Pawn).0;
    let wk = board.king_square(Color::White) as u16;
    let bk = board.king_square(Color::Black) as u16;
    let kk = wk | (bk << 6);

    let prev_i = *last_idx & (PK_CACHE_SIZE - 1);
    if cache[prev_i].wp == wp && cache[prev_i].bp == bp && cache[prev_i].kk == kk && (cache[prev_i].valid_mask & s_bit) != 0 {
        return (&cache[prev_i].beta[s_idx], &cache[prev_i].gamma[s_idx]);
    }

    let hash = (wp.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ (bp.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        ^ ((kk as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    let idx = (hash as usize) & (PK_CACHE_SIZE - 1);
    *last_idx = idx;
    let slot = &mut cache[idx];
    if slot.wp == wp && slot.bp == bp && slot.kk == kk {
        if (slot.valid_mask & s_bit) != 0 {
            return (&slot.beta[s_idx], &slot.gamma[s_idx]);
        }
    } else {
        slot.wp = wp;
        slot.bp = bp;
        slot.kk = kk;
        slot.valid_mask = 0;
    }

    let flip = if stm == Color::Black { 56usize } else { 0usize };
    let own_p = if stm == Color::White { wp } else { bp };
    let opp_p = if stm == Color::White { bp } else { wp };
    let own_ksq = board.king_square(stm) as usize;
    let opp_ksq = board.king_square(!stm) as usize;

    let mut active = [0usize; 128];
    let mut n_act = 0usize;

    let mut b = own_p;
    while b != 0 {
        let s = b.trailing_zeros() as usize;
        b &= b - 1;
        active[n_act] = s ^ flip;
        n_act += 1;
    }
    let mut b = opp_p;
    while b != 0 {
        let s = b.trailing_zeros() as usize;
        b &= b - 1;
        active[n_act] = 64 + (s ^ flip);
        n_act += 1;
    }
    active[n_act] = 128 + (own_ksq ^ flip);
    n_act += 1;
    active[n_act] = 192 + (opp_ksq ^ flip);
    n_act += 1;

    let rank2_7 = !0xFFu64 & !(0xFFu64 << 56);
    let own_holes = morph_black_tophat(own_p | (1u64 << own_ksq)) & rank2_7;
    let opp_holes = morph_black_tophat(opp_p | (1u64 << opp_ksq)) & rank2_7;
    let mut h = own_holes;
    while h != 0 {
        let s = h.trailing_zeros() as usize;
        h &= h - 1;
        active[n_act] = 248 + (s ^ flip);
        n_act += 1;
    }
    let mut h = opp_holes;
    while h != 0 {
        let s = h.trailing_zeros() as usize;
        h &= h - 1;
        active[n_act] = 296 + (s ^ flip);
        n_act += 1;
    }

    let b0_own = betti0_pawn_islands(own_p).clamp(1, 4) - 1;
    let b0_opp = betti0_pawn_islands(opp_p).clamp(1, 4) - 1;
    active[n_act] = 352 + b0_own;
    n_act += 1;
    active[n_act] = 356 + b0_opp;
    n_act += 1;

    let rammed = ((wp << 8) & bp).count_ones().min(3) as usize;
    active[n_act] = 360 + rammed;
    n_act += 1;
    let light_mask = 0x55AA_55AA_55AA_55AAu64;
    if (own_p & light_mask).count_ones() > (own_p & !light_mask).count_ones() {
        active[n_act] = 364;
        n_act += 1;
    }
    if (opp_p & light_mask).count_ones() > (opp_p & !light_mask).count_ones() {
        active[n_act] = 365;
        n_act += 1;
    }
    let own_unstopped = if stm == Color::White {
        let mut bs = bp >> 8;
        bs |= bs >> 8;
        bs |= bs >> 16;
        bs |= bs >> 32;
        wp & !(bs | ((bs >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((bs << 1) & 0xfefe_fefe_fefe_fefe))
    } else {
        let mut wn = wp << 8;
        wn |= wn << 8;
        wn |= wn << 16;
        wn |= wn << 32;
        bp & !(wn | ((wn >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((wn << 1) & 0xfefe_fefe_fefe_fefe))
    };
    if own_unstopped != 0 {
        let opp_kr = (opp_ksq / 8) as isize;
        let opp_kf = (opp_ksq % 8) as isize;
        let promo_r = if stm == Color::White { 7isize } else { 0isize };
        let mut p_it = own_unstopped;
        while p_it != 0 {
            let s = p_it.trailing_zeros() as usize;
            p_it &= p_it - 1;
            let sr = (s / 8) as isize;
            let sf = (s % 8) as isize;
            let dist_p = (promo_r - sr).abs();
            let dist_k = (promo_r - opp_kr).abs().max((sf - opp_kf).abs());
            if dist_p < dist_k {
                active[n_act] = 366;
                n_act += 1;
                break;
            }
        }
    }

    let mut beta = params.pk_beta_biases;
    let mut gamma = params.pk_gamma_biases;

    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let mut h0 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(0).cast());
        let mut h1 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(16).cast());
        let mut h2 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(32).cast());
        let mut h3 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(48).cast());
        let mut h4 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(64).cast());
        let mut h5 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(80).cast());
        let mut h6 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(96).cast());
        let mut h7 = _mm256_load_si256(params.pk_biases_1.as_ptr().add(112).cast());

        for &idx in &active[..n_act] {
            let rp = params.pk_weights_1.get_unchecked(idx).as_ptr();
            h0 = _mm256_adds_epi16(h0, _mm256_loadu_si256(rp.add(0).cast()));
            h1 = _mm256_adds_epi16(h1, _mm256_loadu_si256(rp.add(16).cast()));
            h2 = _mm256_adds_epi16(h2, _mm256_loadu_si256(rp.add(32).cast()));
            h3 = _mm256_adds_epi16(h3, _mm256_loadu_si256(rp.add(48).cast()));
            h4 = _mm256_adds_epi16(h4, _mm256_loadu_si256(rp.add(64).cast()));
            h5 = _mm256_adds_epi16(h5, _mm256_loadu_si256(rp.add(80).cast()));
            h6 = _mm256_adds_epi16(h6, _mm256_loadu_si256(rp.add(96).cast()));
            h7 = _mm256_adds_epi16(h7, _mm256_loadu_si256(rp.add(112).cast()));
        }

        let zero = _mm256_setzero_si256();
        let top = _mm256_set1_epi16(127);
        let c0 = _mm256_min_epi16(_mm256_max_epi16(h0, zero), top);
        let c1 = _mm256_min_epi16(_mm256_max_epi16(h1, zero), top);
        let c2 = _mm256_min_epi16(_mm256_max_epi16(h2, zero), top);
        let c3 = _mm256_min_epi16(_mm256_max_epi16(h3, zero), top);
        let c4 = _mm256_min_epi16(_mm256_max_epi16(h4, zero), top);
        let c5 = _mm256_min_epi16(_mm256_max_epi16(h5, zero), top);
        let c6 = _mm256_min_epi16(_mm256_max_epi16(h6, zero), top);
        let c7 = _mm256_min_epi16(_mm256_max_epi16(h7, zero), top);

        let a0 = _mm256_permute4x64_epi64(_mm256_packus_epi16(c0, c1), 0xD8);
        let a1 = _mm256_permute4x64_epi64(_mm256_packus_epi16(c2, c3), 0xD8);
        let a2 = _mm256_permute4x64_epi64(_mm256_packus_epi16(c4, c5), 0xD8);
        let a3 = _mm256_permute4x64_epi64(_mm256_packus_epi16(c6, c7), 0xD8);
        let ones16 = _mm256_set1_epi16(1);

        let row_dot = |w_ptr: *const i8| -> __m256i {
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_loadu_si256(w_ptr.add(0).cast())), ones16),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_loadu_si256(w_ptr.add(32).cast())), ones16),
                ),
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a2, _mm256_loadu_si256(w_ptr.add(64).cast())), ones16),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a3, _mm256_loadu_si256(w_ptr.add(96).cast())), ones16),
                ),
            )
        };

        for j in (0..L2_SIZE).step_by(4) {
            let b0 = row_dot(params.pk_beta_weights[j].as_ptr());
            let b1 = row_dot(params.pk_beta_weights[j + 1].as_ptr());
            let b2 = row_dot(params.pk_beta_weights[j + 2].as_ptr());
            let b3 = row_dot(params.pk_beta_weights[j + 3].as_ptr());
            let bh = _mm256_hadd_epi32(_mm256_hadd_epi32(b0, b1), _mm256_hadd_epi32(b2, b3));
            let b_sum4 = _mm_add_epi32(_mm256_castsi256_si128(bh), _mm256_extracti128_si256(bh, 1));
            let b_bias = _mm_loadu_si128(beta.as_ptr().add(j).cast());
            _mm_storeu_si128(beta.as_mut_ptr().add(j).cast(), _mm_add_epi32(b_bias, b_sum4));

            let g0 = row_dot(params.pk_gamma_weights[j].as_ptr());
            let g1 = row_dot(params.pk_gamma_weights[j + 1].as_ptr());
            let g2 = row_dot(params.pk_gamma_weights[j + 2].as_ptr());
            let g3 = row_dot(params.pk_gamma_weights[j + 3].as_ptr());
            let gh = _mm256_hadd_epi32(_mm256_hadd_epi32(g0, g1), _mm256_hadd_epi32(g2, g3));
            let g_sum4 = _mm_add_epi32(_mm256_castsi256_si128(gh), _mm256_extracti128_si256(gh, 1));
            let g_bias = _mm_loadu_si128(gamma.as_ptr().add(j).cast());
            let g_tot = _mm_srai_epi32(_mm_add_epi32(g_bias, g_sum4), 5);
            let g_clamped = _mm_min_epi32(_mm_max_epi32(g_tot, _mm_set1_epi32(-127)), _mm_set1_epi32(127));
            _mm_storeu_si128(gamma.as_mut_ptr().add(j).cast(), g_clamped);
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        let mut h_pk = *params.pk_biases_1;
        for &idx in &active[..n_act] {
            let row = &params.pk_weights_1[idx];
            for k in 0..PK_HIDDEN {
                h_pk[k] = h_pk[k].saturating_add(row[k]);
            }
        }
        let mut act = [0u8; PK_HIDDEN];
        for k in 0..PK_HIDDEN {
            act[k] = h_pk[k].clamp(0, 127) as u8;
        }
        for k in 0..PK_HIDDEN {
            let a = act[k] as i32;
            if a != 0 {
                for j in 0..L2_SIZE {
                    beta[j] += a * (params.pk_beta_weights[j][k] as i32);
                    gamma[j] += a * (params.pk_gamma_weights[j][k] as i32);
                }
            }
        }
        for j in 0..L2_SIZE {
            gamma[j] = (gamma[j] >> 5).clamp(-127, 127);
        }
    }

    slot.valid_mask |= s_bit;
    slot.beta[s_idx] = beta;
    slot.gamma[s_idx] = gamma;
    (&slot.beta[s_idx], &slot.gamma[s_idx])
}
