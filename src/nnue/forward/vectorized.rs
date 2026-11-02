#[cfg(target_feature = "avx2")]
use crate::nnue::{AccumulatorSide, Aligned, L1_HALF, L1_SIZE, L2_SIZE, ModelParameters, simd};

#[cfg(target_feature = "avx2")]
#[repr(align(16))]
#[derive(Copy, Clone)]
struct SparseEntry {
    indexes: [u16; 8],
    count: usize,
}

#[cfg(target_feature = "avx2")]
const NNZ_TABLE: [SparseEntry; 256] = {
    let mut table = [SparseEntry { indexes: [0; 8], count: 0 }; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut count = 0usize;
        let mut bit = 0usize;
        while bit < 8 {
            if (byte & (1 << bit)) != 0 {
                table[byte].indexes[count] = bit as u16;
                count += 1;
            }
            bit += 1;
        }
        table[byte].count = count;
        byte += 1;
    }
    table
};

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
pub unsafe fn forward_avx2(
    us: &AccumulatorSide, them: &AccumulatorSide, pk_beta: &[i32; L2_SIZE], pk_gamma: &[i32; L2_SIZE],
    psqt_bucket: usize, stack_bucket: usize, params: &ModelParameters,
) -> (i32, i32, i32) {
    use std::arch::x86_64::*;

    let mut l0 = Aligned::new([0u8; L1_SIZE]);
    let zero = simd::zeroed();
    let top = simd::splat_i16(255);

    let st = &params.stacks[stack_bucket];
    let ones16 = simd::splat_i16(1);
    let mut q_vec = simd::zeroed();
    let mut nnz = Aligned::new([0u16; L1_SIZE / 4 + 8]);
    let mut nnz_len = 0usize;
    let mut v_base = _mm_setzero_si128();
    let v_step = _mm_set1_epi16(8);

    for (side_idx, acc) in [us, them].iter().enumerate() {
        let base = side_idx * L1_HALF;
        for i in (0..L1_HALF).step_by(2 * simd::I16_LANES) {
            let l_lo = _mm256_load_si256(acc.values.as_ptr().add(i).cast());
            let l_hi = _mm256_load_si256(acc.values.as_ptr().add(i + simd::I16_LANES).cast());
            let r_lo = _mm256_load_si256(acc.values.as_ptr().add(L1_HALF + i).cast());
            let r_hi = _mm256_load_si256(acc.values.as_ptr().add(L1_HALF + i + simd::I16_LANES).cast());

            let lc_lo = simd::shift_left_i16::<7>(simd::clamp_i16(l_lo, zero, top));
            let lc_hi = simd::shift_left_i16::<7>(simd::clamp_i16(l_hi, zero, top));
            let rc_lo = simd::min_i16(r_lo, top);
            let rc_hi = simd::min_i16(r_hi, top);

            let p_lo = simd::mul_high_i16(lc_lo, rc_lo);
            let p_hi = simd::mul_high_i16(lc_hi, rc_hi);
            let packed = simd::permute(simd::packus(p_lo, p_hi));
            let off = base + i;
            _mm256_store_si256(l0.as_mut_ptr().add(off).cast(), packed);

            let qw = _mm256_loadu_si256(st.quad_skip_weights.as_ptr().add(off).cast());
            q_vec = _mm256_add_epi32(q_vec, _mm256_madd_epi16(_mm256_maddubs_epi16(packed, qw), ones16));

            let mask = ((_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(packed, zero))) as usize) ^ 0xFF) & 0xFF;
            let entry = NNZ_TABLE.get_unchecked(mask);
            let offsets = _mm_load_si128(entry.indexes.as_ptr().cast());
            _mm_storeu_si128(nnz.as_mut_ptr().add(nnz_len).cast(), _mm_add_epi16(offsets, v_base));
            nnz_len += entry.count;
            v_base = _mm_add_epi16(v_base, v_step);
        }
    }

    let mut acc_lo0 = _mm256_loadu_si256(st.fc1_biases.as_ptr().cast());
    let mut acc_hi0 = _mm256_loadu_si256(st.fc1_biases.as_ptr().add(8).cast());
    let mut acc_lo1 = simd::zeroed();
    let mut acc_hi1 = simd::zeroed();

    let l0_words = l0.as_ptr() as *const i32;
    let mut k = 0usize;
    while k + 1 < nnz_len {
        let g0 = *nnz.get_unchecked(k) as usize;
        let g1 = *nnz.get_unchecked(k + 1) as usize;
        let x0 = simd::splat_i32(*l0_words.add(g0));
        let x1 = simd::splat_i32(*l0_words.add(g1));
        let wp0 = st.fc1_weights.as_ptr().add(g0 * L2_SIZE * 4);
        let wp1 = st.fc1_weights.as_ptr().add(g1 * L2_SIZE * 4);
        acc_lo0 = _mm256_add_epi32(acc_lo0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_loadu_si256(wp0.cast())), ones16));
        acc_hi0 = _mm256_add_epi32(acc_hi0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_loadu_si256(wp0.add(32).cast())), ones16));
        acc_lo1 = _mm256_add_epi32(acc_lo1, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, _mm256_loadu_si256(wp1.cast())), ones16));
        acc_hi1 = _mm256_add_epi32(acc_hi1, _mm256_madd_epi16(_mm256_maddubs_epi16(x1, _mm256_loadu_si256(wp1.add(32).cast())), ones16));
        k += 2;
    }
    if k < nnz_len {
        let g0 = *nnz.get_unchecked(k) as usize;
        let x0 = simd::splat_i32(*l0_words.add(g0));
        let wp0 = st.fc1_weights.as_ptr().add(g0 * L2_SIZE * 4);
        acc_lo0 = _mm256_add_epi32(acc_lo0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_loadu_si256(wp0.cast())), ones16));
        acc_hi0 = _mm256_add_epi32(acc_hi0, _mm256_madd_epi16(_mm256_maddubs_epi16(x0, _mm256_loadu_si256(wp0.add(32).cast())), ones16));
    }
    let acc_lo = _mm256_add_epi32(acc_lo0, acc_lo1);
    let acc_hi = _mm256_add_epi32(acc_hi0, acc_hi1);

    let q128 = _mm_add_epi32(_mm256_castsi256_si128(q_vec), _mm256_extracti128_si256(q_vec, 1));
    let q64 = _mm_add_epi32(q128, _mm_shuffle_epi32(q128, 0x4E));
    let quad_skip = _mm_cvtsi128_si32(_mm_add_epi32(q64, _mm_shuffle_epi32(q64, 0xB1)));

    // Theorem 36: Register-resident AVX2 FiLM Gate + 32-bit Clamped SqrCReLU for `L2` (`a1_vec`)
    let b_lo = _mm256_loadu_si256(pk_beta.as_ptr().cast());
    let b_hi = _mm256_loadu_si256(pk_beta.as_ptr().add(8).cast());
    let g_lo = _mm256_loadu_si256(pk_gamma.as_ptr().cast());
    let g_hi = _mm256_loadu_si256(pk_gamma.as_ptr().add(8).cast());

    let x1_lo = _mm256_add_epi32(_mm256_add_epi32(acc_lo, b_lo), _mm256_mullo_epi32(g_lo, _mm256_srai_epi32(acc_lo, 7)));
    let x1_hi = _mm256_add_epi32(_mm256_add_epi32(acc_hi, b_hi), _mm256_mullo_epi32(g_hi, _mm256_srai_epi32(acc_hi, 7)));

    let hi128 = _mm256_extracti128_si256(x1_hi, 1);
    let skip = _mm_extract_epi32(hi128, 2) - _mm_extract_epi32(hi128, 3);

    let cap8192 = _mm256_set1_epi32(8192);
    let cap127 = _mm256_set1_epi32(127);

    let v1_lo = _mm256_min_epi32(_mm256_max_epi32(x1_lo, zero), cap8192);
    let v1_hi = _mm256_min_epi32(_mm256_max_epi32(x1_hi, zero), cap8192);

    let sq1_lo = _mm256_min_epi32(_mm256_srli_epi32(_mm256_mullo_epi32(v1_lo, v1_lo), 19), cap127);
    let sq1_hi = _mm256_min_epi32(_mm256_srli_epi32(_mm256_mullo_epi32(v1_hi, v1_hi), 19), cap127);
    let lin1_lo = _mm256_min_epi32(_mm256_srli_epi32(v1_lo, 6), cap127);
    let lin1_hi = _mm256_min_epi32(_mm256_srli_epi32(v1_hi, 6), cap127);

    let sq1_16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(sq1_lo, sq1_hi), 0xD8);
    let lin1_16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(lin1_lo, lin1_hi), 0xD8);
    let a1_vec = _mm256_permute4x64_epi64(_mm256_packus_epi16(sq1_16, lin1_16), 0xD8);

    // Theorem 36: Register-resident AVX2 `FC2` + 32-bit Clamped SqrCReLU for `L3` (`a2_lo, a2_hi`)
    let mut sq2_16 = [simd::zeroed(); 2];
    let mut lin2_16 = [simd::zeroed(); 2];
    for blk in 0..2 {
        let j_base = blk * 16;
        let mut x2_pair = [simd::zeroed(); 2];
        for sub in 0..2 {
            let j = j_base + sub * 8;
            let wp = st.fc2_weights.as_ptr().add(j * 32);
            let d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(0).cast())), ones16);
            let d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(32).cast())), ones16);
            let d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(64).cast())), ones16);
            let d3 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(96).cast())), ones16);
            let h01 = _mm256_hadd_epi32(d0, d1);
            let h23 = _mm256_hadd_epi32(d2, d3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let sum_lo = _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));

            let d4 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(128).cast())), ones16);
            let d5 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(160).cast())), ones16);
            let d6 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(192).cast())), ones16);
            let d7 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(wp.add(224).cast())), ones16);
            let h45 = _mm256_hadd_epi32(d4, d5);
            let h67 = _mm256_hadd_epi32(d6, d7);
            let h4567 = _mm256_hadd_epi32(h45, h67);
            let sum_hi = _mm_add_epi32(_mm256_castsi256_si128(h4567), _mm256_extracti128_si256(h4567, 1));

            let sum8 = _mm256_inserti128_si256(_mm256_castsi128_si256(sum_lo), sum_hi, 1);
            let b8 = _mm256_loadu_si256(st.fc2_biases.as_ptr().add(j).cast());
            x2_pair[sub] = _mm256_add_epi32(sum8, b8);
        }
        let v_lo = _mm256_min_epi32(_mm256_max_epi32(x2_pair[0], zero), cap8192);
        let v_hi = _mm256_min_epi32(_mm256_max_epi32(x2_pair[1], zero), cap8192);
        let sq_lo = _mm256_min_epi32(_mm256_srli_epi32(_mm256_mullo_epi32(v_lo, v_lo), 19), cap127);
        let sq_hi = _mm256_min_epi32(_mm256_srli_epi32(_mm256_mullo_epi32(v_hi, v_hi), 19), cap127);
        let lin_lo = _mm256_min_epi32(_mm256_srli_epi32(v_lo, 6), cap127);
        let lin_hi = _mm256_min_epi32(_mm256_srli_epi32(v_hi, 6), cap127);

        sq2_16[blk] = _mm256_permute4x64_epi64(_mm256_packs_epi32(sq_lo, sq_hi), 0xD8);
        lin2_16[blk] = _mm256_permute4x64_epi64(_mm256_packs_epi32(lin_lo, lin_hi), 0xD8);
    }

    let a2_lo = _mm256_permute4x64_epi64(_mm256_packus_epi16(sq2_16[0], sq2_16[1]), 0xD8);
    let a2_hi = _mm256_permute4x64_epi64(_mm256_packus_epi16(lin2_16[0], lin2_16[1]), 0xD8);

    let w3_0 = st.fc3_weights[0].as_ptr();
    let w3_1 = st.fc3_weights[1].as_ptr();
    let s0 = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(w3_0.add(0).cast())), ones16),
        _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_maddubs_epi16(a2_lo, _mm256_loadu_si256(w3_0.add(32).cast())), ones16),
            _mm256_madd_epi16(_mm256_maddubs_epi16(a2_hi, _mm256_loadu_si256(w3_0.add(64).cast())), ones16),
        ),
    );
    let s1 = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_maddubs_epi16(a1_vec, _mm256_loadu_si256(w3_1.add(0).cast())), ones16),
        _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_maddubs_epi16(a2_lo, _mm256_loadu_si256(w3_1.add(32).cast())), ones16),
            _mm256_madd_epi16(_mm256_maddubs_epi16(a2_hi, _mm256_loadu_si256(w3_1.add(64).cast())), ones16),
        ),
    );
    let s0_128 = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1));
    let s0_64 = _mm_add_epi32(s0_128, _mm_shuffle_epi32(s0_128, 0x4E));
    let out0 = st.fc3_biases[0] + _mm_cvtsi128_si32(_mm_add_epi32(s0_64, _mm_shuffle_epi32(s0_64, 0xB1)));

    let s1_128 = _mm_add_epi32(_mm256_castsi256_si128(s1), _mm256_extracti128_si256(s1, 1));
    let s1_64 = _mm_add_epi32(s1_128, _mm_shuffle_epi32(s1_128, 0x4E));
    let out1 = st.fc3_biases[1] + _mm_cvtsi128_si32(_mm_add_epi32(s1_64, _mm_shuffle_epi32(s1_64, 0xB1)));

    let positional = (((out0 as i64) + (skip as i64) + (quad_skip as i64)) * 600 / 16384) as i32;
    let psqt = (us.psqt[psqt_bucket] - them.psqt[psqt_bucket]) / 32;
    let draw_logit = (out1 * 100) / 16384;
    (psqt, positional, draw_logit)
}
