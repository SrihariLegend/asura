use crate::nnue::{AccumulatorSide, L1_HALF, L1_SIZE, L2_SIZE, L3_SIZE, ModelParameters};

pub fn forward_scalar(
    us: &AccumulatorSide, them: &AccumulatorSide, pk_beta: &[i32; L2_SIZE], pk_gamma: &[i32; L2_SIZE],
    psqt_bucket: usize, stack_bucket: usize, params: &ModelParameters,
) -> (i32, i32, i32) {
    let mut l0 = [0u8; L1_SIZE];
    for (side_idx, acc) in [us, them].iter().enumerate() {
        let base = side_idx * L1_HALF;
        for i in 0..L1_HALF {
            let l = acc.values[i].clamp(0, 255) as i32;
            let r = acc.values[i + L1_HALF].clamp(0, 255) as i32;
            l0[base + i] = ((l * r) >> 9) as u8;
        }
    }

    let st = &params.stacks[stack_bucket];
    let mut x1 = st.fc1_biases;
    let mut quad_skip = 0i32;

    for g in 0..(L1_SIZE / 4) {
        let b0 = l0[g * 4] as i32;
        let b1 = l0[g * 4 + 1] as i32;
        let b2 = l0[g * 4 + 2] as i32;
        let b3 = l0[g * 4 + 3] as i32;
        if (b0 | b1 | b2 | b3) == 0 {
            continue;
        }
        quad_skip += b0 * (st.quad_skip_weights[g * 4] as i32)
            + b1 * (st.quad_skip_weights[g * 4 + 1] as i32)
            + b2 * (st.quad_skip_weights[g * 4 + 2] as i32)
            + b3 * (st.quad_skip_weights[g * 4 + 3] as i32);

        let w_base = g * L2_SIZE * 4;
        for j in 0..L2_SIZE {
            let off = w_base + j * 4;
            x1[j] += b0 * (st.fc1_weights[off] as i32)
                + b1 * (st.fc1_weights[off + 1] as i32)
                + b2 * (st.fc1_weights[off + 2] as i32)
                + b3 * (st.fc1_weights[off + 3] as i32);
        }
    }

    for j in 0..L2_SIZE {
        let h = x1[j];
        x1[j] = h + pk_beta[j] + pk_gamma[j] * (h >> 7);
    }

    let skip = x1[L2_SIZE - 2] - x1[L2_SIZE - 1];

    let mut a1 = [0u8; L2_SIZE * 2];
    for j in 0..L2_SIZE {
        let v = x1[j].max(0);
        let sq = ((v as i64 * v as i64) >> 19).min(127) as u8;
        let lin = (v >> 6).min(127) as u8;
        a1[j] = sq;
        a1[L2_SIZE + j] = lin;
    }

    let mut x2 = st.fc2_biases;
    for j in 0..L3_SIZE {
        let row_off = j * (L2_SIZE * 2);
        let mut s = x2[j];
        for i in 0..(L2_SIZE * 2) {
            s += (a1[i] as i32) * (st.fc2_weights[row_off + i] as i32);
        }
        x2[j] = s;
    }

    let mut a2 = [0u8; L3_SIZE * 2];
    for j in 0..L3_SIZE {
        let v = x2[j].max(0);
        let sq = ((v as i64 * v as i64) >> 19).min(127) as u8;
        let lin = (v >> 6).min(127) as u8;
        a2[j] = sq;
        a2[L3_SIZE + j] = lin;
    }

    let mut out0 = st.fc3_biases[0];
    let mut out1 = st.fc3_biases[1];
    for i in 0..(L2_SIZE * 2) {
        out0 += (a1[i] as i32) * (st.fc3_weights[0][i] as i32);
        out1 += (a1[i] as i32) * (st.fc3_weights[1][i] as i32);
    }
    for i in 0..(L3_SIZE * 2) {
        out0 += (a2[i] as i32) * (st.fc3_weights[0][L2_SIZE * 2 + i] as i32);
        out1 += (a2[i] as i32) * (st.fc3_weights[1][L2_SIZE * 2 + i] as i32);
    }

    let positional = (((out0 as i64) + (skip as i64) + (quad_skip as i64)) * 600 / 16384) as i32;
    let psqt = (us.psqt[psqt_bucket] - them.psqt[psqt_bucket]) / 32;
    let draw_logit = (out1 * 100) / 16384;
    (psqt, positional, draw_logit)
}
