//! Frozen estimator semantics for diagnostics/benchmarks ONLY, never tuning.
//! Columns, normalizers, interaction correction and pseudo-SE filter match
//! 6f5c355. The Walsh parity formula equals its Sylvester construction; Lean
//! checks every entry. Endpoints use safe widening, and callers supply one
//! deterministic objective instead of the old unordered floating reduction.

use asura::tuning::{Axis, Point};

pub struct Report {
    pub gradient: [f64; 8],
    pub pseudo_se: [f64; 8],
    pub interactions: [f64; 4],
    pub trials: Vec<Point>,
    pub calls: usize,
}

pub fn probe(
    axes: &[Axis; 8], baseline: f64, mut evaluate: impl FnMut(&Point) -> Result<f64, String>,
) -> Result<Report, String> {
    if !baseline.is_finite() || axes.iter().any(|a| a.step <= 0 || a.min > a.value || a.value > a.max) {
        return Err("invalid legacy diagnostic input".into());
    }
    let columns = [1u32, 2, 4, 8, 5, 10, 6, 11];
    let base = axes.map(|a| a.value);
    let mut gradient = [0.0f64; 8];
    let mut second = [0.0f64; 8];
    let mut interactions = [0.0f64; 4];
    for row in 0u32..16 {
        let signs: [i128; 8] = columns.map(|col| if (row & col).count_ones() % 2 == 0 { 1 } else { -1 });
        let point = |side: i128| {
            std::array::from_fn(|i| {
                (i128::from(base[i]) + side * signs[i] * i128::from(axes[i].step))
                    .clamp(i128::from(axes[i].min), i128::from(axes[i].max)) as i64
            })
        };
        let plus = evaluate(&point(1))?;
        let minus = evaluate(&point(-1))?;
        if !plus.is_finite() || !minus.is_finite() {
            return Err("nonfinite legacy diagnostic observation".into());
        }
        let difference = plus - minus;
        for i in 0..8 {
            let value = difference * signs[i] as f64 / 2.0;
            gradient[i] += value / 16.0;
            second[i] += value * value / 16.0;
        }
        for p in 0..4 {
            interactions[p] += (plus + minus - 2.0 * baseline) * (signs[2 * p] * signs[2 * p + 1]) as f64 / 32.0;
        }
    }
    let pseudo_se = std::array::from_fn(|i| ((second[i] - gradient[i] * gradient[i]).max(1e-9) / 16.0).sqrt());
    let mut adjusted = [0.0f64; 8];
    for i in 0..8 {
        let raw = gradient[i] + 0.15 * interactions[i / 2].clamp(-2.0, 2.0) * gradient[i ^ 1];
        if raw.abs() * 2.0 >= 3.0 * pseudo_se[i] {
            adjusted[i] = raw;
        }
    }
    if gradient.iter().chain(&pseudo_se).chain(&interactions).chain(&adjusted).any(|x| !x.is_finite()) {
        return Err("legacy diagnostic arithmetic overflow".into());
    }
    let largest = adjusted.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let mut trials = Vec::new();
    if largest > 1e-5 {
        for scale in [1.0, 0.5] {
            let point = std::array::from_fn(|i| {
                let change = (axes[i].step as f64 * scale * adjusted[i] / largest).round() as i128;
                (i128::from(base[i]) + change).clamp(i128::from(axes[i].min), i128::from(axes[i].max)) as i64
            });
            if point != base && !trials.contains(&point) {
                trials.push(point);
            }
        }
    }
    Ok(Report { gradient, pseudo_se, interactions, trials, calls: 32 })
}
