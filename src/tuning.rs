//! Deterministic, bounded eight-parameter tuning proposals.
//!
//! The objective is supplied by the caller. Four corners identify an exact
//! bilinear interpolant on each two-coordinate face, not a Hessian of an
//! arbitrary objective. Proposals are not accepted updates or strength claims.

pub const WIDTH: usize = 8;
pub type Point = [i64; WIDTH];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axis {
    pub value: i64,
    pub min: i64,
    pub max: i64,
    pub step: i64,
}

impl Axis {
    fn endpoints(self, index: usize) -> Result<[i64; 2], String> {
        if self.min > self.max || self.value < self.min || self.value > self.max {
            return Err(format!("axis {index}: incumbent is outside its bounds"));
        }
        if self.step <= 0 {
            return Err(format!("axis {index}: step must be positive"));
        }
        let value = i128::from(self.value);
        let step = i128::from(self.step);
        Ok([(value - step).max(i128::from(self.min)) as i64, (value + step).min(i128::from(self.max)) as i64])
    }
}

/// Numerators of the four coefficients; their common denominator is four.
/// Widening before summing four i64 observations makes these operations exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Contrasts {
    pub mean4: i128,
    pub x4: i128,
    pub y4: i128,
    pub interaction4: i128,
}

impl Contrasts {
    pub fn from_corners([ll, lh, hl, hh]: [i64; 4]) -> Self {
        let [ll, lh, hl, hh] = [ll, lh, hl, hh].map(i128::from);
        Self {
            mean4: ll + lh + hl + hh,
            x4: hl + hh - ll - lh,
            y4: lh + hh - ll - hl,
            interaction4: ll + hh - lh - hl,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PairReport {
    pub indices: (usize, usize),
    pub low: [i64; 2],
    pub high: [i64; 2],
    /// Low/low, low/high, high/low, high/high. Degenerate faces repeat scores.
    pub scores: [i64; 4],
    pub contrasts: Contrasts,
    pub best: Point,
    pub best_score: i64,
}

#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub pairs: [PairReport; WIDTH / 2],
    /// At most ten distinct proposals, all fixed before validation observations.
    /// The two global exploration trials are unscored, not free validation calls.
    pub trials: Vec<Point>,
    /// Distinct observed points, including the caller-supplied incumbent first.
    pub observations: Vec<(Point, i64)>,
    /// Additional objective calls. At most sixteen; no statistical claim.
    pub calls: usize,
}

/// Seven rounds cover all 28 unordered coordinate pairs exactly once.
/// The first round retains the historical four within-block pairings.
pub fn pairing(round: usize) -> [(usize, usize); WIDTH / 2] {
    let mut order = [0, 2, 4, 6, 7, 5, 3, 1];
    for _ in 0..round % (WIDTH - 1) {
        order[1..].rotate_right(1);
    }
    std::array::from_fn(|i| (order[i], order[WIDTH - 1 - i]))
}

/// Probe one pair partition of an immutable incumbent.
///
/// The caller must supply observations for one fixed objective/panel. Cached
/// observations are reused, not treated as independent noise samples. The
/// incumbent score must belong to that same objective. Other coordinates stay
/// at the incumbent while a face is probed. Integer endpoints can be asymmetric
/// near a bound, so contrasts are interpreted on the actual face.
pub fn probe_block(
    axes: &[Axis; WIDTH], round: usize, incumbent_score: i64, mut evaluate: impl FnMut(&Point) -> Result<i64, String>,
) -> Result<ProbeReport, String> {
    let incumbent = axes.map(|axis| axis.value);
    let mut endpoints = [[0; 2]; WIDTH];
    for i in 0..WIDTH {
        endpoints[i] = axes[i].endpoints(i)?;
    }
    let mut observations = vec![(incumbent, incumbent_score)];
    let mut pairs = Vec::with_capacity(WIDTH / 2);
    for (x, y) in pairing(round) {
        let mut scores = [0; 4];
        let mut best = incumbent;
        let mut best_score = incumbent_score;
        for (corner, score) in scores.iter_mut().enumerate() {
            let mut point = incumbent;
            point[x] = endpoints[x][corner >> 1];
            point[y] = endpoints[y][corner & 1];
            *score = if let Some((_, value)) = observations.iter().find(|(seen, _)| seen == &point) {
                *value
            } else {
                let value = evaluate(&point)?;
                observations.push((point, value));
                value
            };
            if *score > best_score {
                best = point;
                best_score = *score;
            }
        }
        pairs.push(PairReport {
            indices: (x, y),
            low: [endpoints[x][0], endpoints[y][0]],
            high: [endpoints[x][1], endpoints[y][1]],
            scores,
            contrasts: Contrasts::from_corners(scores),
            best,
            best_score,
        });
    }
    let pairs: [PairReport; WIDTH / 2] = pairs.try_into().expect("one report per fixed pair");
    let mut joint = incumbent;
    for pair in &pairs {
        let (x, y) = pair.indices;
        joint[x] = pair.best[x];
        joint[y] = pair.best[y];
    }
    // Keep a gradient-directed alternative even when every pair corner loses:
    // diagonal curvature on one axis can conceal progress along its partner.
    // These are face contrasts, not statistical signal-to-noise estimates.
    let mut contrast = [0i128; WIDTH];
    for pair in &pairs {
        let (x, y) = pair.indices;
        contrast[x] = pair.contrasts.x4;
        contrast[y] = pair.contrasts.y4;
    }
    let targets: Point = std::array::from_fn(|i| {
        if contrast[i] > 0 {
            endpoints[i][1]
        } else if contrast[i] < 0 {
            endpoints[i][0]
        } else {
            incumbent[i]
        }
    });
    // Outward/immobile axes must not set the normalization for movable axes.
    let magnitude: [i128; WIDTH] =
        std::array::from_fn(|i| if targets[i] == incumbent[i] { 0 } else { contrast[i].abs() });
    let largest = magnitude.into_iter().max().unwrap_or(0);
    let directed = std::array::from_fn(|i| {
        let base = i128::from(incumbent[i]);
        if largest == 0 {
            return incumbent[i];
        }
        // Two bounded products, rather than distance*contrast (which can exceed
        // i128 at i64 extremes). 1024 is a proposal discretization, not a
        // confidence level or a new search/evaluation scale.
        let fraction = magnitude[i] * 1024 / largest;
        (base + (i128::from(targets[i]) - base) * fraction / 1024) as i64
    });
    let halfway = |point: Point| {
        std::array::from_fn(|i| {
            let base = i128::from(incumbent[i]);
            (base + (i128::from(point[i]) - base) / 2) as i64
        })
    };
    // Pair-local probes cannot see every higher-order barrier. Predeclare two
    // antipodal global trials as bounded exploration, never as known improvements.
    // Odd multiplier 73 permutes the 128 seven-bit masks; the antipode supplies
    // the eighth sign. This is a deterministic schedule, not a noise sample.
    let mask = ((round % 128) * 73) % 128;
    let exploration: [Point; 2] =
        std::array::from_fn(|side| std::array::from_fn(|i| endpoints[i][usize::from(mask & (1 << i) == 0) ^ side]));
    let mut trials = Vec::with_capacity(6 + WIDTH / 2);
    for point in [directed, halfway(directed), joint, halfway(joint)]
        .into_iter()
        .chain(pairs.iter().map(|pair| pair.best))
        .chain(exploration)
    {
        if point != incumbent && !trials.contains(&point) {
            trials.push(point);
        }
    }
    Ok(ProbeReport { pairs, trials, calls: observations.len() - 1, observations })
}

/// Exact totals on one explicitly identified, nonempty finite panel.
/// The caller owns panel identity; this is not an estimate of population Elo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelMeasurement {
    pub panel_id: u64,
    pub cases: usize,
    pub utility: i128,
    pub protected: Vec<i128>,
}

/// Select the best *observed* eligible candidate, or retain the incumbent.
///
/// Every candidate and the fixed constraint anchor must use the same panel as
/// the incumbent. Allowed drops are nonnegative tolerances relative to that
/// anchor, not a budget that can be spent again after every accepted update.
/// Ties retain the incumbent, then the earlier candidate. Candidate generation
/// must precede this panel's observations; no confidence/generalization claim is
/// made even when all these finite-panel checks pass.
pub fn select_on_panel(
    incumbent: &PanelMeasurement, candidates: &[PanelMeasurement], constraint_anchor: &PanelMeasurement,
    allowed_drops: &[i128],
) -> Result<Option<usize>, String> {
    if incumbent.cases == 0 {
        return Err("validation panel must not be empty".into());
    }
    if incumbent.protected.len() != allowed_drops.len() || allowed_drops.iter().any(|&drop| drop < 0) {
        return Err("protected metrics require matching nonnegative tolerances".into());
    }
    if constraint_anchor.panel_id != incumbent.panel_id
        || constraint_anchor.cases != incumbent.cases
        || constraint_anchor.protected.len() != allowed_drops.len()
    {
        return Err("constraint anchor and incumbent panels do not match".into());
    }
    let eligible = |measurement: &PanelMeasurement| {
        measurement
            .protected
            .iter()
            .zip(&constraint_anchor.protected)
            .zip(allowed_drops)
            .all(|((&value, &base), &drop)| value >= base || base.abs_diff(value) <= drop as u128)
    };
    if !eligible(incumbent) {
        return Err("incumbent violates the fixed protected-metric limits".into());
    }
    let mut selected = None;
    let mut best = incumbent.utility;
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.panel_id != incumbent.panel_id
            || candidate.cases != incumbent.cases
            || candidate.protected.len() != allowed_drops.len()
        {
            return Err("candidate and incumbent validation panels do not match".into());
        }
        if eligible(candidate) && candidate.utility > best {
            selected = Some(index);
            best = candidate.utility;
        }
    }
    Ok(selected)
}
