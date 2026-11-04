use asura::parameters::{is_spsa_active, set_spsa_active, tunables};
use asura::tuning::{Axis, Point};

#[derive(Clone, Copy)]
pub struct ParamSpec {
    pub name: &'static str,
    pub step: i64,
    pub min: i64,
    pub max: i64,
}

pub struct Block {
    pub name: &'static str,
    pub specs: [ParamSpec; 8],
}

pub fn blocks() -> Vec<Block> {
    let blocks: [(&str, [ParamSpec; 8]); 6] = [
        (
            "Block 1: ATN-1 Compensation, Shield & Tactical Ordering (Box 5 Only)",
            [
                ParamSpec { name: "HINDSIGHT_SAC_COMP_MIN", step: 10, min: 50, max: 180 },
                ParamSpec {
                    name: "HINDSIGHT_EXT_MIN_REDUCTION",
                    step: 200,
                    min: 1000,
                    max: 3500,
                },
                ParamSpec {
                    name: "SEE_KING_RING_SAC_SHIELD",
                    step: 20,
                    min: 60,
                    max: 350,
                },
                ParamSpec {
                    name: "QUIET_GREEDOID_UNBLOCK_BONUS",
                    step: 90,
                    min: 100,
                    max: 2000,
                },
                ParamSpec {
                    name: "QUIET_PHYSARUM_ARTERY_BONUS",
                    step: 90,
                    min: 100,
                    max: 2000,
                },
                ParamSpec {
                    name: "QUIET_OUTPOST_HOLE_BONUS",
                    step: 90,
                    min: 100,
                    max: 2200,
                },
                ParamSpec {
                    name: "QUIET_PETRI_OVERLOAD_BONUS",
                    step: 90,
                    min: 200,
                    max: 2200,
                },
                ParamSpec { name: "LMR_CAUSAL_RED_PENALTY", step: 35, min: 50, max: 600 },
            ],
        ),
        (
            "Block 2: Coupled LMR & Singular Extensions Core",
            [
                ParamSpec { name: "LMR_TABLE_SCALE", step: 14, min: 300, max: 560 },
                ParamSpec { name: "SE_BETA_MARGIN", step: 4, min: 10, max: 55 },
                ParamSpec { name: "LMR_CUT_BIAS", step: 90, min: 1200, max: 3000 },
                ParamSpec { name: "LMR_TT_PV_BONUS", step: 80, min: 500, max: 1800 },
                ParamSpec {
                    name: "LMR_NON_IMPROVING_PENALTY",
                    step: 80,
                    min: 500,
                    max: 1800,
                },
                ParamSpec {
                    name: "LMR_PRIORITY_QUIET_BONUS",
                    step: 80,
                    min: 400,
                    max: 1600,
                },
                ParamSpec { name: "LMR_CHECK_BONUS", step: 70, min: 300, max: 1400 },
                ParamSpec { name: "LMR_THREAT_BONUS", step: 70, min: 300, max: 1400 },
            ],
        ),
        (
            "Block 3: Forward Pruning (RFP, NMP, Razoring & Hindsight)",
            [
                ParamSpec { name: "RFP_DEPTH_MARGIN", step: 5, min: 40, max: 105 },
                ParamSpec { name: "RFP_IMPROVING_BONUS", step: 5, min: 20, max: 85 },
                ParamSpec { name: "RFP_NO_THREATS_BONUS", step: 5, min: 10, max: 75 },
                ParamSpec {
                    name: "HINDSIGHT_REDUCTION_MARGIN",
                    step: 8,
                    min: 45,
                    max: 160,
                },
                ParamSpec { name: "NMP_DEEP_FLOOR", step: 12, min: -200, max: -20 },
                ParamSpec { name: "NMP_TT_SEE_MARGIN", step: 20, min: 80, max: 350 },
                ParamSpec { name: "RAZOR_BASE", step: 18, min: 100, max: 340 },
                ParamSpec { name: "RAZOR_MARGIN", step: 18, min: 120, max: 350 },
            ],
        ),
        (
            "Block 4: ProbCut, Futility, SEE & History Pruning",
            [
                ParamSpec { name: "PROBCUT_BETA_MARGIN", step: 10, min: 80, max: 210 },
                ParamSpec { name: "PROBCUT_IMPROVING_BONUS", step: 6, min: 15, max: 85 },
                ParamSpec { name: "FP_BASE_MARGIN", step: 10, min: 60, max: 200 },
                ParamSpec { name: "FP_DEPTH_MARGIN", step: 8, min: 55, max: 165 },
                ParamSpec { name: "SEE_QUIET_QUAD_COEFF", step: 3, min: -40, max: -6 },
                ParamSpec { name: "SEE_NOISY_DEPTH_COEFF", step: 8, min: -160, max: -40 },
                ParamSpec { name: "HP_IMPROVING_OFFSET", step: 20, min: -350, max: -80 },
                ParamSpec { name: "SE_DOUBLE_GAP", step: 4, min: 10, max: 55 },
            ],
        ),
        (
            "Block 5: Move Ordering & Continuation History Weights",
            [
                ParamSpec { name: "QUIET_HIST_WEIGHT", step: 100, min: 800, max: 2400 },
                ParamSpec { name: "PAWN_HIST_WEIGHT", step: 90, min: 500, max: 1800 },
                ParamSpec { name: "CONT_HIST_1_WEIGHT", step: 100, min: 800, max: 2400 },
                ParamSpec { name: "CONT_HIST_2_WEIGHT", step: 90, min: 500, max: 1800 },
                ParamSpec {
                    name: "QUIET_THREATENED_DEST_PENALTY",
                    step: 500,
                    min: 4000,
                    max: 12000,
                },
                ParamSpec {
                    name: "QUIET_CREATE_THREAT_BONUS",
                    step: 300,
                    min: 1500,
                    max: 6500,
                },
                ParamSpec {
                    name: "QUIET_WALL_PAWN_PENALTY",
                    step: 350,
                    min: 1500,
                    max: 7000,
                },
                ParamSpec {
                    name: "QUIET_PETRI_OVERLOAD_BONUS",
                    step: 90,
                    min: 200,
                    max: 2000,
                },
            ],
        ),
        (
            "Block 6: Quiescence Pruning & Late-Quiet Closure",
            [
                ParamSpec { name: "QS_FUTILITY_MARGIN", step: 15, min: 100, max: 320 },
                ParamSpec { name: "QS_SEE_FUTILITY_MARGIN", step: 20, min: 150, max: 450 },
                ParamSpec { name: "QS_EARLY_CUTOFF_MARGIN", step: 18, min: 120, max: 350 },
                ParamSpec { name: "QS_SEE_BASE", step: 8, min: 30, max: 130 },
                ParamSpec {
                    name: "QS_QUIET_HIST_THRESHOLD",
                    step: 40,
                    min: -600,
                    max: -100,
                },
                ParamSpec { name: "LATE_QUIET_BASE_MARGIN", step: 6, min: 20, max: 90 },
                ParamSpec { name: "LATE_QUIET_DEPTH_MARGIN", step: 6, min: 40, max: 120 },
                ParamSpec {
                    name: "DEFER_QUIET_THRESHOLD",
                    step: 150,
                    min: -2500,
                    max: -300,
                },
            ],
        ),
    ];
    // A shared parameter has one feasible domain, even when it occurs in two
    // blocks. In particular, do not move it outside a later block's bounds.
    let all_specs: Vec<_> = blocks.iter().flat_map(|(_, specs)| specs.iter().copied()).collect();
    blocks
        .into_iter()
        .map(|(name, mut specs)| {
            for spec in &mut specs {
                for other in all_specs.iter().filter(|other| other.name == spec.name) {
                    spec.min = spec.min.max(other.min);
                    spec.max = spec.max.min(other.max);
                }
            }
            Block { name, specs }
        })
        .collect()
}

impl Block {
    pub fn values(&self) -> Result<Point, String> {
        let mut point = [0; 8];
        for (value, spec) in point.iter_mut().zip(&self.specs) {
            *value = tunables::get_by_name(spec.name).ok_or_else(|| format!("unknown parameter {}", spec.name))?;
        }
        Ok(point)
    }

    pub fn axes(&self, pass: usize) -> Result<[Axis; 8], String> {
        let values = self.values()?;
        Ok(std::array::from_fn(|i| {
            let spec = self.specs[i];
            let divisor = 1i128 << pass.min(62);
            let step = ((i128::from(spec.step) + divisor - 1) / divisor).max(1) as i64;
            Axis { value: values[i], min: spec.min, max: spec.max, step }
        }))
    }

    pub fn apply(&self, point: &Point) -> Result<(), String> {
        for (spec, &value) in self.specs.iter().zip(point) {
            if value < spec.min || value > spec.max || tunables::get_by_name(spec.name).is_none() {
                return Err(format!("invalid value or name for {}", spec.name));
            }
        }
        for (spec, &value) in self.specs.iter().zip(point) {
            assert!(tunables::set_by_name(spec.name, value));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot(pub Vec<(&'static str, i64)>);

impl Snapshot {
    pub fn capture(blocks: &[Block]) -> Result<Self, String> {
        let mut values = Vec::new();
        for spec in blocks.iter().flat_map(|block| &block.specs) {
            if !values.iter().any(|&(name, _)| name == spec.name) {
                let value =
                    tunables::get_by_name(spec.name).ok_or_else(|| format!("unknown parameter {}", spec.name))?;
                values.push((spec.name, value));
            }
        }
        Ok(Self(values))
    }

    pub fn apply(&self) {
        // Names were validated against the immutable registry at capture time.
        for &(name, value) in &self.0 {
            assert!(tunables::set_by_name(name, value));
        }
    }
}

/// Declare this before the evaluator so workers are joined before restoration.
pub struct Restore {
    pub before: Snapshot,
    active: bool,
}

impl Restore {
    pub fn enter(blocks: &[Block]) -> Result<Self, String> {
        let before = Snapshot::capture(blocks)?;
        let active = is_spsa_active();
        set_spsa_active(true);
        Ok(Self { before, active })
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        self.before.apply();
        set_spsa_active(self.active);
    }
}
