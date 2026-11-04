//! Bounded, reproducible optimizer comparisons, not chess-strength evidence.
//! All methods use the same objective-call cap and an observed-improvement gate.
//! Training and validation calls are charged separately, even on these analytic
//! fixtures where their functions coincide. The ordinary baseline uses two
//! Rademacher probes and a bounded normalized SPSA step, not a tuned SPSA schedule.

use super::legacy;
use asura::tuning::{Axis, Point, probe_block};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn signed(&mut self, radius: i64) -> i64 {
        (self.next() % (2 * radius + 1) as u64) as i64 - radius
    }
}

struct Problem {
    family: usize,
    target: Point,
    weights: [i64; 8],
    matrix: [[i64; 8]; 8],
}
impl Problem {
    fn new(family: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let target = std::array::from_fn(|i| {
            if family == 3 { if i < 3 { if rng.next() & 1 == 0 { 2 } else { -2 } } else { 0 } } else { rng.signed(3) }
        });
        let weights = std::array::from_fn(|_| 1 + (rng.next() % 8) as i64);
        let matrix = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 8 } else { rng.signed(2) }));
        Self { family, target, weights, matrix }
    }
    fn score(&self, x: &Point) -> i64 {
        let d: Point = std::array::from_fn(|i| x[i] - self.target[i]);
        match self.family {
            0 => -d.iter().enumerate().map(|(i, v)| self.weights[i] * v * v).sum::<i64>(),
            1 => -self.matrix.iter().map(|row| row.iter().zip(d).map(|(&a, b)| a * b).sum::<i64>().pow(2)).sum::<i64>(),
            2 => -d.iter().enumerate().map(|(i, v)| self.weights[i] * ((v.abs() + 1) / 2).pow(2)).sum::<i64>(),
            _ => {
                let on_target_side = (0..3).filter(|&i| x[i] * self.target[i] > 0).count();
                -d.iter().map(|v| v * v).sum::<i64>() - if on_target_side == 1 || on_target_side == 2 { 100 } else { 0 }
            }
        }
    }
}

fn optimize(problem: &Problem, design: usize, seed: u64, cap: usize) -> Result<(i64, usize, usize), String> {
    let mut point = [0; 8];
    let mut current = problem.score(&point);
    let mut calls = 1;
    let mut rounds = 0;
    let mut rng = Rng(seed);
    // Reserve a complete round, including all its validation proposals.
    let maximum_round = [34, 26, 4][design];
    while calls + maximum_round <= cap && rounds < cap {
        let axes = point.map(|value| Axis { value, min: -4, max: 4, step: 1 });
        let mut observe = |p: &Point| -> Result<i64, String> {
            if calls >= cap {
                return Err("synthetic query budget exceeded".into());
            }
            calls += 1;
            Ok(problem.score(p))
        };
        let trials = match design {
            0 => legacy::probe(&axes, current as f64, |p| Ok(observe(p)? as f64))?.trials,
            1 => probe_block(&axes, rounds, current, &mut observe)?.trials,
            _ => {
                let signs: Point = std::array::from_fn(|_| if rng.next() & 1 == 0 { -1 } else { 1 });
                let plus = std::array::from_fn(|i| (point[i] + signs[i]).clamp(-4, 4));
                let minus = std::array::from_fn(|i| (point[i] - signs[i]).clamp(-4, 4));
                let difference = observe(&plus)? - observe(&minus)?;
                if difference > 0 {
                    vec![plus]
                } else if difference < 0 {
                    vec![minus]
                } else {
                    Vec::new()
                }
            }
        };
        // All candidates are declared before validation. The true deterministic
        // analytic objective is identical on both panels in this benchmark only.
        for trial in trials {
            let value = observe(&trial)?;
            if value > current {
                point = trial;
                current = value;
            }
        }
        rounds += 1;
    }
    Ok((current, calls, rounds))
}

pub fn run(seed: u64, cap: usize, repeats: usize) -> Result<(), String> {
    let families = ["separable_quadratic", "dense_quadratic", "integer_plateau", "three_way_barrier"];
    let designs = ["legacy", "paired", "two_probe_spsa"];
    println!(
        "MATH_SCOPE\tquery_cap={cap}\tvalidation_calls_charged=true\tnoise=none\tglobal_optimum=0\tchess_strength=false"
    );
    for (family, name) in families.into_iter().enumerate() {
        for repeat in 0..repeats {
            let run_seed = seed.wrapping_add((repeat as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let problem = Problem::new(family, run_seed);
            for (design, method) in designs.into_iter().enumerate() {
                let (score, calls, rounds) = optimize(&problem, design, run_seed, cap)?;
                println!(
                    "MATH\tfamily={name}\trepeat={repeat}\tdesign={method}\tregret={}\tcalls={calls}\trounds={rounds}",
                    -score
                );
            }
        }
    }
    Ok(())
}
