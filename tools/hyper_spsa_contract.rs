//! Thin adapter to the compiled PUBLIC production optimizer, not an extracted model.
//! Link with `--extern production=<fresh release rlib> --edition=2024
//! -C target-cpu=native -C panic=abort`. See check_hyper_spsa.py for the protocol.
//! No NNUE, search, training, matches, or parameter writes are performed.

use production::tuning::{Axis, Contrasts, PanelMeasurement, Point, ProbeReport, WIDTH, probe_block, select_on_panel};
use std::io::{self, BufRead, Write};

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn decimal(s: &str) -> Result<i128, String> {
    let digits = s.strip_prefix('-').unwrap_or(s);
    require(!digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit()), "invalid decimal")?;
    s.parse().map_err(|_| "integer outside i128".into())
}

fn i64s(xs: &[i128]) -> Result<Vec<i64>, String> {
    xs.iter().map(|&x| i64::try_from(x).map_err(|_| "integer outside i64".into())).collect()
}

fn words(tag: &str, xs: impl IntoIterator<Item = i128>) -> String {
    let mut out = tag.to_string();
    for x in xs {
        out.push(' ');
        out.push_str(&x.to_string());
    }
    out
}

fn contrasts(c: Contrasts) -> [i128; 4] {
    [c.mean4, c.x4, c.y4, c.interaction4]
}

fn contrast_record(xs: &[i128]) -> Result<String, String> {
    require(xs.len() == 4, "C shape")?;
    let corners: [i64; 4] = i64s(xs)?.try_into().map_err(|_| "C shape")?;
    Ok(words("C", contrasts(Contrasts::from_corners(corners))))
}

fn bound_record(xs: &[i128]) -> Result<String, String> {
    require(xs.len() == 5, "B shape")?;
    let x = i64s(xs)?;
    require((-1..=1).contains(&x[4]), "B direction")?;
    let mut axes = [Axis { value: 0, min: 0, max: 0, step: 1 }; WIDTH];
    axes[0] = Axis { min: x[0], max: x[1], value: x[2], step: x[3] };
    let mut actual = Vec::new();
    let result = probe_block(&axes, 0, 0, |p| {
        require(p[1..].iter().all(|&v| v == 0), "B moved fixed coordinate")?;
        let score = (i128::from(x[4]) * (i128::from(p[0]) - i128::from(x[2]))).signum() as i64;
        actual.push((*p, score));
        Ok(score)
    });
    if !(x[0] <= x[2] && x[2] <= x[1] && x[3] > 0) {
        require(result.is_err() && actual.is_empty(), "B invalid axis accepted or evaluated")?;
        return Ok("B ERR".into());
    }
    let report = result?;
    require(report.calls == actual.len() && report.observations[1..] == actual, "B call accounting")?;
    require(report.trials.iter().all(|p| p[1..].iter().all(|&v| v == 0)), "B trial moved fixed axis")?;
    let mut out = vec![
        i128::from(report.pairs[0].low[0]),
        i128::from(report.pairs[0].high[0]),
        report.calls as i128,
        report.trials.len() as i128,
    ];
    out.extend(report.trials.iter().map(|p| i128::from(p[0])));
    Ok(words("B", out))
}

fn selector_record(xs: &[i128]) -> Result<String, String> {
    require(xs.len() >= 5, "S shape")?;
    let n = xs[4];
    require((0..=64).contains(&n) && xs.len() == 5 + 2 * n as usize, "S candidate count")?;
    let measurement = |utility, metric| PanelMeasurement { panel_id: 17, cases: 1, utility, protected: vec![metric] };
    let anchor = measurement(0, xs[0]);
    let incumbent = measurement(xs[1], xs[2]);
    let candidates: Vec<_> = (0..n as usize).map(|i| measurement(xs[5 + 2 * i], xs[6 + 2 * i])).collect();
    Ok(match select_on_panel(&incumbent, &candidates, &anchor, &[xs[3]]) {
        Ok(Some(index)) => format!("S {index}"),
        Ok(None) => "S -1".into(),
        Err(_) => "S ERR".into(),
    })
}

fn report_words(report: &ProbeReport) -> Vec<i128> {
    let mut out = vec![report.calls as i128];
    for pair in &report.pairs {
        out.extend([pair.indices.0 as i128, pair.indices.1 as i128]);
        out.extend(pair.low.map(i128::from));
        out.extend(pair.high.map(i128::from));
        out.extend(pair.scores.map(i128::from));
        out.extend(contrasts(pair.contrasts));
        out.extend(pair.best.map(i128::from));
        out.push(i128::from(pair.best_score));
    }
    out.push(report.observations.len() as i128);
    for (point, score) in &report.observations {
        out.extend(point.map(i128::from));
        out.push(i128::from(*score));
    }
    out.push(report.trials.len() as i128);
    for point in &report.trials {
        out.extend(point.map(i128::from));
    }
    out
}

fn quadratic_record(xs: &[i128]) -> Result<String, String> {
    require(xs.len() == 106, "Q shape")?;
    require((0..=895).contains(&xs[0]), "Q round")?;
    require((-1000..=1000).contains(&xs[1]) && xs[34..].iter().all(|v| (-1000..=1000).contains(v)), "Q coefficients")?;
    for i in 0..WIDTH {
        let a = &xs[2 + 4 * i..6 + 4 * i];
        require(-32 <= a[0] && a[0] <= a[2] && a[2] <= a[1] && a[1] <= 32 && (1..=64).contains(&a[3]), "Q axes")?;
    }
    let axes = std::array::from_fn(|i| Axis {
        min: xs[2 + 4 * i] as i64,
        max: xs[3 + 4 * i] as i64,
        value: xs[4 + 4 * i] as i64,
        step: xs[5 + 4 * i] as i64,
    });
    // Fixture objective only. The optimizer, including its private directed and
    // half-step construction, is called through probe_block, never reimplemented.
    let score = |point: &Point| -> Result<i64, String> {
        let mut total = xs[1];
        for i in 0..WIDTH {
            total += xs[34 + i] * i128::from(point[i]);
            for j in 0..WIDTH {
                total += xs[42 + 8 * i + j] * i128::from(point[i]) * i128::from(point[j]);
            }
        }
        i64::try_from(total).map_err(|_| "Q fixture score overflow".into())
    };
    let base = axes.map(|a| a.value);
    let mut actual = Vec::new();
    let report = probe_block(&axes, xs[0] as usize, score(&base)?, |p| {
        let value = score(p)?;
        actual.push((*p, value));
        Ok(value)
    })?;
    require(report.calls == actual.len() && report.observations[1..] == actual, "Q call accounting")?;
    require(report.calls <= 16 && report.trials.len() <= 10, "Q query/trial bound")?;
    Ok(words("Q", report_words(&report)))
}

fn respond(line: &str) -> Result<String, String> {
    require(line.len() <= 16384, "line too long")?;
    let mut tokens = line.split(' ').filter(|s| !s.is_empty());
    let tag = tokens.next().ok_or("empty record")?;
    // Collect Result, not filter_map: malformed fields must stop the run.
    let xs = tokens.map(decimal).collect::<Result<Vec<_>, _>>()?;
    match tag {
        "C" => contrast_record(&xs),
        "B" => bound_record(&xs),
        "S" => selector_record(&xs),
        "Q" => quadratic_record(&xs),
        _ => Err("unknown record".into()),
    }
}

fn run() -> Result<(), String> {
    let mut input = io::stdin().lock();
    let mut output = io::BufWriter::new(io::stdout().lock());
    let mut count = 0;
    loop {
        let mut bytes = Vec::new();
        // Limit allocation even for an unterminated malicious record.
        let size = io::Read::take(&mut input, 16386).read_until(b'\n', &mut bytes).map_err(|e| e.to_string())?;
        if size == 0 {
            break;
        }
        require(size <= 16385, "line too long")?;
        count += 1;
        require(count <= 8192, "too many records")?;
        let line = std::str::from_utf8(&bytes).map_err(|_| "non-UTF8 record")?.trim_ascii();
        let response = respond(line).map_err(|e| format!("record {count}: {e}"))?;
        writeln!(output, "{response}").map_err(|e| e.to_string())?;
    }
    require(count != 0, "empty fixture stream")?;
    output.flush().map_err(|e| e.to_string())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
