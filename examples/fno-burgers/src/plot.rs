//! Figure layout follows the neighboring DeepONet example: no font backend is
//! needed; the README supplies the panel meanings, axes and colour key.

use plotters::prelude::*;
use std::error::Error;

pub(crate) fn write_prediction_png(
    path: &str,
    n: usize,
    initial: &[f64],
    reference: &[f64],
    prediction: &[f64],
) -> Result<(), Box<dyn Error>> {
    if n < 2
        || initial.is_empty()
        || !initial.len().is_multiple_of(n)
        || reference.len() != initial.len()
        || prediction.len() != initial.len()
        || initial
            .iter()
            .chain(reference)
            .chain(prediction)
            .any(|v| !v.is_finite())
    {
        return Err("figure needs matching finite [n, samples, 1] grids with n >= 2".into());
    }
    let count = (initial.len() / n).min(4);
    let bounds = |values: &[f64]| {
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pad = (0.08 * (hi - lo)).max(1e-6);
        (lo - pad, hi + pad)
    };
    let (u_lo, u_hi) = bounds(&initial[..count * n]);
    let (r_lo, r_hi) = bounds(&reference[..count * n]);
    let (p_lo, p_hi) = bounds(&prediction[..count * n]);
    let root = BitMapBackend::new(path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 4));
    for sample in 0..count {
        let start = sample * n;
        let mut top = ChartBuilder::on(&panels[sample])
            .margin(12)
            .build_cartesian_2d(0.0..1.0, u_lo..u_hi)?;
        top.configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .max_light_lines(0)
            .disable_axes()
            .draw()?;
        top.draw_series(LineSeries::new(
            (0..n).map(|j| (j as f64 / n as f64, initial[start + j])),
            BLUE.stroke_width(2),
        ))?;
        let mut bottom = ChartBuilder::on(&panels[4 + sample])
            .margin(12)
            .build_cartesian_2d(0.0..1.0, r_lo.min(p_lo)..r_hi.max(p_hi))?;
        bottom
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .max_light_lines(0)
            .disable_axes()
            .draw()?;
        bottom.draw_series(LineSeries::new(
            (0..n).map(|j| (j as f64 / n as f64, reference[start + j])),
            BLUE.stroke_width(3),
        ))?;
        bottom.draw_series(LineSeries::new(
            (0..n).map(|j| (j as f64 / n as f64, prediction[start + j])),
            RED.stroke_width(2),
        ))?;
    }
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests;
