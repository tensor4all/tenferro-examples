//! Figure for the neural ODE: the two trajectory components and the training
//! loss, using the `plotters` crate.
//!
//! The plots draw no text: `plotters` is built without a font backend, because
//! drawing a caption or axis label without one panics (#1968). The README states
//! the axis ranges and the colours. In the two trajectory panels the blue curve
//! is the reference and the red curve the prediction; the right panel is the
//! training loss on a logarithmic y-axis.

use plotters::prelude::*;
use std::error::Error;

/// Padded `(low, high)` bounds of the finite values produced by `values`.
///
/// An empty or entirely non-finite sequence falls back to `(-1, 1)`, and a
/// constant series is padded by 1, so a chart can always be built.
pub(crate) fn linear_bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            low = low.min(value);
            high = high.max(value);
        }
    }
    if !low.is_finite() || !high.is_finite() {
        return (-1.0, 1.0);
    }
    if (high - low).abs() < f64::EPSILON {
        return (low - 1.0, high + 1.0);
    }
    let pad = 0.05 * (high - low);
    (low - pad, high + pad)
}

/// Multiplicative `(low, high)` bounds for the log-scale loss panel.
///
/// Non-positive and non-finite entries cannot be drawn on a log axis and are
/// ignored; a history without any positive value falls back to `(1e-8, 1)`.
pub(crate) fn log_bounds(history: &[f64]) -> (f64, f64) {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for &value in history {
        if value.is_finite() && value > 0.0 {
            low = low.min(value);
            high = high.max(value);
        }
    }
    if !low.is_finite() || !high.is_finite() {
        return (1e-8, 1.0);
    }
    (low / 1.5, high * 1.5)
}

/// Write the 1x3 figure: `z1(t)`, `z2(t)` and the training loss.
///
/// `times`, `truth` and `pred` describe the whole trajectory (`truth` and
/// `pred` column-major as `[times.len(), 2]`); `loss` holds one value per
/// plotted block (the caller decides the block size).
pub(crate) fn write_figure_png(
    path: &str,
    times: &[f64],
    truth: &[f64],
    pred: &[f64],
    loss: &[f64],
) -> Result<(), Box<dyn Error>> {
    const WIDTH: u32 = 1400;
    const HEIGHT: u32 = 420;
    const COLUMNS: usize = 3;

    let n = times.len();
    assert!(
        truth.len() >= 2 * n && pred.len() >= 2 * n,
        "short trajectory"
    );
    let x_hi = times.last().copied().unwrap_or(1.0);

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, COLUMNS));

    for (component, panel) in panels.iter().enumerate().take(2) {
        let offset = component * n;
        let (low, high) = linear_bounds((0..n).flat_map(|i| [truth[i + offset], pred[i + offset]]));
        let mut chart = ChartBuilder::on(panel)
            .margin(8)
            .build_cartesian_2d(0.0..x_hi, low..high)?;
        chart
            .configure_mesh()
            .x_labels(6)
            .y_labels(5)
            .disable_axes()
            .draw()?;
        chart.draw_series(LineSeries::new(
            times
                .iter()
                .enumerate()
                .map(|(i, &t)| (t, truth[i + offset])),
            BLUE.stroke_width(2),
        ))?;
        chart.draw_series(LineSeries::new(
            times
                .iter()
                .enumerate()
                .map(|(i, &t)| (t, pred[i + offset])),
            RED.stroke_width(2),
        ))?;
    }

    let (low, high) = log_bounds(loss);
    let mut chart = ChartBuilder::on(&panels[2])
        .margin(8)
        .build_cartesian_2d(0.0..loss.len().max(1) as f64, (low..high).log_scale())?;
    chart
        .configure_mesh()
        .x_labels(6)
        .y_labels(5)
        .disable_axes()
        .draw()?;
    chart.draw_series(LineSeries::new(
        loss.iter().enumerate().map(|(i, &value)| {
            let y = if value.is_finite() && value > 0.0 {
                value
            } else {
                low
            };
            (i as f64, y)
        }),
        RED.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests;
