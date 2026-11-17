//! The loss figure: cumulative wall time on the x axis, the training loss on a
//! logarithmic y axis, one polyline per optimizer.
//!
//! The figure draws no text: `plotters` is built without a font backend, and
//! drawing a caption or axis label without one panics (the same restriction as
//! the neural-ODE example). The README states the axis ranges and the colours.

use plotters::prelude::*;
use plotters::style::RGBColor;
use std::error::Error;

#[cfg(test)]
mod tests;

const WIDTH: u32 = 900;
const HEIGHT: u32 = 500;

/// Padded `(low, high)` bounds of the finite values produced by `values`.
///
/// An empty or entirely non-finite sequence falls back to `(0, 1)`, and a
/// constant series is padded by 1, so a chart can always be built.
pub(crate) fn linear_bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values.filter(|value| value.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    if !low.is_finite() || !high.is_finite() {
        return (0.0, 1.0);
    }
    if (high - low).abs() < f64::EPSILON {
        return (low - 1.0, high + 1.0);
    }
    let pad = 0.05 * (high - low);
    (low - pad, high + pad)
}

/// Multiplicative `(low, high)` bounds for the log-scale loss axis.
///
/// Non-positive and non-finite values cannot be drawn on a log axis and are
/// ignored; a history without any positive value falls back to `(1e-8, 1)`.
pub(crate) fn log_bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values.filter(|value| value.is_finite() && *value > 0.0) {
        low = low.min(value);
        high = high.max(value);
    }
    if !low.is_finite() || !high.is_finite() {
        return (1e-8, 1.0);
    }
    (low / 1.5, high * 1.5)
}

/// Write the loss curves, each as `(colour, (seconds, loss) samples)`.
pub(crate) fn write_loss_png(
    path: &str,
    curves: &[(RGBColor, &[(f64, f64)])],
) -> Result<(), Box<dyn Error>> {
    let (x_low, x_high) = linear_bounds(
        curves
            .iter()
            .flat_map(|(_, samples)| samples.iter().map(|&(seconds, _)| seconds)),
    );
    let (y_low, y_high) = log_bounds(
        curves
            .iter()
            .flat_map(|(_, samples)| samples.iter().map(|&(_, loss)| loss)),
    );
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .x_label_area_size(8)
        .y_label_area_size(8)
        .build_cartesian_2d(x_low..x_high, (y_low..y_high).log_scale())?;
    // No font backend: mesh labels would panic, so the mesh draws no text.
    chart
        .configure_mesh()
        .x_labels(6)
        .y_labels(5)
        .disable_axes()
        .draw()?;
    for (colour, samples) in curves {
        let points: Vec<(f64, f64)> = samples
            .iter()
            .copied()
            .filter(|&(seconds, loss)| seconds.is_finite() && loss.is_finite() && loss > 0.0)
            .collect();
        chart.draw_series(LineSeries::new(points, colour.stroke_width(2)))?;
    }
    Ok(())
}
