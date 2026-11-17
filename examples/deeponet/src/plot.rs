//! Figure for the DeepONet test functions, using the `plotters` crate.
//!
//! The plot draws no text: `plotters` is built without a font backend, because
//! drawing a caption or axis label without one panics (#1968). The top row
//! shows the input field of four test functions (blue); the bottom row shows
//! the true antiderivative (blue) and the prediction (red). The README states
//! the axis ranges and colours.

use plotters::prelude::*;
use std::error::Error;

/// Padded `(low, high)` bounds of the finite values in `values`.
///
/// An empty or entirely non-finite iterator falls back to `(-1, 1)`, and a
/// constant series is padded by 1 so a chart can always be built.
pub(crate) fn axis_bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (-1.0, 1.0);
    }
    if (hi - lo).abs() < f64::EPSILON {
        return (lo - 1.0, hi + 1.0);
    }
    let pad = 0.05 * (hi - lo);
    (lo - pad, hi + pad)
}

/// Write a 2×4 panel figure: inputs on top, antiderivatives below.
///
/// `u` is `[n_test, sensors]` and `truth`/`pred` are `[n_test, y.len()]`,
/// all column-major. The figure has one column per plotted function, up to four.
pub(crate) fn write_test_png(
    path: &str,
    sensors: &[f64],
    y: &[f64],
    u: &[f64],
    truth: &[f64],
    pred: &[f64],
    n_test: usize,
) -> Result<(), Box<dyn Error>> {
    const COLUMNS: usize = 4;
    const W: u32 = 1200;
    const H: u32 = 600;

    let n_plot = COLUMNS.min(n_test);
    let (u_lo, u_hi) =
        axis_bounds((0..n_plot).flat_map(|f| (0..sensors.len()).map(move |i| u[f + i * n_test])));
    let (g_lo, g_hi) = axis_bounds((0..n_plot).flat_map(|f| {
        (0..y.len()).flat_map(move |j| [truth[f + j * n_test], pred[f + j * n_test]])
    }));

    let root = BitMapBackend::new(path, (W, H)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, COLUMNS));

    for column in 0..n_plot {
        let mut input_chart = ChartBuilder::on(&panels[column])
            .margin(6)
            .build_cartesian_2d(0.0..1.0, u_lo..u_hi)?;
        input_chart
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .disable_axes()
            .draw()?;
        input_chart.draw_series(LineSeries::new(
            sensors
                .iter()
                .enumerate()
                .map(|(i, &x)| (x, u[column + i * n_test])),
            BLUE.stroke_width(2),
        ))?;

        let mut target_chart = ChartBuilder::on(&panels[COLUMNS + column])
            .margin(6)
            .build_cartesian_2d(0.0..1.0, g_lo..g_hi)?;
        target_chart
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .disable_axes()
            .draw()?;
        target_chart.draw_series(LineSeries::new(
            y.iter()
                .enumerate()
                .map(|(j, &x)| (x, truth[column + j * n_test])),
            BLUE.stroke_width(2),
        ))?;
        target_chart.draw_series(LineSeries::new(
            y.iter()
                .enumerate()
                .map(|(j, &x)| (x, pred[column + j * n_test])),
            RED.stroke_width(2),
        ))?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests;
