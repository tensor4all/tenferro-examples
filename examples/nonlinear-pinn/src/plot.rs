//! Snapshot figure, using the `plotters` crate.
//!
//! The plot draws no text: `plotters` is built without a font backend, because
//! drawing a caption or axis label without one panics. Each of the four panels
//! is one time (`t = 0.25, 0.5, 0.75, 1`, left to right) over `x ∈ [-1, 1]`
//! and `u ∈ [-1.1, 1.1]`; the reference is blue and the prediction red. The
//! README states this.

use plotters::prelude::*;
use std::error::Error;

/// Write a 1 x 4 panel figure, one panel per `(reference, prediction)` pair.
pub(crate) fn write_snapshots_png(
    path: &str,
    x: &[f64],
    snapshots: &[(Vec<f64>, Vec<f64>)],
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (1600, 420)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, snapshots.len().max(1)));
    for (panel, (truth, pred)) in panels.iter().zip(snapshots) {
        let mut chart = ChartBuilder::on(panel)
            .margin(12)
            .build_cartesian_2d(-1.0..1.0, -1.1..1.1)?;
        chart.configure_mesh().disable_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            x.iter().copied().zip(truth.iter().copied()),
            BLUE.stroke_width(3),
        ))?;
        chart.draw_series(LineSeries::new(
            x.iter().copied().zip(pred.iter().copied()),
            RED.stroke_width(2),
        ))?;
    }
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests;
