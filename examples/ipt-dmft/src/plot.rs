//! Figure of the quasiparticle weight against `U`, using the `plotters` crate.
//!
//! The plot draws no text: `plotters` is built without a font backend, because
//! drawing a caption or axis label without one panics. The sweep up from the
//! metal is blue, the sweep down from the insulator red; the README states the
//! axis ranges.

use plotters::prelude::*;
use std::error::Error;

/// Write `Z(U)` of both sweeps, on `U` from the first to the last entry of `us`
/// and `Z` from 0 to 1.1 times its largest value.
pub(crate) fn write_z_png(
    path: &str,
    us: &[f64],
    z_up: &[f64],
    z_down: &[f64],
) -> Result<(), Box<dyn Error>> {
    let (u_lo, u_hi) = (us[0], us[us.len() - 1]);
    let z_hi = 1.1
        * z_up
            .iter()
            .chain(z_down)
            .copied()
            .fold(f64::MIN_POSITIVE, f64::max);
    let root = BitMapBackend::new(path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .build_cartesian_2d(u_lo..u_hi, 0.0..z_hi)?;
    chart
        .configure_mesh()
        .x_labels(9)
        .y_labels(6)
        .disable_axes()
        .draw()?;
    for (z, colour) in [(z_up, BLUE), (z_down, RED)] {
        let points = us.iter().copied().zip(z.iter().copied());
        chart.draw_series(LineSeries::new(points.clone(), colour.stroke_width(2)))?;
        chart.draw_series(points.map(|p| Circle::new(p, 3, colour.filled())))?;
    }
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests;
