//! PNG outputs for the NQS example, using the `plotters` crate.
//!
//! The plots draw no text: `plotters` is built without a font backend, because
//! its `ttf` backend needs the FreeType and fontconfig system libraries on
//! Linux, and drawing a caption, axis label or legend without one panics.
//! The README states what each axis and color shows.

use crate::model::Model;
use plotters::prelude::*;
use std::error::Error;

#[cfg(test)]
mod tests;

/// Color ranges of the paper's Figure 2: `+-0.15` for chains, `+-0.30` for
/// square lattices. Larger weights saturate.
pub(crate) fn color_limit(model: &Model) -> f64 {
    if model.ly == 1 {
        0.15
    } else {
        0.30
    }
}

/// Diverging colormap: blue for `-limit`, white for zero, red for `+limit`.
pub(crate) fn diverging_color(value: f64, limit: f64) -> RGBColor {
    const BLUE_END: (f64, f64, f64) = (33.0, 102.0, 172.0);
    const RED_END: (f64, f64, f64) = (178.0, 24.0, 43.0);
    let t = (value / limit).clamp(-1.0, 1.0);
    let end = if t < 0.0 { BLUE_END } else { RED_END };
    let mix = |c: f64| (255.0 + (c - 255.0) * t.abs()).round() as u8;
    RGBColor(mix(end.0), mix(end.1), mix(end.2))
}

/// Translate one filter so that its largest-magnitude weight sits at site
/// `(lx/2, ly/2)`, and flip its sign so that weight is positive. Both are
/// exact symmetries of the zero-bias ansatz. Input and output use the site
/// layout `x * ly + y`.
pub(crate) fn center_filter(weights: &[f64], lx: usize, ly: usize) -> Vec<f64> {
    assert_eq!(weights.len(), lx * ly, "filter length must be lx * ly");
    let peak = (0..weights.len())
        .max_by(|&a, &b| weights[a].abs().total_cmp(&weights[b].abs()))
        .expect("filter has at least one weight");
    let sign = if weights[peak] < 0.0 { -1.0 } else { 1.0 };
    let (px, py) = (peak / ly, peak % ly);
    let mut out = vec![0.0; weights.len()];
    for x in 0..lx {
        for y in 0..ly {
            let nx = (x + lx + lx / 2 - px) % lx;
            let ny = (y + ly + ly / 2 - py) % ly;
            out[nx * ly + ny] = sign * weights[x * ly + y];
        }
    }
    out
}

/// Draw every filter of `model` after [`center_filter`], colored with
/// [`diverging_color`] in the range [`color_limit`].
///
/// A chain shows one row per filter, sites left to right. A square lattice
/// shows one square tile per filter, `x` to the right and `y` upward, with the
/// tiles in rows of `ceil(sqrt(features))`.
pub(crate) fn write_filters_png(path: &str, model: &Model) -> Result<(), Box<dyn Error>> {
    const MARGIN: i32 = 8;
    const GAP: i32 = 8;
    let (lx, ly) = (model.lx as i32, model.ly as i32);
    let features = model.features as i32;
    let limit = color_limit(model);
    let n = model.n();
    // (cell width, cell height, tiles per row)
    let (cw, ch, per_row) = if model.ly == 1 {
        ((960 / lx).max(4), 32, 1)
    } else {
        let per_row = (model.features as f64).sqrt().ceil() as i32;
        let cell = (240 / lx.max(ly)).max(4);
        (cell, cell, per_row)
    };
    let tile_w = lx * cw;
    let tile_h = ly * ch;
    let rows = (features + per_row - 1) / per_row;
    let width = 2 * MARGIN + per_row * tile_w + (per_row - 1) * GAP;
    let height = 2 * MARGIN + rows * tile_h + (rows - 1) * GAP;

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    for (f, filter) in model.weights.chunks(n).enumerate() {
        let f = f as i32;
        let left = MARGIN + (f % per_row) * (tile_w + GAP);
        let top = MARGIN + (f / per_row) * (tile_h + GAP);
        let image = center_filter(filter, model.lx, model.ly);
        for x in 0..lx {
            for y in 0..ly {
                let value = image[(x * ly + y) as usize];
                let x0 = left + x * cw;
                let y0 = top + (ly - 1 - y) * ch;
                root.draw(&Rectangle::new(
                    [(x0, y0), (x0 + cw, y0 + ch)],
                    diverging_color(value, limit).filled(),
                ))?;
            }
        }
    }
    root.present()?;
    Ok(())
}

/// Linear y-axis bounds covering the finite `values` and the `reference`,
/// with a 5% margin.
pub(crate) fn energy_axis_bounds(values: &[f64], reference: Option<f64>) -> (f64, f64) {
    let finite = values
        .iter()
        .copied()
        .chain(reference)
        .filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    if lo.is_finite() {
        let pad = ((hi - lo) * 0.05).max(1e-6);
        (lo - pad, hi + pad)
    } else {
        (-1.0, 1.0)
    }
}

/// Write the training energy per site against SR iteration, 800x600 pixels.
///
/// The red curve is the Monte Carlo estimate at each SR iteration on a linear
/// y-axis; the black horizontal line, when a `reference` is given, is the
/// reference ground-state energy per site.
pub(crate) fn write_energy_png(
    path: &str,
    energies: &[f64],
    reference: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let x_hi = energies.len().max(1) as f64;
    let (lo, hi) = energy_axis_bounds(energies, reference);
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .build_cartesian_2d(0.0..x_hi, lo..hi)?;
    chart.configure_mesh().disable_axes().draw()?;
    if let Some(reference) = reference {
        chart.draw_series(LineSeries::new(
            [(0.0, reference), (x_hi, reference)],
            BLACK.stroke_width(1),
        ))?;
    }
    chart.draw_series(LineSeries::new(
        energies
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64, v.clamp(lo, hi))),
        RED.stroke_width(2),
    ))?;
    root.present()?;
    Ok(())
}
