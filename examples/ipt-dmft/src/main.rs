//! DMFT for the half-filled Hubbard model on the Bethe lattice with the IPT
//! impurity solver (Georges et al., Rev. Mod. Phys. 68, 13 (1996)).
//!
//! Sweeps `U` up from the metal and down from the insulator, each step starting
//! from the previous solution, and prints the quasiparticle weight `Z` of both
//! branches. Where they differ, a metallic and an insulating solution coexist:
//! the first-order Mott transition.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p ipt_dmft
//! ```
//!
//! Flags:
//!
//! - `--beta B` sets the inverse temperature in units of `1/D` (default 100).
//! - `--n-matsubara M` sets the number of positive Matsubara frequencies (default 2048).
//! - `--iterations N` caps the self-consistency iterations per `U` (default 3000).
//! - `--plot <path>` writes `Z` against `U` for both sweeps.

mod ipt;
mod plot;

use ipt::{max_abs_difference, Grid, Result};
use std::time::Instant;
use tenferro_cpu::CpuEngine;

const U_MIN: f64 = 2.0;
const U_MAX: f64 = 3.6;
const U_STEPS: usize = 32;
/// Two sweeps whose `Z` differ by more than this are different solutions.
const Z_DIFFERENT: f64 = 1e-3;

/// Return the value following the given command-line `flag`, if present.
fn arg_value(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for window in args.windows(2) {
        if window[0] == flag {
            return Some(window[1].clone());
        }
    }
    None
}

fn parsed<T: std::str::FromStr>(flag: &str, default: T) -> Result<T>
where
    T::Err: std::error::Error + 'static,
{
    Ok(match arg_value(flag) {
        Some(value) => value.parse()?,
        None => default,
    })
}

/// One sweep over `us`, each solve starting from the previous `G`.
/// Returns `Z` and whether the loop converged, per `U`.
fn sweep(
    grid: &mut Grid,
    us: &[f64],
    start: tenferro_tensor::Tensor,
    max_iterations: usize,
    engine: &mut CpuEngine,
) -> Result<Vec<(f64, bool)>> {
    let mut g = start;
    let mut results = Vec::with_capacity(us.len());
    for &u in us {
        let solution = grid.solve(u, g, max_iterations, engine)?;
        results.push((
            grid.quasiparticle_weight(&solution.sigma)?,
            solution.converged,
        ));
        g = solution.g;
    }
    Ok(results)
}

/// The first and last `U` where the two sweeps reached different solutions.
fn coexistence(us: &[f64], z_up: &[f64], z_down: &[f64]) -> Option<(f64, f64)> {
    let differ: Vec<f64> = us
        .iter()
        .zip(z_up.iter().zip(z_down))
        .filter(|(_, (up, down))| (*up - *down).abs() > Z_DIFFERENT)
        .map(|(u, _)| *u)
        .collect();
    Some((*differ.first()?, *differ.last()?))
}

fn main() -> Result<()> {
    let beta = parsed("--beta", 100.0)?;
    let n_iw = parsed("--n-matsubara", 2048)?;
    let max_iterations = parsed("--iterations", 3000)?;
    println!("β D = {beta}, Matsubara frequencies: {n_iw}, U/D in [{U_MIN}, {U_MAX}]");

    let start = Instant::now();
    let mut engine = CpuEngine::serial();
    let mut grid = Grid::new(beta, n_iw)?;
    let us: Vec<f64> = (0..=U_STEPS)
        .map(|i| U_MIN + (U_MAX - U_MIN) * i as f64 / U_STEPS as f64)
        .collect();
    let metal_start = grid.semicircle()?;
    let free = grid.solve(0.0, grid.atomic(1.0)?, max_iterations, &mut engine)?;
    println!(
        "U = 0 check: max |G - G_semicircle| = {:.1e}",
        max_abs_difference(&free.g, &metal_start)?
    );
    let up = sweep(&mut grid, &us, metal_start, max_iterations, &mut engine)?;
    let reversed: Vec<f64> = us.iter().rev().copied().collect();
    let insulator_start = grid.atomic(U_MAX)?;
    let mut down = sweep(
        &mut grid,
        &reversed,
        insulator_start,
        max_iterations,
        &mut engine,
    )?;
    down.reverse();

    println!("{:>6} {:>12} {:>12}", "U/D", "Z (up)", "Z (down)");
    for ((u, (z_up, ok_up)), (z_down, ok_down)) in us.iter().zip(&up).zip(&down) {
        let mark = |ok: &bool| if *ok { ' ' } else { '*' };
        println!(
            "{u:6.3} {z_up:12.6}{} {z_down:12.6}{}",
            mark(ok_up),
            mark(ok_down)
        );
    }
    if up.iter().chain(&down).any(|(_, ok)| !ok) {
        println!("* not converged within {max_iterations} iterations");
    }
    let z_up: Vec<f64> = up.iter().map(|(z, _)| *z).collect();
    let z_down: Vec<f64> = down.iter().map(|(z, _)| *z).collect();
    match coexistence(&us, &z_up, &z_down) {
        Some((low, high)) => {
            println!("coexistence: insulator down to U = {low:.3} D, metal up to U = {high:.3} D")
        }
        None => println!("no coexistence on this U grid"),
    }
    println!("time: {:.1} s", start.elapsed().as_secs_f64());

    if let Some(path) = arg_value("--plot") {
        plot::write_z_png(&path, &us, &z_up, &z_down)?;
        println!("wrote {path}");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
