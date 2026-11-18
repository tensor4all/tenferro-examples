//! Benchmark ground-state energies and exact checks on small lattices.
//!
//! All energies use Pauli matrices, as in the paper's Hamiltonians; energies
//! quoted for `S_i . S_j` must be multiplied by four before comparison.

use crate::model::{Hamiltonian, Model};
use std::error::Error;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::prelude::*;

#[cfg(test)]
mod tests;

/// Ground-state energy per site of the 10x10 periodic Heisenberg model, from
/// quantum Monte Carlo, `E/N = -4 * 0.67155267` (Chen and Heyl,
/// <https://pmc.ncbi.nlm.nih.gov/articles/PMC11392813/>).
const HEISENBERG_10X10: f64 = -4.0 * 0.671_552_67;

/// Largest Hilbert-space sector that [`exact_check`] diagonalizes densely.
pub(crate) const MAX_EXACT_DIMENSION: usize = 4096;

/// Reference ground-state energy per site, when one is known for the lattice.
pub(crate) fn reference_energy(model: &Model) -> Result<Option<f64>, Box<dyn Error>> {
    let n = model.n();
    Ok(match (model.hamiltonian, model.ly) {
        (Hamiltonian::Ising, 1) if n >= 3 => Some(ising_chain(n)),
        (Hamiltonian::Heisenberg, 1) if n >= 4 => Some(heisenberg_chain(n)?),
        (Hamiltonian::Heisenberg, 10) if model.lx == 10 => Some(HEISENBERG_10X10),
        _ => None,
    })
}

/// Exact ground-state energy per site of the periodic critical Ising chain,
/// `-2 / (N sin(pi / 2N))`, from its free-fermion solution.
pub(crate) fn ising_chain(n: usize) -> f64 {
    let n = n as f64;
    -2.0 / (n * (std::f64::consts::PI / (2.0 * n)).sin())
}

/// Exact ground-state energy per site of the periodic Heisenberg chain with
/// an even number `n >= 4` of sites, from the Bethe ansatz.
///
/// The `m = n/2` real rapidities solve
/// `n atan(2 x_j) - sum_k atan(x_j - x_k) = pi I_j` with
/// `I_j = j - (m - 1)/2`, and `E/N = 1 - (2/n) sum_j 1 / (x_j^2 + 1/4)`.
/// The equations are solved by Newton's method.
pub(crate) fn heisenberg_chain(n: usize) -> Result<f64, Box<dyn Error>> {
    if n < 4 || !n.is_multiple_of(2) {
        return Err(
            format!("Bethe reference needs an even chain of at least 4 sites, got {n}").into(),
        );
    }
    let m = n / 2;
    let nf = n as f64;
    let pi = std::f64::consts::PI;
    let quantum: Vec<f64> = (0..m).map(|j| j as f64 - (m as f64 - 1.0) / 2.0).collect();
    let mut x: Vec<f64> = quantum
        .iter()
        .map(|q| (pi * q / nf).tan().asinh() / pi)
        .collect();
    let residual = |x: &[f64]| -> Vec<f64> {
        (0..m)
            .map(|j| {
                nf * (2.0 * x[j]).atan()
                    - (0..m).map(|k| (x[j] - x[k]).atan()).sum::<f64>()
                    - pi * quantum[j]
            })
            .collect()
    };
    let mut engine = CpuEngine::serial();
    for _ in 0..100 {
        let r = residual(&x);
        if r.iter().all(|v| v.abs() < 1e-12) {
            let sum: f64 = x.iter().map(|v| 1.0 / (v * v + 0.25)).sum();
            return Ok(1.0 - 2.0 * sum / nf);
        }
        // Column-major Jacobian of the residual.
        let mut jacobian = vec![0.0; m * m];
        for j in 0..m {
            let mut diagonal = 2.0 * nf / (1.0 + 4.0 * x[j] * x[j]);
            for l in 0..m {
                if l != j {
                    let coupling = 1.0 / (1.0 + (x[j] - x[l]).powi(2));
                    jacobian[j + m * l] = coupling;
                    diagonal -= coupling;
                }
            }
            jacobian[j + m * j] = diagonal;
        }
        let jacobian = TypedTensor::from_vec_col_major(vec![m, m], jacobian)?;
        let rhs = TypedTensor::from_vec_col_major(vec![m, 1], r)?;
        let step = jacobian.solve(&rhs, &mut engine)?;
        for (x, d) in x.iter_mut().zip(step.as_slice()?) {
            *x -= d;
        }
    }
    Err(format!("Bethe equations for {n} sites did not converge").into())
}

/// Fail unless the sampled sector of `n` sites has at most
/// [`MAX_EXACT_DIMENSION`] states.
pub(crate) fn check_exact_size(hamiltonian: Hamiltonian, n: usize) -> Result<(), String> {
    // Number of configurations: 2^n, or n choose n/2 at zero magnetization.
    let dimension = match hamiltonian {
        _ if n >= 64 => None,
        Hamiltonian::Ising => Some(1u64 << n),
        Hamiltonian::Heisenberg => (0..n as u64 / 2).try_fold(1u64, |c, k| {
            c.checked_mul(n as u64 - k).map(|c| c / (k + 1))
        }),
    };
    match dimension {
        Some(d) if d <= MAX_EXACT_DIMENSION as u64 => Ok(()),
        _ => Err(format!(
            "the exact check supports sectors of at most {MAX_EXACT_DIMENSION} states; \
             {n} sites have more"
        )),
    }
}

/// Exact quantities of the trained state on a lattice small enough to
/// diagonalize; energies are per site.
#[derive(Debug)]
pub(crate) struct ExactCheck {
    pub(crate) dimension: usize,
    pub(crate) ground_energy: f64,
    pub(crate) variational_energy: f64,
    /// `<H^2> - <H>^2` of the trained state, per site.
    pub(crate) variance: f64,
    /// `|<ground|psi>|^2 / <psi|psi>`.
    pub(crate) fidelity: f64,
}

/// Build the Hamiltonian as a dense matrix on the sampled sector (all
/// configurations for Ising, zero magnetization for Heisenberg, in the
/// Marshall-rotated basis), diagonalize it with tenferro, and evaluate the
/// trained wavefunction exactly.
pub(crate) fn exact_check(model: &Model) -> Result<ExactCheck, Box<dyn Error>> {
    let n = model.n();
    check_exact_size(model.hamiltonian, n)?;
    let states: Vec<u32> = (0..1u32 << n)
        .filter(|s| model.hamiltonian == Hamiltonian::Ising || s.count_ones() as usize == n / 2)
        .collect();
    let d = states.len();
    let index = |state: u32| {
        states
            .binary_search(&state)
            .expect("Hamiltonian connects only states inside the sector")
    };
    let spin = |state: u32, i: usize| if state & (1 << i) == 0 { -1.0 } else { 1.0 };
    let mut h = vec![0.0; d * d];
    for (column, &state) in states.iter().enumerate() {
        for &(i, j) in &model.bonds {
            let zz = spin(state, i) * spin(state, j);
            match model.hamiltonian {
                Hamiltonian::Ising => h[column + d * column] -= zz,
                Hamiltonian::Heisenberg => {
                    h[column + d * column] += zz;
                    if zz < 0.0 {
                        h[index(state ^ (1 << i) ^ (1 << j)) + d * column] -= 2.0;
                    }
                }
            }
        }
        if model.hamiltonian == Hamiltonian::Ising {
            for i in 0..n {
                h[index(state ^ (1 << i)) + d * column] -= 1.0;
            }
        }
    }
    let logs: Vec<f64> = states
        .iter()
        .map(|&state| {
            let spins: Vec<f64> = (0..n).map(|i| spin(state, i)).collect();
            model.log_psi(&spins)
        })
        .collect();
    let max_log = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let psi: Vec<f64> = logs.iter().map(|l| (l - max_log).exp()).collect();
    let h_psi: Vec<f64> = (0..d)
        .map(|r| (0..d).map(|c| h[r + d * c] * psi[c]).sum())
        .collect();
    let norm: f64 = psi.iter().map(|x| x * x).sum();
    let energy = dot(&psi, &h_psi) / norm;
    let variance = dot(&h_psi, &h_psi) / norm - energy * energy;

    let Eigh { values, vectors } =
        TypedTensor::from_vec_col_major(vec![d, d], h)?.eigh(&mut CpuEngine::serial())?;
    let values = values.as_slice()?;
    let vectors = vectors.as_slice()?;
    let ground = (0..d)
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .ok_or("empty Hilbert space")?;
    let overlap = dot(&vectors[ground * d..(ground + 1) * d], &psi);
    let nf = n as f64;
    Ok(ExactCheck {
        dimension: d,
        ground_energy: values[ground] / nf,
        variational_energy: energy / nf,
        variance: variance / nf,
        fidelity: overlap * overlap / norm,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
