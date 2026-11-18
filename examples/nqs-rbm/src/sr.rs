//! Stochastic reconfiguration (SR), the natural-gradient update of the paper.
//!
//! With `O_k(s)` the log-derivatives and `E_loc(s)` the local energies of
//! `N_s` samples, SR solves
//!
//! ```text
//! (S + lambda diag(S) + epsilon I) delta = F,
//! S_kl = <O_k O_l> - <O_k><O_l>,   F_k = <O_k E_loc> - <O_k><E_loc>,
//! ```
//!
//! and moves the weights by `-rate * delta`. tenferro forms the matrix
//! products and solves the linear system.

use std::error::Error;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::prelude::*;
use tenferro_runtime::CoreOps;

#[cfg(test)]
mod tests;

/// Largest step rate.
const MAX_RATE: f64 = 0.05;
/// Largest step length `sqrt(Var(sum_k O_k delta_k))` in the sampled metric.
const MAX_METRIC_STEP: f64 = 0.12;
/// Largest root-mean-square weight change per step.
const MAX_RMS_STEP: f64 = 0.002;
/// Smallest diagonal damping `lambda`.
pub(crate) const MIN_DAMPING: f64 = 0.01;

/// Diagonal damping `lambda = max(100 * 0.9^iteration, 0.01)`.
pub(crate) fn damping(iteration: usize) -> f64 {
    (100.0 * 0.9_f64.powi(iteration.min(i32::MAX as usize) as i32)).max(MIN_DAMPING)
}

/// Solve the regularized SR equations for `delta`.
///
/// `gradients` is sample-major (`gradients[s * p + k]` is `O_k` of sample
/// `s`). When there are at least as many samples as parameters, the
/// `p x p` system is solved directly. Otherwise, the Woodbury identity turns it
/// into an `N_s x N_s` system with the same solution, which avoids forming `S`.
pub(crate) fn sr_step(
    gradients: &[f64],
    energies: &[f64],
    p: usize,
    lambda: f64,
    engine: &mut CpuEngine,
    identity_shift: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let ns = energies.len();
    if gradients.len() != ns * p {
        return Err(format!(
            "{} gradient entries for {ns} samples and {p} parameters",
            gradients.len()
        )
        .into());
    }
    let means: Vec<f64> = (0..p)
        .map(|k| (0..ns).map(|s| gradients[s * p + k]).sum::<f64>() / ns as f64)
        .collect();
    let mean_e = energies.iter().sum::<f64>() / ns as f64;
    let scale = (ns as f64).sqrt();
    // Column s of the column-major [p, ns] matrix `centered` is
    // (O(s) - <O>) / sqrt(ns), so S = centered * centered^T.
    let mut centered = gradients.to_vec();
    for col in centered.chunks_mut(p) {
        for k in 0..p {
            col[k] = (col[k] - means[k]) / scale;
        }
    }
    let diagonal: Vec<f64> = (0..p)
        .map(|k| {
            (0..ns).map(|s| centered[s * p + k].powi(2)).sum::<f64>() * lambda + identity_shift
        })
        .collect();
    if p <= ns {
        let c = TypedTensor::from_vec_col_major(vec![p, ns], centered.clone())?;
        let ct = TypedTensor::from_vec_col_major(vec![ns, p], transpose(&centered, p, ns))?;
        let covariance = c.matmul(&ct, engine)?;
        let mut values = covariance.host_data()?.to_vec();
        for k in 0..p {
            values[k + p * k] += diagonal[k];
        }
        let matrix = TypedTensor::from_vec_col_major(vec![p, p], values)?;
        let force: Vec<f64> = (0..p)
            .map(|k| {
                (0..ns)
                    .map(|s| centered[s * p + k] * (energies[s] - mean_e) / scale)
                    .sum()
            })
            .collect();
        let rhs = TypedTensor::from_vec_col_major(vec![p, 1], force)?;
        Ok(matrix.solve(&rhs, engine)?.as_slice()?.to_vec())
    } else {
        // With D the regularized diagonal and Q = D^{-1/2} centered, the
        // equation reads D^{1/2} (Q Q^T + I) D^{1/2} delta = Q r for the
        // scaled energies r, and (Q Q^T + I)^{-1} Q = Q (Q^T Q + I)^{-1}.
        for col in centered.chunks_mut(p) {
            for k in 0..p {
                col[k] /= diagonal[k].sqrt();
            }
        }
        let q = TypedTensor::from_vec_col_major(vec![p, ns], centered.clone())?;
        let qt = TypedTensor::from_vec_col_major(vec![ns, p], transpose(&centered, p, ns))?;
        let gram = qt.matmul(&q, engine)?;
        let mut values = gram.host_data()?.to_vec();
        for k in 0..ns {
            values[k + ns * k] += 1.0;
        }
        let matrix = TypedTensor::from_vec_col_major(vec![ns, ns], values)?;
        let rhs = TypedTensor::from_vec_col_major(
            vec![ns, 1],
            energies.iter().map(|e| (e - mean_e) / scale).collect(),
        )?;
        let y = matrix.solve(&rhs, engine)?;
        let step = q.matmul(&y, engine)?;
        Ok(step
            .as_slice()?
            .iter()
            .zip(&diagonal)
            .map(|(x, d)| x / d.sqrt())
            .collect())
    }
}

/// Column-major transpose of a column-major `[rows, cols]` matrix.
fn transpose(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| values[c * rows + r]))
        .collect()
}

/// Step rate for `delta`, bounded so that the step stays small both in the
/// sampled metric and in weight space. The sampled metric does not constrain
/// directions absent from the batch, hence the weight-space bounds.
/// `max_weight_step = 0` disables the bound on the largest single change.
pub(crate) fn step_rate(gradients: &[f64], delta: &[f64], max_weight_step: f64) -> f64 {
    let projections: Vec<f64> = gradients
        .chunks(delta.len())
        .map(|o| o.iter().zip(delta).map(|(x, d)| x * d).sum())
        .collect();
    let (_, distance) = crate::sampler::mean_variance(&projections);
    let largest = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let rms = rms(delta);
    let rate = MAX_RATE
        .min(MAX_METRIC_STEP / (distance.sqrt() + 1e-12))
        .min(MAX_RMS_STEP / (rms + 1e-12));
    if max_weight_step > 0.0 {
        rate.min(max_weight_step / (largest + 1e-12))
    } else {
        rate
    }
}

pub(crate) fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64).sqrt()
}
