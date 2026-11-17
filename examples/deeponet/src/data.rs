//! Host-side data for the antiderivative DeepONet.
//!
//! A Gaussian random field with the RBF covariance
//! `exp(-(s-t)^2 / (2 * 0.2^2))` is sampled on a fine grid. The factorization is
//! a **Cholesky factorization of the covariance matrix**, computed on the host
//! with `TensorLinalgExt::cholesky` from tenferro-linalg 0.11.0; it is not a
//! graph operation because the data is generated once before training. A
//! diagonal jitter (`1e-10`) keeps the 397-point covariance positive definite.
//!
//! The branch network sees every fourth fine-grid point (`m = 100` sensors),
//! and the target is the cumulative trapezoid rule on the fine grid.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tenferro_cpu::CpuEngine;
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{Result, Tensor};

/// Fixed sensor points of the branch network.
pub(crate) const SENSORS: usize = 100;
/// Every fourth fine-grid point is a sensor.
const FINE_STRIDE: usize = 4;
/// Query/target points of the fine grid; `y[FINE_STRIDE * i]` is sensor `i`.
pub(crate) const Y_POINTS: usize = FINE_STRIDE * (SENSORS - 1) + 1;
/// RBF length scale from the issue.
const LENGTH_SCALE: f64 = 0.2;
/// Diagonal jitter that makes the RBF covariance positive definite in f64.
const JITTER: f64 = 1e-10;

/// The 100 uniformly spaced sensor points on `[0, 1]`.
pub(crate) fn sensor_grid() -> Vec<f64> {
    (0..SENSORS)
        .map(|i| i as f64 / (SENSORS - 1) as f64)
        .collect()
}

/// The 397 uniformly spaced query points on `[0, 1]`.
pub(crate) fn y_grid() -> Vec<f64> {
    (0..Y_POINTS)
        .map(|j| j as f64 / (Y_POINTS - 1) as f64)
        .collect()
}

/// A set of functions: branch inputs and their antiderivative targets.
///
/// Both buffers are column-major with functions as the row dimension:
/// `u[function + sensor * functions]` is sensor `sensor` of `function`, and
/// `g[function + query * functions]` is the target at query `query`. A
/// minibatch is a strided copy from each sensor/query column.
pub(crate) struct Dataset {
    pub(crate) functions: usize,
    pub(crate) u: Vec<f64>,
    pub(crate) g: Vec<f64>,
}

impl Dataset {
    /// Gather `indices` into `[batch, SENSORS]` and `[batch, Y_POINTS]` tensors.
    pub(crate) fn minibatch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let batch = indices.len();
        let mut u = vec![0.0; batch * SENSORS];
        let mut g = vec![0.0; batch * Y_POINTS];
        for (column, &function) in indices.iter().enumerate() {
            assert!(function < self.functions, "function index out of range");
            for i in 0..SENSORS {
                u[column + i * batch] = self.u[function + i * self.functions];
            }
            for j in 0..Y_POINTS {
                g[column + j * batch] = self.g[function + j * self.functions];
            }
        }
        Ok((
            Tensor::from_vec_col_major(vec![batch, SENSORS], u)?,
            Tensor::from_vec_col_major(vec![batch, Y_POINTS], g)?,
        ))
    }
}

/// Generate the training and test function sets.
///
/// The two sets come from independent seeds, so the test set is disjoint from
/// the training set and does not depend on the training count.
pub(crate) fn generate(
    train_count: usize,
    test_count: usize,
    train_seed: u64,
    test_seed: u64,
    engine: &mut CpuEngine,
) -> Result<(Dataset, Dataset)> {
    let lower = cholesky_factor(&y_grid(), engine)?;
    Ok((
        draw(train_count, train_seed, &lower),
        draw(test_count, test_seed, &lower),
    ))
}

fn draw(count: usize, seed: u64, lower: &[f64]) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let h = 1.0 / (Y_POINTS - 1) as f64;
    let mut u = vec![0.0; count * SENSORS];
    let mut g = vec![0.0; count * Y_POINTS];
    for function in 0..count {
        let field = sample_field(lower, &mut rng);
        let integral = cumulative_trapezoid(&field, h);
        for i in 0..SENSORS {
            u[function + i * count] = field[FINE_STRIDE * i];
        }
        for j in 0..Y_POINTS {
            g[function + j * count] = integral[j];
        }
    }
    Dataset {
        functions: count,
        u,
        g,
    }
}

/// Column-major RBF covariance matrix on `xs`, without the diagonal jitter.
fn covariance(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut k = vec![0.0; n * n];
    let scale = 2.0 * LENGTH_SCALE * LENGTH_SCALE;
    for j in 0..n {
        for i in 0..n {
            let d = xs[i] - xs[j];
            k[i + j * n] = (-d * d / scale).exp();
        }
    }
    k
}

/// Lower-triangular Cholesky factor of `covariance(xs) + JITTER * I`.
fn cholesky_factor(xs: &[f64], engine: &mut CpuEngine) -> Result<Vec<f64>> {
    let n = xs.len();
    let mut k = covariance(xs);
    for i in 0..n {
        k[i + i * n] += JITTER;
    }
    let matrix = Tensor::from_vec_col_major(vec![n, n], k)?;
    let lower = matrix.cholesky(engine)?;
    Ok(lower.as_slice::<f64>()?.to_vec())
}

/// Draw `u = L z` for one function, with `z` standard normal.
fn sample_field(lower: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    let n = lower.len().isqrt();
    let z: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
    let mut field = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..=i {
            sum += lower[i + j * n] * z[j];
        }
        field[i] = sum;
    }
    field
}

/// Box–Muller draw; the first uniform is `1 - U`, which lies in `(0, 1]`.
fn standard_normal(rng: &mut impl Rng) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Cumulative trapezoid rule on a uniform grid with spacing `h`.
fn cumulative_trapezoid(values: &[f64], h: f64) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for i in 1..values.len() {
        out[i] = out[i - 1] + 0.5 * (values[i - 1] + values[i]) * h;
    }
    out
}

#[cfg(test)]
mod tests;
