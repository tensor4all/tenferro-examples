//! The Levenberg-Marquardt step: the damped solve in the graph, the
//! accept/reject decision and the damping update on the host.
//!
//! `(J^T J + lambda I) delta = -J^T r` is built from the traced `J^T` and `r`,
//! so one program run returns the step and every number the host needs to accept
//! or reject it. The factorization is **not** differentiated: the AD transform is
//! applied to `sum(r)`, which is upstream of the Cholesky, and `lambda` is an
//! ordinary program input.

use tenferro_linalg::{MatOp, TracedLinalgOps, TriangularSolveOptions};
use tenferro_runtime::{Error, Result, TracedTensor};
use tenferro_tensor::{ErrorKind, Tensor};

#[cfg(test)]
mod tests;

/// Smallest damping factor the loop uses, so `J^T J + lambda I` stays definable.
pub(crate) const LAMBDA_MIN: f64 = 1e-6;
/// Largest damping factor; at this value a failing factorization ends the run.
pub(crate) const LAMBDA_MAX: f64 = 1e6;

/// The damped step and the reduction its linear model predicts.
pub(crate) struct Step {
    /// `delta`, shape `[P, 1]`.
    pub(crate) delta: TracedTensor,
    /// `0.5 * (||r||^2 - ||r + J delta||^2)`, a scalar.
    pub(crate) predicted: TracedTensor,
}

/// Build the damped solve `(J^T J + lambda I) delta = -J^T r` in the graph.
///
/// `jt` is `J^T` of shape `[P, N]`, `r` the residual `[N, 1]`, and `lambda` a
/// scalar placeholder. The Cholesky of `J^T J + lambda I` can fail numerically
/// at execution time; that surfaces as an [`Error`] whose
/// [`ErrorKind`] is [`ErrorKind::NumericalFailure`], and the caller raises
/// `lambda` and retries.
pub(crate) fn solve_step(
    jt: &TracedTensor,
    r: &TracedTensor,
    lambda: &TracedTensor,
) -> Result<Step> {
    let p = jt.try_concrete_shape().expect("J^T shape is concrete")[0];
    let j = jt.transpose(&[1, 0])?;
    let normal = jt.matmul(&j)?;
    let gradient = jt.matmul(r)?;
    // lambda I without a host buffer: a scalar broadcast to the diagonal.
    let identity = lambda.broadcast_in_dim(&[p], &[])?.embed_diag(0, 1)?;
    // A = L L^T, so (A + lambda I)^-1 b is two triangular solves.
    let factor = normal.add(&identity)?.cholesky()?;
    let step = factor.triangular_solve(&gradient)?;
    let transpose = TriangularSolveOptions::default().op(MatOp::Transpose);
    let delta = factor.triangular_solve_with(&step, &transpose)?.neg()?;
    let linear = j.matmul(&delta)?.add(r)?;
    let predicted = r
        .reduce_sum_squares(None)?
        .sub(&linear.reduce_sum_squares(None)?)?
        .scale_real(0.5)?;
    Ok(Step { delta, predicted })
}

/// What the host does with one step.
pub(crate) struct Decision {
    /// Whether the step is applied.
    pub(crate) accept: bool,
    /// Gain ratio `(loss - loss_candidate) / predicted`.
    pub(crate) rho: f64,
    /// Damping factor of the next iteration.
    pub(crate) next_lambda: f64,
}

/// Accept or reject a step from its gain ratio and pick the next damping.
///
/// The standard rule: accept when `rho > 0`, shrink the damping when the linear
/// model was good (`rho > 0.75`), grow it when the model was poor
/// (`rho < 0.25`), and grow it tenfold on a rejection. A non-finite or
/// non-positive prediction means the model carries no information, so the step
/// is rejected; that also covers a factorization that returned NaNs instead of
/// an error.
pub(crate) fn decide(lambda: f64, loss: f64, loss_candidate: f64, predicted: f64) -> Decision {
    let reduction = loss - loss_candidate;
    let rho = if predicted > 0.0 {
        reduction / predicted
    } else {
        f64::NEG_INFINITY
    };
    let accept = rho > 0.0;
    let next = if accept {
        if rho > 0.75 {
            (lambda / 3.0).max(LAMBDA_MIN)
        } else if rho < 0.25 {
            lambda * 2.0
        } else {
            lambda
        }
    } else {
        lambda * 10.0
    };
    Decision {
        accept,
        rho,
        next_lambda: next.clamp(LAMBDA_MIN, LAMBDA_MAX),
    }
}

/// Whether `error` is a numerical failure, the only error the loop retries with
/// a larger damping factor.
pub(crate) fn is_numerical_failure(error: &Error) -> bool {
    error.kind() == ErrorKind::NumericalFailure
}

/// Apply a step to the parameters on the host, block by block.
///
/// `delta` is `[P, 1]` in parameter order; each block holds one parameter's
/// entries in its own order. A new parameter list comes back, so the caller
/// keeps the old one until the step is accepted.
pub(crate) fn stepped(params: &[Tensor], delta: &[f64]) -> Result<Vec<Tensor>> {
    let mut stepped = Vec::with_capacity(params.len());
    let mut offset = 0;
    for param in params {
        let values = param.as_slice::<f64>()?;
        let updated: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(index, value)| value + delta[offset + index])
            .collect();
        stepped.push(Tensor::from_vec_col_major(param.shape().to_vec(), updated)?);
        offset += values.len();
    }
    assert_eq!(offset, delta.len(), "the step covers every parameter");
    Ok(stepped)
}
