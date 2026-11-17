//! The RBM log-amplitude in tenferro eager autodiff.
//!
//! [`rbm_log_psi`] evaluates a general RBM with biases,
//!
//! ```text
//! log psi(s) = a^T s + sum_j log(2 cosh(b_j + (W s)_j)),
//! ```
//!
//! for a batch of configurations. The translation-invariant ansatz of
//! [`crate::model`] is this RBM with zero biases, summed over the `N`
//! translated copies of one configuration, so tenferro's reverse-mode gradient
//! with respect to `W` is the tied-weight derivative that the training loop
//! computes by hand. [`tied_gradient`] returns that gradient for checking.

use crate::model::Model;
use std::error::Error;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

#[cfg(test)]
mod tests;

/// Unnormalized log amplitudes of a batch of configurations.
///
/// Shapes: `spins` is `[n_spins, batch]` with one configuration per column,
/// `a` is `[n_spins, 1]`, `b` is `[n_hidden, 1]`, `w` is `[n_hidden, n_spins]`,
/// and the result is `[batch]`.
pub(crate) fn rbm_log_psi(
    session: &mut EagerSession<'_>,
    spins: &EagerTensor,
    a: &EagerTensor,
    b: &EagerTensor,
    w: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    let projection = session.matmul(w, spins)?;
    let z = session.add(&projection, b)?;
    // log(2 cosh(z)) = softplus(2z) - z does not overflow for large z.
    let twice_z = session.add(&z, &z)?;
    let softplus = session.softplus(&twice_z)?;
    let log_cosh = session.sub(&softplus, &z)?;
    let hidden = session.reduce_sum(&log_cosh, Some(&[0]))?;
    let products = session.mul(a, spins)?;
    let visible = session.reduce_sum(&products, Some(&[0]))?;
    session.add(&visible, &hidden)
}

/// `d log psi(spins) / d W` of the translation-invariant model, computed by
/// tenferro reverse-mode autodiff and returned in the layout of
/// `model.weights`.
pub(crate) fn tied_gradient(model: &Model, spins: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
    let n = model.n();
    let features = model.features;
    let runtime = EagerRuntime::new()?;
    let mut engine = CpuEngine::serial();
    // Column t holds the configuration translated by t: entry (i, t) is s_{i+t}.
    let translated: Vec<f64> = (0..n)
        .flat_map(|t| (0..n).map(move |i| (i, t)))
        .map(|(i, t)| spins[model.translate(i, t)])
        .collect();
    // Column-major [features, n]: entry (f, i) is W^(f)_i.
    let column_major: Vec<f64> = (0..n)
        .flat_map(|i| (0..features).map(move |f| model.weights[f * n + i]))
        .collect();
    let spins = runtime.constant_from(Tensor::from_vec_col_major(vec![n, n], translated)?)?;
    let a = runtime.constant_from(Tensor::from_vec_col_major(vec![n, 1], vec![0.0_f64; n])?)?;
    let b = runtime.constant_from(Tensor::from_vec_col_major(
        vec![features, 1],
        vec![0.0_f64; features],
    )?)?;
    let w = runtime.variable_from(Tensor::from_vec_col_major(vec![features, n], column_major)?)?;
    runtime.with_eager_session(&mut engine, |session| {
        let per_translation = rbm_log_psi(session, &spins, &a, &b, &w)?;
        let loss = session.reduce_sum(&per_translation, None)?;
        session.backward(&loss)
    })?;
    let gradient = w.grad()?.ok_or("weights did not receive a gradient")?;
    let gradient = gradient.as_slice::<f64>()?;
    Ok((0..features)
        .flat_map(|f| (0..n).map(move |i| gradient[f + features * i]))
        .collect())
}
