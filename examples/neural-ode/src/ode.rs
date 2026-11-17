//! Fixed-step RK4 for a traced right-hand side.
//!
//! The stepper takes the right-hand side as a closure that builds traced ops,
//! not as a fixed network, so it can move to `tenferro-diffeq` later
//! (tenferro-rs#2419). `rollout` chains it for the forward-only evaluation; the
//! training loop calls `rk4_step` once per checkpointed step (see `train`).

use tenferro_runtime::{Result, TracedTensor};

/// One classical fourth-order Runge-Kutta step
/// `z + h/6 (k1 + 2 k2 + 2 k3 + k4)`.
pub(crate) fn rk4_step(
    rhs: &impl Fn(&TracedTensor) -> Result<TracedTensor>,
    z: &TracedTensor,
    h: f64,
) -> Result<TracedTensor> {
    let k1 = rhs(z)?;
    let k2 = rhs(&z.add(&k1.scale_real(0.5 * h)?)?)?;
    let k3 = rhs(&z.add(&k2.scale_real(0.5 * h)?)?)?;
    let k4 = rhs(&z.add(&k3.scale_real(h)?)?)?;
    let stages = k1.add(&k2.scale_real(2.0)?)?.add(&k3.scale_real(2.0)?)?;
    z.add(&k4.add(&stages)?.scale_real(h / 6.0)?)
}

/// Apply `steps` RK4 steps of size `h` to `z0`, returning the state after each
/// step (the input state is not included).
pub(crate) fn rollout(
    rhs: &impl Fn(&TracedTensor) -> Result<TracedTensor>,
    z0: &TracedTensor,
    h: f64,
    steps: usize,
) -> Result<Vec<TracedTensor>> {
    let mut z = z0.clone();
    let mut states = Vec::with_capacity(steps);
    for _ in 0..steps {
        z = rk4_step(rhs, &z, h)?;
        states.push(z.clone());
    }
    Ok(states)
}

#[cfg(test)]
mod tests;
