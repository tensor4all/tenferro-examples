//! The reference trajectory and the 10-step training windows of the spiral ODE
//! `dz/dt = A z³`.
//!
//! The reference is integrated on the host, outside the graph, with RK4 at
//! `SUBSTEPS` steps per sample interval (`H / SUBSTEPS`), a hundred times finer
//! than the step the learned solver uses. The sampled states are the targets;
//! the model never sees the reference while it is being trained except through
//! these targets.

use tenferro_tensor::{Result, Tensor};

/// Number of sample times on `[0, 25]`, including both ends.
pub(crate) const N_SAMPLES: usize = 1000;
/// RK4 steps inside one training window. The windows are processed by the
/// checkpointed adjoint in `train`, one step program run per step.
pub(crate) const WINDOW_STEPS: usize = 10;
/// Window starts drawn per epoch.
pub(crate) const BATCH: usize = 64;
/// The model's fixed solver step: the spacing of the sample times.
pub(crate) const H: f64 = 25.0 / (N_SAMPLES - 1) as f64;
/// Fine RK4 steps per sample interval used for the reference trajectory.
const SUBSTEPS: usize = 100;
/// Initial state `z(0) = [2, 0]` from the issue.
const Z0: [f64; 2] = [2.0, 0.0];

/// The ground-truth velocity `dz/dt = A z³` with
/// `A = [[-0.1, 2.0], [-2.0, -0.1]]` (elementwise cube).
pub(crate) fn velocity(z: [f64; 2]) -> [f64; 2] {
    let x_cubed = z[0] * z[0] * z[0];
    let y_cubed = z[1] * z[1] * z[1];
    [
        -0.1 * x_cubed + 2.0 * y_cubed,
        -2.0 * x_cubed - 0.1 * y_cubed,
    ]
}

/// `z + scale * k`, the stage update of one RK4 step.
fn shifted(z: [f64; 2], k: [f64; 2], scale: f64) -> [f64; 2] {
    [z[0] + scale * k[0], z[1] + scale * k[1]]
}

/// One classical RK4 step of [`velocity`].
fn rk4(z: [f64; 2], h: f64) -> [f64; 2] {
    let k1 = velocity(z);
    let k2 = velocity(shifted(z, k1, 0.5 * h));
    let k3 = velocity(shifted(z, k2, 0.5 * h));
    let k4 = velocity(shifted(z, k3, h));
    [
        z[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        z[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]
}

/// The sampled reference trajectory, column-major as `[N_SAMPLES, 2]`.
pub(crate) struct Dataset {
    states: Vec<f64>,
}

impl Dataset {
    /// Integrate the reference trajectory at the resolution used for training.
    pub(crate) fn generate() -> Self {
        Self::generate_with_substeps(SUBSTEPS)
    }

    /// Integrate with an explicit number of fine steps per sample interval.
    ///
    /// Used by the convergence test; the training path always uses
    /// [`SUBSTEPS`].
    fn generate_with_substeps(substeps: usize) -> Self {
        assert!(substeps > 0, "the reference needs at least one substep");
        let h = H / substeps as f64;
        let mut states = vec![0.0; N_SAMPLES * 2];
        let mut z = Z0;
        states[0] = z[0];
        states[N_SAMPLES] = z[1];
        for sample in 1..N_SAMPLES {
            for _ in 0..substeps {
                z = rk4(z, h);
            }
            states[sample] = z[0];
            states[sample + N_SAMPLES] = z[1];
        }
        Self { states }
    }

    /// The reference state at sample time `index`, for `index < N_SAMPLES`.
    pub(crate) fn state(&self, index: usize) -> [f64; 2] {
        [self.states[index], self.states[index + N_SAMPLES]]
    }

    /// The whole reference buffer, column-major as `[N_SAMPLES, 2]`.
    pub(crate) fn states(&self) -> &[f64] {
        &self.states
    }

    /// Gather `starts` into the initial states `[B, 2]` and one target
    /// `[B, 2]` per window step.
    ///
    /// A window covers the reference states at `start + 1 ..= start + 10`; the
    /// initial state is bound separately, so the loss does not count the
    /// trivially reproduced `start` state.
    pub(crate) fn minibatch(&self, starts: &[usize]) -> Result<(Tensor, Vec<Tensor>)> {
        let batch = starts.len();
        let mut initial = vec![0.0; batch * 2];
        let mut targets = vec![vec![0.0; batch * 2]; WINDOW_STEPS];
        for (column, &start) in starts.iter().enumerate() {
            assert!(
                start + WINDOW_STEPS < N_SAMPLES,
                "window start {start} leaves the reference trajectory"
            );
            for component in 0..2 {
                initial[column + batch * component] = self.state(start)[component];
            }
            for (target, step) in targets.iter_mut().zip(1..=WINDOW_STEPS) {
                for component in 0..2 {
                    // [B, 2], column-major.
                    target[column + batch * component] = self.state(start + step)[component];
                }
            }
        }
        Ok((
            Tensor::from_vec_col_major(vec![batch, 2], initial)?,
            targets
                .into_iter()
                .map(|target| Tensor::from_vec_col_major(vec![batch, 2], target))
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

#[cfg(test)]
mod tests;
