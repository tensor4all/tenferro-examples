//! Learning-rate schedule for the KdV PINN.
//!
//! The optimizer itself lives in `sciml-kit::optim`; the schedule stays with
//! the example because no other example uses this shape yet.

/// Step-decayed learning rate: `base` for the first half of training, then
/// `base / 2`, then `base / 4`.
///
/// Decaying the learning rate sharpens convergence of the PINN once Adam has
/// reached a coarse minimum, which keeps the predicted soliton from losing
/// amplitude at later times.
pub(crate) fn step_decay_lr(epoch: usize, total_epochs: usize, base: f64) -> f64 {
    if epoch < total_epochs / 2 {
        base
    } else if epoch < 3 * total_epochs / 4 {
        base * 0.5
    } else {
        base * 0.25
    }
}

#[cfg(test)]
mod tests;
