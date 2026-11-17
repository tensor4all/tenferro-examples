//! The learned velocity field `f_θ(z) = MLP(z³)`.
//!
//! The issue prescribes a `[2, 50, 2]` MLP with `tanh` applied to the
//! elementwise cube of the state, so the model sees the cube rather than `z`.

use sciml_kit::nn::Mlp;
use tenferro_runtime::{Result, TracedTensor};

/// Layer sizes of `f_θ`: one hidden layer of width 50.
pub(crate) const MLP_SIZES: [usize; 3] = [2, 50, 2];

/// The right-hand side `dz/dt = f_θ(z³)` as a traced expression.
pub(crate) fn rhs(net: &Mlp, z: &TracedTensor) -> Result<TracedTensor> {
    let z_cubed = z.mul(z)?.mul(z)?;
    net.forward(&z_cubed)
}

#[cfg(test)]
mod tests;
