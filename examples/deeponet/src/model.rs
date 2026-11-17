//! The DeepONet: a branch network over the sensor values, a trunk network
//! over the query points, their contraction, and the output bias `b0`.
//!
//! The ordered parameter set is one list: branch weights and biases, trunk
//! weights and biases, then `b0`. The program inputs follow the trace creation
//! order, so the parameters lead and `b0` is the last parameter.

use rand::Rng;
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use tenferro_runtime::{DType, DotGeneralConfig, Result, TraceContext, TracedTensor};
use tenferro_tensor::Tensor;

/// Hidden width of both networks and the latent dimension `p`.
const HIDDEN: usize = 40;
const LATENT: usize = 40;

/// Branch and trunk MLPs plus the scalar output bias.
pub(crate) struct DeepONet {
    branch: Mlp,
    trunk: Mlp,
    b0: TracedTensor,
}

impl DeepONet {
    /// Create both networks and `b0` in one trace, in parameter-set order.
    pub(crate) fn new(trace: &TraceContext, sensors: usize) -> Result<Self> {
        let branch = Mlp::new(
            trace,
            &[sensors, HIDDEN, HIDDEN, LATENT],
            TracedTensor::tanh,
        )?;
        let trunk = Mlp::new(trace, &[1, HIDDEN, HIDDEN, LATENT], TracedTensor::tanh)?;
        let b0 = trace.input_concrete_shape(DType::F64, &[1])?;
        Ok(Self { branch, trunk, b0 })
    }

    /// Predict `[batch, query]` from sensor values `[batch, m]` and query
    /// points `[query, 1]`.
    pub(crate) fn forward(&self, u: &TracedTensor, y: &TracedTensor) -> Result<TracedTensor> {
        let b = self.branch.forward(u)?;
        let t = self.trunk.forward(y)?;
        let pred = b.dot_general(
            &t,
            DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [1].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        )?;
        pred.add(&self.b0.reshape([1, 1])?)
    }
}

impl Params for DeepONet {
    fn placeholders(&self) -> Vec<&TracedTensor> {
        let mut placeholders = self.branch.placeholders();
        placeholders.extend(self.trunk.placeholders());
        placeholders.push(&self.b0);
        placeholders
    }

    fn initial_values(&self, rng: &mut impl Rng) -> Result<Vec<Tensor>> {
        let mut values = self.branch.initial_values(rng)?;
        values.extend(self.trunk.initial_values(rng)?);
        values.push(Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?);
        Ok(values)
    }
}

#[cfg(test)]
mod tests;
