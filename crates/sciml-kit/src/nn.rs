//! A multi-layer perceptron built from `TracedTensor` placeholders.
//!
//! Parameters are placeholders, not data, so the model can be built into a
//! graph that is compiled once and then executed with fresh parameter values.

use crate::func::Params;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use tenferro_runtime::{DType, DotGeneralConfig, Result, TraceContext, TracedTensor};
use tenferro_tensor::Tensor;

/// Activation applied between hidden layers, as a traced function.
pub type Activation = fn(&TracedTensor) -> Result<TracedTensor>;

/// A fully connected layer.
pub struct Linear {
    weight: TracedTensor,
    bias: TracedTensor,
    out_features: usize,
}

impl Linear {
    /// Create a layer mapping `in_features` inputs to `out_features` outputs.
    ///
    /// The placeholders are created in `trace`, which must be the same trace
    /// that holds the inputs passed to [`Linear::forward`].
    pub fn new(trace: &TraceContext, in_features: usize, out_features: usize) -> Result<Self> {
        let weight = trace.input_concrete_shape(DType::F64, &[in_features, out_features])?;
        let bias = trace.input_concrete_shape(DType::F64, &[out_features])?;
        Ok(Self {
            weight,
            bias,
            out_features,
        })
    }

    /// Apply the layer to a batch of inputs of shape `[batch, in_features]`.
    pub fn forward(&self, x: &TracedTensor) -> Result<TracedTensor> {
        let y = x.dot_general(
            &self.weight,
            DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        )?;
        y.add(&self.bias.reshape([1, self.out_features])?)
    }
}

/// A multi-layer perceptron: `Linear` layers with `activation` between them.
pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
}

impl Mlp {
    /// Create a network with one `Linear` layer per adjacent pair of sizes.
    ///
    /// For example `&[2, 64, 64, 1]` is two hidden layers of width 64. All
    /// parameter placeholders are created in `trace`; build the network inputs
    /// in the same trace so they can be combined.
    pub fn new(
        trace: &TraceContext,
        layer_sizes: &[usize],
        activation: Activation,
    ) -> Result<Self> {
        assert!(
            layer_sizes.len() >= 2,
            "Mlp needs at least an input and an output size"
        );
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for pair in layer_sizes.windows(2) {
            layers.push(Linear::new(trace, pair[0], pair[1])?);
        }
        Ok(Self { layers, activation })
    }

    /// Run a forward pass over a batch of inputs of shape `[batch, inputs]`.
    pub fn forward(&self, x: &TracedTensor) -> Result<TracedTensor> {
        let mut y = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            y = layer.forward(&y)?;
            if i + 1 < self.layers.len() {
                y = (self.activation)(&y)?;
            }
        }
        Ok(y)
    }
}

impl Params for Mlp {
    fn placeholders(&self) -> Vec<&TracedTensor> {
        let mut placeholders = Vec::with_capacity(2 * self.layers.len());
        for layer in &self.layers {
            placeholders.push(&layer.weight);
            placeholders.push(&layer.bias);
        }
        placeholders
    }

    /// Xavier-uniform weights in `±sqrt(6 / (fan_in + fan_out))`, zero biases.
    fn initial_values(&self, rng: &mut impl Rng) -> Result<Vec<Tensor>> {
        let mut values = Vec::with_capacity(2 * self.layers.len());
        for layer in &self.layers {
            let shape = layer
                .weight
                .try_concrete_shape()
                .expect("weight placeholder shape is concrete");
            let (fan_in, fan_out) = (shape[0], shape[1]);
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let uniform = Uniform::new(-scale, scale);
            values.push(Tensor::from_vec_col_major(
                shape,
                (0..fan_in * fan_out).map(|_| uniform.sample(rng)).collect(),
            )?);

            let bias_shape = layer
                .bias
                .try_concrete_shape()
                .expect("bias placeholder shape is concrete");
            let len = bias_shape.iter().product::<usize>();
            values.push(Tensor::from_vec_col_major(bias_shape, vec![0.0_f64; len])?);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests;
