//! The tanh MLP in the two parameter layouts the example needs.
//!
//! Both layouts take the same input, `x` of shape `[dim, N]` with the point axis
//! trailing, and return `[1, N]`:
//!
//! - **plain**: one weight `[out, in]` and one bias `[out]`, the form
//!   `sciml_kit::nn::Mlp` uses;
//! - **per-point**: the same numbers with a trailing point axis, `[out, in, N]`
//!   and `[out, N]`, so every point carries its own copy of the parameters.
//!
//! A layer multiplies the broadcast weight and input activations and reduces
//! over the input features,
//!
//! ```text
//! y[out, n] = b[out, n] + sum_in W[out, in, n] * x[in, n]
//! ```
//!
//! which is the same expression in both layouts: the plain weights are reshaped
//! to `[out, in, 1]` and broadcast along the point axis. (A batched
//! `dot_general` would express this too, but tenferro 0.11 mis-indexes the
//! left-hand batch slice — a batched contraction with `lhs [in, N]`,
//! `rhs [out, in, N]` returns the slice of point *out* for every point — so the
//! example uses the elementwise form instead. The cost is the `out * in * N`
//! product tensor, about 17 MB in the widest layer of the runs below.)
//!
//! The per-point initial values are the plain ones tiled along the point axis,
//! so one seed starts the Adam and the Levenberg-Marquardt path from the same
//! `theta`.

use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use sciml_kit::func::Params;
use tenferro_runtime::{DType, Result, TraceContext, TracedTensor};
use tenferro_tensor::Tensor;

#[cfg(test)]
mod tests;

/// A fully connected tanh MLP whose parameters may carry a per-point axis.
pub(crate) struct Model {
    sizes: Vec<usize>,
    points: Option<usize>,
    params: Vec<TracedTensor>,
}

impl Model {
    /// Create the network for layer sizes `sizes`, e.g. `[2, 32, 32, 1]`.
    ///
    /// `points` is `Some(n)` for the per-point layout, which gives every
    /// parameter a trailing point axis of length `n`; `None` gives plain
    /// parameters.
    ///
    /// # Panics
    ///
    /// If `sizes` has fewer than two entries or its last entry is not 1 (the
    /// network predicts one scalar field).
    pub(crate) fn new(
        trace: &TraceContext,
        sizes: &[usize],
        points: Option<usize>,
    ) -> Result<Self> {
        assert!(sizes.len() >= 2, "sizes needs an input and an output size");
        assert_eq!(
            sizes.last(),
            Some(&1),
            "the network predicts one scalar field"
        );
        let mut params = Vec::with_capacity(2 * (sizes.len() - 1));
        for pair in sizes.windows(2) {
            let (in_features, out_features) = (pair[0], pair[1]);
            let weight_shape: Vec<usize> = match points {
                None => vec![out_features, in_features],
                Some(n) => vec![out_features, in_features, n],
            };
            let bias_shape: Vec<usize> = match points {
                None => vec![out_features],
                Some(n) => vec![out_features, n],
            };
            params.push(trace.input_concrete_shape(DType::F64, &weight_shape)?);
            params.push(trace.input_concrete_shape(DType::F64, &bias_shape)?);
        }
        Ok(Self {
            sizes: sizes.to_vec(),
            points,
            params,
        })
    }

    /// Run the network on `x` of shape `[dim, N]`, returning `[1, N]`.
    ///
    /// `params` are the parameter tensors in [`Params::placeholders`] order;
    /// they may be this model's own placeholders or any traced tensors of the
    /// matching layout (for example a candidate step).
    ///
    /// # Panics
    ///
    /// If `params` does not hold one weight and one bias per layer.
    pub(crate) fn forward(
        &self,
        params: &[&TracedTensor],
        x: &TracedTensor,
    ) -> Result<TracedTensor> {
        let layers = self.sizes.len() - 1;
        assert_eq!(
            params.len(),
            2 * layers,
            "one weight and one bias per layer"
        );
        let points = x
            .try_concrete_shape()
            .expect("a point tensor shape is concrete")[1];
        let mut y = x.clone();
        for layer in 0..layers {
            let (in_features, out_features) = (self.sizes[layer], self.sizes[layer + 1]);
            let weight = params[2 * layer];
            let weight = match self.points {
                Some(_) => weight.clone(),
                None => weight
                    .reshape([out_features, in_features, 1])?
                    .broadcast_in_dim(&[out_features, in_features, points], &[0, 1, 2])?,
            };
            let bias = params[2 * layer + 1];
            let bias = match self.points {
                Some(_) => bias.clone(),
                None => bias
                    .reshape([out_features, 1])?
                    .broadcast_in_dim(&[out_features, points], &[0, 1])?,
            };
            let activations = y.broadcast_in_dim(&[out_features, in_features, points], &[1, 2])?;
            y = weight
                .mul(&activations)?
                .reduce_sum(Some(&[1]))?
                .add(&bias)?;
            if layer + 1 < layers {
                y = y.tanh()?;
            }
        }
        Ok(y)
    }

    /// Place `values`, stored in their plain shape, in this model's layout.
    fn place(&self, values: &[f64], shape: &[usize]) -> Result<Tensor> {
        match self.points {
            None => Ok(Tensor::from_vec_col_major(shape.to_vec(), values.to_vec())?),
            Some(n) => {
                let mut tiled_shape = shape.to_vec();
                tiled_shape.push(n);
                // index = out + O*in + O*I*k, so the plain block repeats.
                let tiled = values.iter().copied().cycle().take(values.len() * n);
                Ok(Tensor::from_vec_col_major(tiled_shape, tiled.collect())?)
            }
        }
    }
}

impl Params for Model {
    fn placeholders(&self) -> Vec<&TracedTensor> {
        self.params.iter().collect()
    }

    /// Xavier-uniform weights in `±sqrt(6 / (fan_in + fan_out))`, zero biases,
    /// as in `sciml_kit::nn::Mlp`, tiled along the point axis if this model has
    /// one. Both layouts draw the same values from `rng`.
    fn initial_values(&self, rng: &mut impl Rng) -> Result<Vec<Tensor>> {
        let mut values = Vec::with_capacity(2 * (self.sizes.len() - 1));
        for pair in self.sizes.windows(2) {
            let (fan_in, fan_out) = (pair[0], pair[1]);
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let uniform = Uniform::new(-scale, scale);
            let weights: Vec<f64> = (0..fan_in * fan_out).map(|_| uniform.sample(rng)).collect();
            values.push(self.place(&weights, &[fan_out, fan_in])?);
            values.push(self.place(&vec![0.0; fan_out], &[fan_out])?);
        }
        Ok(values)
    }
}

/// Copy plain parameters into the per-point layout: each value repeated along a
/// new trailing axis of length `points`.
///
/// This is the copy the host makes on every step, `points * P * 8` bytes.
pub(crate) fn tile(params: &[Tensor], points: usize) -> Result<Vec<Tensor>> {
    let mut tiled = Vec::with_capacity(params.len());
    for param in params {
        let values = param.as_slice::<f64>()?;
        let mut shape = param.shape().to_vec();
        shape.push(points);
        let repeated = values.iter().copied().cycle().take(values.len() * points);
        tiled.push(Tensor::from_vec_col_major(shape, repeated.collect())?);
    }
    Ok(tiled)
}
