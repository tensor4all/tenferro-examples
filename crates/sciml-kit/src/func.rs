//! The ordered parameter set of a traced model.
//!
//! A model keeps its parameter placeholders in one fixed order. That same order
//! is the order of the initial values and of the compiled program's inputs,
//! which is what lets a training loop bind parameters and data without keeping
//! a second list of shapes and dtypes by hand.

use rand::Rng;
use tenferro_runtime::{Result, TracedTensor};
use tenferro_tensor::Tensor;

/// A model whose parameters are traced placeholders in one fixed order.
///
/// This is the hand-written stand-in for the planned `#[derive(Params)]`
/// (`tenferro-func`).
pub trait Params {
    /// The parameter placeholders, in program-input order.
    fn placeholders(&self) -> Vec<&TracedTensor>;

    /// Concrete initial values, in the same order as [`Params::placeholders`].
    fn initial_values(&self, rng: &mut impl Rng) -> Result<Vec<Tensor>>;
}
