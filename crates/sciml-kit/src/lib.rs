//! Model, parameter and optimizer code shared by the SciML examples.
//!
//! The modules mirror the tiers planned in
//! [tenferro-rs#2419](https://github.com/tensor4all/tenferro-rs/issues/2419),
//! so this code can move to `tenferro-func` / `tenferro-nn` /
//! `tenferro-optim` later with mechanical changes:
//!
//! | module | contents | later home |
//! | --- | --- | --- |
//! | [`func`] | the ordered parameter set a model is built from | `tenferro-func` |
//! | [`nn`] | `Linear`, `Mlp`, Xavier initialization | `tenferro-nn` |
//! | [`optim`] | `Sgd`, `Adam`, `Lbfgs` | `tenferro-optim` |
//!
//! Nothing here wraps a tenferro call: compiling, preparing and running stay
//! the plain `GraphCompiler` / `Runtime` calls in each example.
//!
//! Parameters are `f64` today, so the initial values are generated in `f64`.
//! tenferro 0.11 has no public `Tensor::cast`, and every example is `f64`, so a
//! dtype argument whose initializer could only serve `f64` would be a
//! half-supported option.

pub mod func;
pub mod nn;
pub mod optim;
