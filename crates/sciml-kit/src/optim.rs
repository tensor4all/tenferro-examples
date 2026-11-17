//! First-order optimizers over concrete parameter tensors.
//!
//! The optimizers are host-side: tenferro has no input donation yet
//! (tenferro-rs#2419), so a step reads the gradient tensors, updates the
//! parameter tensors in place, and the next program run binds them again.
//! Their interface takes and returns parameters in the parameter-set order, so
//! a graph-side version can keep it.

use tenferro_runtime::DType;
use tenferro_tensor::{Complex64, Tensor, TensorScalar};

/// Stochastic gradient descent with a fixed learning rate.
pub struct Sgd {
    lr: f64,
}

impl Sgd {
    /// Create an SGD optimizer with learning rate `lr`.
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }

    /// Update `params` in place along `-grads`.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length or a tensor is not `f64`.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have the same length"
        );
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let p = param
                .as_slice_mut::<f64>()
                .expect("parameter tensor must be f64");
            let g = grad.as_slice::<f64>().expect("gradient tensor must be f64");
            assert_eq!(
                p.len(),
                g.len(),
                "parameter and gradient buffers must have the same length"
            );
            for (p, g) in p.iter_mut().zip(g.iter()) {
                *p -= self.lr * g;
            }
        }
    }
}

/// Adam with first- and second-moment estimates.
///
/// Moment buffers are allocated on the first [`Adam::step`] and reused. The
/// second moment is `g * conj(g)`, so complex parameters (whose gradients
/// follow tenferro's `Re(conj(a) * b)` convention) take the real-valued
/// variance and the same `p -= lr * m_hat / (sqrt(v_hat) + eps)` step.
///
/// shortcut: parameters must be `f64` or `c64`; add `f32` when an example uses it.
pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl Adam {
    /// Create an Adam optimizer with learning rate `lr`.
    ///
    /// Defaults: `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Override the learning rate of the following steps.
    ///
    /// The accumulated moments are kept, so a schedule can be applied without
    /// restarting the optimizer.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Perform one Adam update of `params` in place.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length, or a tensor's dtype is neither
    /// `f64` nor `c64`, or the parameters mix dtypes.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have the same length"
        );
        match params.first().map(Tensor::dtype) {
            Some(DType::F64) => self.step_typed::<f64>(params, grads),
            Some(DType::C64) => self.step_typed::<Complex64>(params, grads),
            Some(other) => panic!("Adam supports f64 and c64 parameters, got {other:?}"),
            None => {}
        }
    }

    fn step_typed<T: AdamScalar>(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        if self.m.is_empty() {
            for param in params.iter() {
                let shape = param.shape().to_vec();
                let len = shape.iter().product::<usize>();
                self.m
                    .push(Tensor::from_vec_col_major(shape.clone(), vec![T::zero(); len]).unwrap());
                self.v
                    .push(Tensor::from_vec_col_major(shape, vec![T::zero(); len]).unwrap());
            }
        }

        self.t += 1;
        let (beta1, beta2, lr, eps) = (self.beta1, self.beta2, self.lr, self.eps);
        let correction1 = 1.0 - beta1.powi(self.t as i32);
        let correction2 = 1.0 - beta2.powi(self.t as i32);

        for ((param, grad), (m, v)) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            let p = param
                .as_slice_mut::<T>()
                .expect("parameter dtype matches the first parameter");
            let g = grad
                .as_slice::<T>()
                .expect("gradient dtype matches the parameter");
            let m = m
                .as_slice_mut::<T>()
                .expect("moment buffer dtype matches the parameter");
            let v = v
                .as_slice_mut::<T>()
                .expect("moment buffer dtype matches the parameter");
            assert_eq!(
                p.len(),
                g.len(),
                "parameter and gradient lengths must match"
            );
            assert_eq!(
                p.len(),
                m.len(),
                "parameter and first-moment buffer lengths must match"
            );
            assert_eq!(
                p.len(),
                v.len(),
                "parameter and second-moment buffer lengths must match"
            );

            for i in 0..p.len() {
                m[i] = m[i].scale(beta1).add_scaled(1.0 - beta1, g[i]);
                v[i] = T::from_real(beta2 * v[i].real() + (1.0 - beta2) * g[i].norm_sq());
                let m_hat = m[i].scale(1.0 / correction1);
                let v_hat = v[i].real() / correction2;
                p[i] = p[i].add_scaled(-lr / (v_hat.sqrt() + eps), m_hat);
            }
        }
    }
}

/// Step-decayed learning rate: `base` for the first half of training, then
/// `base / 2`, then `base / 4`.
///
/// Decaying the learning rate sharpens convergence once Adam has reached a
/// coarse minimum. The KdV PINN and the neural ODE share this shape.
pub fn step_decay_lr(epoch: usize, total_epochs: usize, base: f64) -> f64 {
    if epoch < total_epochs / 2 {
        base
    } else if epoch < 3 * total_epochs / 4 {
        base * 0.5
    } else {
        base * 0.25
    }
}

/// Host arithmetic an Adam step needs from a parameter scalar.
///
/// Real-valued quantities (`norm_sq`, `real`) are `f64` because the update is
/// performed in double precision for both supported dtypes.
trait AdamScalar: TensorScalar {
    fn zero() -> Self;
    fn norm_sq(self) -> f64;
    fn real(self) -> f64;
    fn from_real(value: f64) -> Self;
    fn scale(self, factor: f64) -> Self;
    fn add_scaled(self, factor: f64, other: Self) -> Self;
}

impl AdamScalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn norm_sq(self) -> f64 {
        self * self
    }
    fn real(self) -> f64 {
        self
    }
    fn from_real(value: f64) -> Self {
        value
    }
    fn scale(self, factor: f64) -> Self {
        self * factor
    }
    fn add_scaled(self, factor: f64, other: Self) -> Self {
        self + factor * other
    }
}

impl AdamScalar for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn norm_sq(self) -> f64 {
        self.norm_sqr()
    }
    fn real(self) -> f64 {
        self.re
    }
    fn from_real(value: f64) -> Self {
        Complex64::new(value, 0.0)
    }
    fn scale(self, factor: f64) -> Self {
        self * factor
    }
    fn add_scaled(self, factor: f64, other: Self) -> Self {
        self + factor * other
    }
}

#[cfg(test)]
mod tests;
