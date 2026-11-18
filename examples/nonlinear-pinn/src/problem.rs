//! The two problems: their coefficients, initial data, network input and
//! traced PDE residual.
//!
//! - Viscous Burgers, `u_t + u u_x = (0.01 / pi) u_xx` on `x ∈ [-1, 1]`,
//!   `t ∈ [0, 1]`, `u(x, 0) = -sin(pi x)`, `u(±1, t) = 0` (Raissi, Perdikaris and
//!   Karniadakis, J. Comput. Phys. 378, 686 (2019)). The network sees `(x, t)`
//!   and the boundary values are a loss term.
//! - Allen-Cahn, `u_t = 1e-4 u_xx - 5 u³ + 5 u` on the same domain,
//!   `u(x, 0) = x² cos(pi x)`, periodic in `x`. The network sees
//!   `(cos(pi x), sin(pi x), t)`, so every prediction is exactly periodic in `x`
//!   with all its derivatives, and the boundary needs no loss term.

use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{Error, Result, TracedTensor};

/// Burgers viscosity.
pub(crate) const NU: f64 = 0.01 / std::f64::consts::PI;
/// Allen-Cahn diffusion.
pub(crate) const EPSILON: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Problem {
    Burgers,
    AllenCahn,
}

impl Problem {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "burgers" => Some(Self::Burgers),
            "allen-cahn" => Some(Self::AllenCahn),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Burgers => "burgers",
            Self::AllenCahn => "allen-cahn",
        }
    }

    /// The coefficient `c` of `u_xx`.
    pub(crate) fn diffusion(self) -> f64 {
        match self {
            Self::Burgers => NU,
            Self::AllenCahn => EPSILON,
        }
    }

    pub(crate) fn initial(self, x: f64) -> f64 {
        let pi = std::f64::consts::PI;
        match self {
            Self::Burgers => -(pi * x).sin(),
            Self::AllenCahn => x * x * (pi * x).cos(),
        }
    }

    /// Width of the first network layer.
    pub(crate) fn input_width(self) -> usize {
        match self {
            Self::Burgers => 2,
            Self::AllenCahn => 3,
        }
    }

    /// The network input for column vectors `x` and `t` of shape `[n, 1]`.
    pub(crate) fn features(self, x: &TracedTensor, t: &TracedTensor) -> Result<TracedTensor> {
        let n = column_len(x)?;
        match self {
            Self::Burgers => TracedTensor::stack(&[x, t], 1)?.reshape([n, 2]),
            Self::AllenCahn => {
                let angle = x.scale_real(std::f64::consts::PI)?;
                TracedTensor::stack(&[&angle.cos()?, &angle.sin()?, t], 1)?.reshape([n, 3])
            }
        }
    }

    /// The PDE residual at the collocation points `x`, `t`, where `u` is the
    /// network output there and `ones` a column of ones of the same shape.
    pub(crate) fn residual(
        self,
        u: &TracedTensor,
        x: &TracedTensor,
        t: &TracedTensor,
        ones: &TracedTensor,
    ) -> Result<TracedTensor> {
        let u_t = u.jvp(t, ones)?;
        let u_x = u.jvp(x, ones)?;
        let u_xx = u_x.jvp(x, ones)?;
        match self {
            Self::Burgers => u_t.add(&u.mul(&u_x)?)?.sub(&u_xx.scale_real(NU)?),
            Self::AllenCahn => {
                let cubic = u.mul(u)?.mul(u)?;
                u_t.sub(&u_xx.scale_real(EPSILON)?)?
                    .add(&cubic.sub(u)?.scale_real(5.0)?)
            }
        }
    }
}

fn column_len(x: &TracedTensor) -> Result<usize> {
    x.try_concrete_shape()
        .map(|shape| shape[0])
        .ok_or_else(|| Error::Internal("collocation inputs need a concrete shape".into()))
}

#[cfg(test)]
mod tests;
