//! DMFT self-consistency with the IPT impurity solver on the Matsubara axis,
//! for the half-filled Hubbard model on the Bethe lattice
//! (Georges et al., Rev. Mod. Phys. 68, 13 (1996), Sec. VI.B.2).
//!
//! Energies are in units of the half-bandwidth `D = 2t = 1`.
//! A Matsubara function is stored for the `M` positive frequencies
//! `ω_n = (2n + 1)π/β`, `n = 0, …, M - 1`, as a `Complex64` tensor of shape `[M]`;
//! the negative ones follow from `F(-iω_n) = F(iω_n)*`.
//! An imaginary-time function is stored on `N = 2M` points `τ_k = kβ/N`.

use std::error::Error;
use std::f64::consts::PI;

use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm, FftOptions};
use tenferro_runtime::CoreOps;
use tenferro_tensor::{Complex64, SliceConfig, Tensor};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Squared hopping `t² = D²/4` of the Bethe lattice with `D = 1`.
const T2: f64 = 0.25;

/// Matsubara grid with the constant tensors of the two Fourier transforms.
pub struct Grid {
    beta: f64,
    n_iw: usize,
    /// `iω_n` on the positive frequencies, `[M]`.
    iw: Tensor,
    /// `1/(iω_n)` on all `N` frequencies in FFT order (`n = 0, …, M - 1, -M, …, -1`), `[N]`.
    inv_iw_all: Tensor,
    /// `e^{-iπk/N}/β`, `[N]`.
    to_tau_phase: Tensor,
    /// `Δτ e^{iπk/N}`, `[N]`.
    to_iw_phase: Tensor,
    /// Rank-0 constants, broadcast by the element-wise operations.
    one: Tensor,
    half: Tensor,
    /// Reused across every transform, so the FFT plans are built once.
    fft: FftExecutor,
}

/// The converged (or last) iterate of [`Grid::solve`].
pub struct Solution {
    pub g: Tensor,
    pub sigma: Tensor,
    pub converged: bool,
}

/// Largest `|G_new(iω_n) - G(iω_n)|` accepted as converged.
const TOLERANCE: f64 = 1e-9;
/// Weight of the new iterate: `G ← MIX G_new + (1 - MIX) G`.
const MIX: f64 = 0.5;

impl Grid {
    pub fn new(beta: f64, n_iw: usize) -> Result<Self> {
        let n = 2 * n_iw;
        let dtau = beta / n as f64;
        let iw: Vec<Complex64> = (0..n_iw)
            .map(|k| Complex64::new(0.0, matsubara(beta, k)))
            .collect();
        let inv_iw_all = (0..n)
            .map(|k| {
                let index = if k < n_iw {
                    k as i64
                } else {
                    k as i64 - n as i64
                };
                Complex64::new(0.0, -1.0 / fermionic(beta, index))
            })
            .collect();
        let phase = |k: usize| PI * k as f64 / n as f64;
        let to_tau_phase = (0..n)
            .map(|k| Complex64::from_polar(1.0 / beta, -phase(k)))
            .collect();
        let to_iw_phase = (0..n)
            .map(|k| Complex64::from_polar(dtau, phase(k)))
            .collect();
        Ok(Self {
            beta,
            n_iw,
            iw: Tensor::from_vec_col_major([n_iw], iw)?,
            inv_iw_all: Tensor::from_vec_col_major([n], inv_iw_all)?,
            to_tau_phase: Tensor::from_vec_col_major([n], to_tau_phase)?,
            to_iw_phase: Tensor::from_vec_col_major([n], to_iw_phase)?,
            one: Tensor::from_vec_col_major([0_usize; 0], vec![Complex64::new(1.0, 0.0)])?,
            half: Tensor::from_vec_col_major([0_usize; 0], vec![Complex64::new(0.5, 0.0)])?,
            fft: FftExecutor::default(),
        })
    }

    /// `F(τ_k) = (1/β) Σ_n e^{-iω_n τ_k} F(iω_n)` for a function with the
    /// tail `1/(iω_n)`, which is subtracted before the transform and added back as `-1/2`.
    ///
    /// With `ω_n τ_k = 2π nk/N + πk/N`, the sum is one forward FFT over the
    /// `N` frequencies in FFT order, followed by the phase `e^{-iπk/N}/β`.
    /// The result is real and returned as `Complex64` with zero imaginary part.
    pub fn iw_to_tau(&mut self, f_iw: &Tensor, engine: &mut CpuEngine) -> Result<Tensor> {
        let negative = f_iw.conj(engine)?.reverse(&[0], engine)?;
        let all = Tensor::concatenate(&[f_iw, &negative], 0, engine)?;
        let all = all.sub(&self.inv_iw_all, engine)?;
        let sum = self.fft.fft_with(&all, &FftOptions::default(), engine)?;
        let f_tau = sum.mul(&self.to_tau_phase, engine)?;
        let real = f_tau
            .add(&f_tau.conj(engine)?, engine)?
            .scale_real(0.5, engine)?;
        Ok(real.sub(&self.half, engine)?)
    }

    /// `F(iω_n) = ∫_0^β dτ e^{iω_n τ} F(τ)` for a particle-hole symmetric
    /// `F(τ) = F(β - τ)`, whose transform is purely imaginary.
    ///
    /// The trapezoidal sum `Δτ Σ_k e^{iω_n τ_k} F(τ_k)` is one unnormalized
    /// inverse FFT; its end-point correction is real and drops out of the
    /// imaginary part. The leading Euler-Maclaurin term,
    /// `iω_n Δτ²/12 (F(0⁺) + F(β⁻)) = iω_n Δτ²/6 F(0⁺)`, is added,
    /// so the error is `O(Δτ⁴)` instead of `O(Δτ²)`.
    pub fn tau_to_iw(&mut self, f_tau: &Tensor, engine: &mut CpuEngine) -> Result<Tensor> {
        let phased = f_tau.mul(&self.to_iw_phase, engine)?;
        let options = FftOptions::default().norm(FftNorm::Forward);
        let sum = self.fft.ifft_with(&phased, &options, engine)?;
        let positive = SliceConfig {
            starts: vec![0],
            limits: vec![self.n_iw],
            strides: vec![1],
        };
        let f_iw = sum.slice(positive, engine)?;
        let imaginary = f_iw
            .sub(&f_iw.conj(engine)?, engine)?
            .scale_real(0.5, engine)?;
        let f_0 = f_tau.as_slice::<Complex64>()?[0];
        let dtau = self.beta / (2 * self.n_iw) as f64;
        let correction = self.iw.scale_complex(f_0 * dtau * dtau / 6.0, engine)?;
        Ok(imaginary.add(&correction, engine)?)
    }

    /// One IPT step: the Weiss field `G0 = 1/(iω_n - t² G)`,
    /// `Σ(τ) = U² G0(τ)³`, and the new `G = 1/(G0^{-1} - Σ)`.
    pub fn step(&mut self, u: f64, g: &Tensor, engine: &mut CpuEngine) -> Result<(Tensor, Tensor)> {
        let weiss_inv = self.iw.sub(&g.scale_real(T2, engine)?, engine)?;
        let weiss = self.one.div(&weiss_inv, engine)?;
        let weiss_tau = self.iw_to_tau(&weiss, engine)?;
        let sigma_tau = weiss_tau
            .mul(&weiss_tau, engine)?
            .mul(&weiss_tau, engine)?
            .scale_real(u * u, engine)?;
        let sigma = self.tau_to_iw(&sigma_tau, engine)?;
        let g_new = self.one.div(&weiss_inv.sub(&sigma, engine)?, engine)?;
        Ok((g_new, sigma))
    }

    /// Iterate [`Self::step`] from `g` with linear mixing until converged.
    pub fn solve(
        &mut self,
        u: f64,
        mut g: Tensor,
        max_iterations: usize,
        engine: &mut CpuEngine,
    ) -> Result<Solution> {
        let mut sigma =
            Tensor::from_vec_col_major([self.n_iw], vec![Complex64::default(); self.n_iw])?;
        for _ in 0..max_iterations {
            let (g_new, sigma_new) = self.step(u, &g, engine)?;
            let mixed = g_new
                .scale_real(MIX, engine)?
                .add(&g.scale_real(1.0 - MIX, engine)?, engine)?;
            let change = max_abs_difference(&mixed, &g)?;
            g = mixed;
            sigma = sigma_new;
            if change < TOLERANCE {
                return Ok(Solution {
                    g,
                    sigma,
                    converged: true,
                });
            }
        }
        Ok(Solution {
            g,
            sigma,
            converged: false,
        })
    }

    /// Quasiparticle weight `Z = 1/(1 - Im Σ(iω_0)/ω_0)` from the lowest frequency.
    pub fn quasiparticle_weight(&self, sigma: &Tensor) -> Result<f64> {
        let sigma_0 = sigma.as_slice::<Complex64>()?[0];
        Ok(1.0 / (1.0 - sigma_0.im / matsubara(self.beta, 0)))
    }

    /// The `U = 0` Green's function of the semicircular density of states,
    /// `G(iω_n) = 2(iω_n - i sgn(ω_n) √(ω_n² + D²))/D²`: the metallic start.
    pub fn semicircle(&self) -> Result<Tensor> {
        self.host_function(|w| Complex64::new(0.0, 2.0 * (w - (w * w + 1.0).sqrt())))
    }

    /// The atomic-limit Green's function `G(iω_n) = iω_n/((iω_n)² - U²/4)`:
    /// the insulating start.
    pub fn atomic(&self, u: f64) -> Result<Tensor> {
        self.host_function(|w| Complex64::new(0.0, -w / (w * w + u * u / 4.0)))
    }

    fn host_function(&self, f: impl Fn(f64) -> Complex64) -> Result<Tensor> {
        let values = (0..self.n_iw).map(|k| f(matsubara(self.beta, k))).collect();
        Ok(Tensor::from_vec_col_major([self.n_iw], values)?)
    }
}

/// `ω_n = (2n + 1)π/β` for `n ≥ 0`.
pub fn matsubara(beta: f64, n: usize) -> f64 {
    fermionic(beta, n as i64)
}

fn fermionic(beta: f64, n: i64) -> f64 {
    (2 * n + 1) as f64 * PI / beta
}

/// Largest `|a_n - b_n|` of two Matsubara functions.
pub fn max_abs_difference(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a.as_slice::<Complex64>()?;
    let b = b.as_slice::<Complex64>()?;
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max))
}

#[cfg(test)]
mod tests;
