//! Seeded periodic Gaussian fields and viscous Burgers reference targets.
//!
//! On `x_j = j / 1024`, sample the zero-mean covariance operator
//! `625 (-∂xx + 25 I)^(-2)`: independent cosine/sine coefficients have standard
//! deviation `sqrt(2) * 25 / ((2πm)^2 + 25)`. Only `0 < 3m < 1024` is retained.
//! The same fine input/target pair is subsampled at every model resolution.
//!
//! The Lawson integrating-factor RK4 scheme and concrete FFT conventions
//! follow this repository's `examples/nonlinear-pinn/src/reference.rs`.
//! Diffusion is integrated exactly; the quadratic term uses the 2/3 rule,
//! including projection of the initial data before any quadratic product.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOptions};
use tenferro_tensor::{Complex64, Tensor};

pub(crate) const REFERENCE_N: usize = 1024;
const NU: f64 = 0.1;
const REFERENCE_DT: f64 = 0.002;

pub(crate) struct Dataset {
    /// Contiguous 1024-point trajectories, with samples as columns.
    initial: Vec<f64>,
    final_values: Vec<f64>,
}

impl Dataset {
    /// Generate `count` independent initial/final pairs at `t = 0, 1`.
    ///
    /// # Errors
    /// Rejects an empty or unallocatable dataset, FFT failures, or non-finite
    /// reference solutions.
    pub(crate) fn generate(count: usize, seed: u64) -> Result<Self, Box<dyn Error>> {
        if count == 0 {
            return Err(
                IoError::new(ErrorKind::InvalidInput, "dataset count must be positive").into(),
            );
        }
        let size = count
            .checked_mul(REFERENCE_N)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "dataset size overflows usize"))?;
        let mut initial = Vec::new();
        let mut final_values = Vec::new();
        initial.try_reserve_exact(size)?;
        final_values.try_reserve_exact(size)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut solver = Solver::new(REFERENCE_N, REFERENCE_DT)?;
        for _ in 0..count {
            let field = sample_field(&mut solver, &mut rng)?;
            let target = solver.solve(&field, (1.0 / REFERENCE_DT).round() as usize)?;
            initial.extend_from_slice(&field);
            final_values.extend_from_slice(&target);
        }
        Ok(Self {
            initial,
            final_values,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.initial.len() / REFERENCE_N
    }

    /// Gather unscaled pairs into column-major tensors `[n, batch, 1]`.
    ///
    /// # Errors
    /// Rejects an empty batch, indices outside this dataset, and grids that
    /// do not divide the 1024-point reference grid; allocation/FFT tensor
    /// construction failures are propagated.
    pub(crate) fn batch(
        &self,
        indices: &[usize],
        n: usize,
    ) -> Result<(Tensor, Tensor), Box<dyn Error>> {
        if n == 0 || n > REFERENCE_N || !REFERENCE_N.is_multiple_of(n) {
            return Err(IoError::new(ErrorKind::InvalidInput, "grid must divide 1024").into());
        }
        if indices.is_empty() || indices.iter().any(|&i| i >= self.len()) {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "batch must contain valid sample indices",
            )
            .into());
        }
        let size = n
            .checked_mul(indices.len())
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "batch size overflows usize"))?;
        let mut input = Vec::new();
        let mut target = Vec::new();
        input.try_reserve_exact(size)?;
        target.try_reserve_exact(size)?;
        let stride = REFERENCE_N / n;
        for &sample in indices {
            let start = sample * REFERENCE_N;
            input.extend(
                self.initial[start..start + REFERENCE_N]
                    .iter()
                    .step_by(stride)
                    .copied(),
            );
            target.extend(
                self.final_values[start..start + REFERENCE_N]
                    .iter()
                    .step_by(stride)
                    .copied(),
            );
        }
        Ok((
            Tensor::from_vec_col_major(vec![n, indices.len(), 1], input)?,
            Tensor::from_vec_col_major(vec![n, indices.len(), 1], target)?,
        ))
    }

    pub(crate) fn rms(&self) -> (f64, f64) {
        let rms = |values: &[f64]| {
            (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
        };
        (rms(&self.initial), rms(&self.final_values))
    }
}

struct Solver {
    fft: FftExecutor,
    engine: CpuEngine,
    n: usize,
    kept: usize,
    dt: f64,
    e: Vec<f64>,
    e2: Vec<f64>,
}

impl Solver {
    fn new(n: usize, dt: f64) -> Result<Self, Box<dyn Error>> {
        if n < 4 || !n.is_multiple_of(2) || !dt.is_finite() || dt <= 0.0 {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "solver needs an even grid and positive finite dt",
            )
            .into());
        }
        let eigenvalue = |m: usize| -NU * (std::f64::consts::TAU * m as f64).powi(2);
        Ok(Self {
            fft: FftExecutor::default(),
            engine: CpuEngine::serial(),
            n,
            kept: (n - 1) / 3 + 1,
            dt,
            e: (0..=n / 2).map(|m| (eigenvalue(m) * dt).exp()).collect(),
            e2: (0..=n / 2)
                .map(|m| (eigenvalue(m) * dt / 2.0).exp())
                .collect(),
        })
    }

    fn inverse(&mut self, spectrum: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        Ok(self.fft.irfft_with(
            spectrum,
            &FftOptions::default().axis(0).n(Some(self.n)),
            &mut self.engine,
        )?)
    }

    fn nonlinear(&mut self, spectrum: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        // INVARIANT: solve projects the initial spectrum; every RK stage keeps
        // modes m >= kept at zero, so no high-mode input can alias into u².
        let mut squared = self.inverse(spectrum)?;
        for value in squared.as_slice_mut::<f64>()? {
            *value *= *value;
        }
        // Concrete FFTs allocate their output tensors; the executor reuses plans.
        let mut result =
            self.fft
                .rfft_with(&squared, &FftOptions::default().axis(0), &mut self.engine)?;
        let values = result.as_slice_mut::<Complex64>()?;
        for (m, value) in values[..self.kept].iter_mut().enumerate() {
            *value *= Complex64::new(0.0, -std::f64::consts::PI * m as f64);
        }
        values[self.kept..].fill(Complex64::new(0.0, 0.0));
        Ok(result)
    }

    fn solve(&mut self, initial: &[f64], steps: usize) -> Result<Vec<f64>, Box<dyn Error>> {
        if initial.len() != self.n || initial.iter().any(|v| !v.is_finite()) {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "initial field must match the grid and be finite",
            )
            .into());
        }
        let input = Tensor::from_vec_col_major(vec![self.n], initial.to_vec())?;
        let mut spectrum =
            self.fft
                .rfft_with(&input, &FftOptions::default().axis(0), &mut self.engine)?;
        spectrum.as_slice_mut::<Complex64>()?[self.kept..].fill(Complex64::new(0.0, 0.0));
        if steps > 0 {
            let dt = self.dt;
            let mut k1 = self.nonlinear(&spectrum)?;
            let u = spectrum.as_slice::<Complex64>()?;
            let k = k1.as_slice::<Complex64>()?;
            let mut stage = Tensor::from_vec_col_major(
                vec![u.len()],
                u.iter()
                    .zip(k)
                    .zip(&self.e2)
                    .map(|((&u, &k), &e2)| (u + k * (dt / 2.0)) * e2)
                    .collect(),
            )?;
            for step in 0..steps {
                let k2 = self.nonlinear(&stage)?;
                let u = spectrum.as_slice::<Complex64>()?;
                let k = k2.as_slice::<Complex64>()?;
                for (m, value) in stage.as_slice_mut::<Complex64>()?.iter_mut().enumerate() {
                    *value = u[m] * self.e2[m] + k[m] * (dt / 2.0);
                }
                let k3 = self.nonlinear(&stage)?;
                let u = spectrum.as_slice::<Complex64>()?;
                let k = k3.as_slice::<Complex64>()?;
                for (m, value) in stage.as_slice_mut::<Complex64>()?.iter_mut().enumerate() {
                    *value = u[m] * self.e[m] + k[m] * (dt * self.e2[m]);
                }
                let k4 = self.nonlinear(&stage)?;
                let a = k1.as_slice::<Complex64>()?;
                let b = k2.as_slice::<Complex64>()?;
                let c = k3.as_slice::<Complex64>()?;
                let d = k4.as_slice::<Complex64>()?;
                for (m, value) in spectrum.as_slice_mut::<Complex64>()?.iter_mut().enumerate() {
                    *value = *value * self.e[m]
                        + (a[m] * self.e[m] + (b[m] + c[m]) * (2.0 * self.e2[m]) + d[m])
                            * (dt / 6.0);
                }
                if step + 1 < steps {
                    k1 = self.nonlinear(&spectrum)?;
                    let u = spectrum.as_slice::<Complex64>()?;
                    let k = k1.as_slice::<Complex64>()?;
                    for (m, value) in stage.as_slice_mut::<Complex64>()?.iter_mut().enumerate() {
                        *value = (u[m] + k[m] * (dt / 2.0)) * self.e2[m];
                    }
                }
            }
        }
        let result = self.inverse(&spectrum)?;
        let values = result.as_slice::<f64>()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "Burgers reference became non-finite",
            )
            .into());
        }
        Ok(values.to_vec())
    }
}

fn sample_field(solver: &mut Solver, rng: &mut StdRng) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut spectrum = Vec::with_capacity(solver.n / 2 + 1);
    spectrum.push(Complex64::new(0.0, 0.0));
    for m in 1..solver.kept {
        // Box-Muller: independent standard normals from U in (0,1], V in [0,1).
        let radius = (-2.0 * (1.0 - rng.gen::<f64>()).ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.gen::<f64>();
        let sigma =
            std::f64::consts::SQRT_2 * 25.0 / ((std::f64::consts::TAU * m as f64).powi(2) + 25.0);
        let scale = solver.n as f64 * 0.5 * sigma * radius;
        spectrum.push(Complex64::new(scale * angle.cos(), -scale * angle.sin()));
    }
    spectrum.resize(solver.n / 2 + 1, Complex64::new(0.0, 0.0));
    let spectrum = Tensor::from_vec_col_major(vec![spectrum.len()], spectrum)?;
    Ok(solver.inverse(&spectrum)?.as_slice::<f64>()?.to_vec())
}

#[cfg(test)]
mod tests;
