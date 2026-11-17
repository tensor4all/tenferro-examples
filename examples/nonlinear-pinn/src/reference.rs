//! Pseudo-spectral reference solutions, computed with `tenferro-fft`.
//!
//! Both problems are periodic on `x ∈ [-1, 1)` (Burgers through the odd
//! symmetry of `-sin(pi x)`, which keeps `u(±1, t) = 0`). In Fourier space
//! `û_t = -c k² û + N(û)` with `k = pi m` and `c` the diffusion coefficient. The
//! stiff linear part is integrated exactly and the nonlinear part with the
//! Lawson integrating-factor RK4 scheme:
//!
//! ```text
//! k1 = N(û)
//! k2 = N(E2 (û + dt/2 k1))
//! k3 = N(E2 û + dt/2 k2)
//! k4 = N(E û + dt E2 k3)
//! û' = E û + dt/6 (E k1 + 2 E2 (k2 + k3) + k4),   E = exp(-c k² dt), E2 = exp(-c k² dt / 2)
//! ```
//!
//! `N` evaluates the nonlinearity on the grid (`irfft`, pointwise, `rfft`) and
//! zeroes the top third of the modes (the 2/3 rule).

use crate::problem::Problem;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm};
use tenferro_tensor::{Complex64, Tensor};

/// The reference on `n` grid points `x_j = -1 + 2 j / n`, at the times
/// `0, dt_out, 2 dt_out, ..., t_end`.
pub(crate) struct Reference {
    pub(crate) x: Vec<f64>,
    pub(crate) t: Vec<f64>,
    /// `u[s][j]`: the solution at `t[s]` and `x[j]`.
    pub(crate) u: Vec<Vec<f64>>,
}

/// Spectral solver state: the FFT executor (plans reused across steps), the
/// engine, and the per-mode factors.
struct Solver {
    fft: FftExecutor,
    engine: CpuEngine,
    n: usize,
    problem: Problem,
    /// `exp(-c k² dt)` and `exp(-c k² dt / 2)` per mode.
    e: Vec<f64>,
    e2: Vec<f64>,
    /// 1 for the modes kept by the 2/3 rule, 0 above.
    keep: Vec<f64>,
}

impl Solver {
    fn new(problem: Problem, n: usize, dt: f64) -> Self {
        let modes = n / 2 + 1;
        let c = problem.diffusion();
        let k2 = |m: usize| (std::f64::consts::PI * m as f64).powi(2);
        Self {
            fft: FftExecutor::default(),
            engine: CpuEngine::serial(),
            n,
            problem,
            e: (0..modes).map(|m| (-c * k2(m) * dt).exp()).collect(),
            e2: (0..modes).map(|m| (-c * k2(m) * dt / 2.0).exp()).collect(),
            keep: (0..modes)
                .map(|m| if 3 * m < n { 1.0 } else { 0.0 })
                .collect(),
        }
    }

    fn forward(&mut self, u: Vec<f64>) -> Vec<Complex64> {
        let u = Tensor::from_vec_col_major(vec![self.n], u).expect("grid vector");
        let spectrum = self
            .fft
            .rfft(&u, None, 0, FftNorm::Backward, &mut self.engine)
            .expect("rfft of the grid values");
        spectrum
            .as_slice::<Complex64>()
            .expect("complex spectrum")
            .to_vec()
    }

    fn inverse(&mut self, spectrum: &[Complex64]) -> Vec<f64> {
        let spectrum = Tensor::from_vec_col_major(vec![spectrum.len()], spectrum.to_vec())
            .expect("spectrum vector");
        let u = self
            .fft
            .irfft(
                &spectrum,
                Some(self.n),
                0,
                FftNorm::Backward,
                &mut self.engine,
            )
            .expect("irfft of the spectrum");
        u.as_slice::<f64>().expect("real grid values").to_vec()
    }

    /// The dealiased nonlinear term `N(û)`.
    fn nonlinear(&mut self, spectrum: &[Complex64]) -> Vec<Complex64> {
        let u = self.inverse(spectrum);
        let pi = std::f64::consts::PI;
        match self.problem {
            // -(u²/2)_x  ->  -i k FFT(u²) / 2
            Problem::Burgers => {
                let squared = self.forward(u.iter().map(|v| v * v).collect());
                squared
                    .iter()
                    .enumerate()
                    .map(|(m, s)| Complex64::new(0.0, -0.5 * pi * m as f64) * s * self.keep[m])
                    .collect()
            }
            // 5 u - 5 u³
            Problem::AllenCahn => {
                let reaction = self.forward(u.iter().map(|v| 5.0 * v - 5.0 * v * v * v).collect());
                reaction
                    .iter()
                    .enumerate()
                    .map(|(m, s)| s * self.keep[m])
                    .collect()
            }
        }
    }

    fn step(&mut self, u: &[Complex64], dt: f64) -> Vec<Complex64> {
        let combine = |a: &[Complex64],
                       b: &[Complex64],
                       f: &dyn Fn(usize, Complex64, Complex64) -> Complex64| {
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(m, (&x, &y))| f(m, x, y))
                .collect::<Vec<_>>()
        };
        let (e, e2) = (self.e.clone(), self.e2.clone());
        let k1 = self.nonlinear(u);
        let k2 = self.nonlinear(&combine(u, &k1, &|m, u, k| (u + k * (dt / 2.0)) * e2[m]));
        let k3 = self.nonlinear(&combine(u, &k2, &|m, u, k| u * e2[m] + k * (dt / 2.0)));
        let k4 = self.nonlinear(&combine(u, &k3, &|m, u, k| u * e[m] + k * (dt * e2[m])));
        (0..u.len())
            .map(|m| {
                u[m] * e[m] + (k1[m] * e[m] + (k2[m] + k3[m]) * (2.0 * e2[m]) + k4[m]) * (dt / 6.0)
            })
            .collect()
    }
}

/// Solve `problem` on `n` grid points with step `dt`, recording every
/// `dt_out` up to `t_end`. `dt_out` must be a multiple of `dt`.
pub(crate) fn solve(problem: Problem, n: usize, dt: f64, dt_out: f64, t_end: f64) -> Reference {
    let per_output = (dt_out / dt).round() as usize;
    assert!(
        ((per_output as f64) * dt - dt_out).abs() < 1e-12,
        "dt_out must be a multiple of dt"
    );
    let outputs = (t_end / dt_out).round() as usize;
    let x: Vec<f64> = (0..n).map(|j| -1.0 + 2.0 * j as f64 / n as f64).collect();
    let mut solver = Solver::new(problem, n, dt);
    let mut spectrum = solver.forward(x.iter().map(|&x| problem.initial(x)).collect());
    let mut t = vec![0.0];
    let mut u = vec![solver.inverse(&spectrum)];
    for s in 1..=outputs {
        for _ in 0..per_output {
            spectrum = solver.step(&spectrum, dt);
        }
        t.push(s as f64 * dt_out);
        u.push(solver.inverse(&spectrum));
    }
    Reference { x, t, u }
}

#[cfg(test)]
mod tests;
