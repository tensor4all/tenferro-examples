//! Translation-invariant RBM wavefunction and the two lattice Hamiltonians.
//!
//! The ansatz has `features` filters `W^(f)`, each a vector of one weight per
//! lattice site, tied across all lattice translations `t`:
//!
//! ```text
//! log psi(s) = sum_f sum_t log(2 cosh(theta_{f,t})),
//! theta_{f,t} = sum_i W^(f)_i s_{i+t}.
//! ```
//!
//! Visible and hidden biases are zero and the weights are real, so every
//! amplitude is positive. Sites are numbered `i = x * ly + y` with periodic
//! boundaries in both directions; a chain has `ly = 1`.

use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(test)]
mod tests;

/// Lattice Hamiltonian, written with Pauli matrices as in Carleo and Troyer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hamiltonian {
    /// Critical transverse-field Ising: `H = -sum_<ij> sz_i sz_j - sum_i sx_i`.
    Ising,
    /// Antiferromagnetic Heisenberg: `H = sum_<ij> sigma_i . sigma_j`.
    ///
    /// The model works in the Marshall-rotated basis, where the ground state
    /// of a bipartite lattice has positive amplitudes, and in the sector of
    /// zero total magnetization.
    Heisenberg,
}

#[derive(Clone)]
pub(crate) struct Model {
    pub(crate) lx: usize,
    pub(crate) ly: usize,
    pub(crate) features: usize,
    pub(crate) hamiltonian: Hamiltonian,
    /// Filter weights; `weights[f * n + i]` is `W^(f)_i`.
    pub(crate) weights: Vec<f64>,
    /// `shift[i * n + t]` is the site reached from site `i` by translation `t`.
    shift: Vec<usize>,
    /// `inverse[i * n + t]` is the site reached from site `i` by translation `-t`.
    inverse: Vec<usize>,
    /// Nearest-neighbor bonds, each listed once.
    pub(crate) bonds: Vec<(usize, usize)>,
}

impl Model {
    /// Build the model with weights drawn uniformly from `[-0.02, 0.02)`.
    pub(crate) fn new(
        lx: usize,
        ly: usize,
        features: usize,
        hamiltonian: Hamiltonian,
        seed: u64,
    ) -> Self {
        let n = lx * ly;
        let mut rng = StdRng::seed_from_u64(seed);
        let shift = (0..n)
            .flat_map(|i| {
                (0..n).map(move |t| ((i / ly + t / ly) % lx) * ly + (i % ly + t % ly) % ly)
            })
            .collect();
        let inverse = (0..n)
            .flat_map(|i| {
                (0..n)
                    .map(move |t| ((i / ly + lx - t / ly) % lx) * ly + (i % ly + ly - t % ly) % ly)
            })
            .collect();
        let mut bonds = Vec::new();
        for x in 0..lx {
            for y in 0..ly {
                let i = x * ly + y;
                // A periodic length of 2 would otherwise list the same bond twice.
                if lx > 2 || x == 0 {
                    bonds.push((i, ((x + 1) % lx) * ly + y));
                }
                if ly > 1 && (ly > 2 || y == 0) {
                    bonds.push((i, x * ly + (y + 1) % ly));
                }
            }
        }
        Self {
            lx,
            ly,
            features,
            hamiltonian,
            shift,
            inverse,
            bonds,
            weights: (0..features * n)
                .map(|_| rng.gen_range(-0.02..0.02))
                .collect(),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.lx * self.ly
    }

    /// Site reached from site `i` by translation `t`.
    pub(crate) fn translate(&self, i: usize, t: usize) -> usize {
        self.shift[i * self.n() + t]
    }

    /// Hidden-unit inputs `theta_{f,t}`, stored at `f * n + t`.
    pub(crate) fn fields(&self, spins: &[f64]) -> Vec<f64> {
        let n = self.n();
        let mut fields = vec![0.0; self.features * n];
        for (f, weights) in self.weights.chunks(n).enumerate() {
            for (i, &weight) in weights.iter().enumerate() {
                for t in 0..n {
                    fields[f * n + t] += weight * spins[self.shift[i * n + t]];
                }
            }
        }
        fields
    }

    pub(crate) fn log_psi(&self, spins: &[f64]) -> f64 {
        log_fields(&self.fields(spins))
    }

    /// Log-derivatives `O_k(s) = d log psi(s) / d W_k`, in the layout of `weights`.
    pub(crate) fn derivatives(&self, spins: &[f64], fields: &[f64]) -> Vec<f64> {
        let n = self.n();
        let tanh: Vec<f64> = fields.iter().map(|x| x.tanh()).collect();
        let mut out = vec![0.0; self.weights.len()];
        for f in 0..self.features {
            for i in 0..n {
                out[f * n + i] = tanh[f * n..(f + 1) * n]
                    .iter()
                    .zip(&self.shift[i * n..(i + 1) * n])
                    .map(|(t, &site)| t * spins[site])
                    .sum();
            }
        }
        out
    }

    /// `log psi` of the configuration with the spins at `flips` reversed,
    /// computed from the cached `fields` of `spins`.
    pub(crate) fn proposed_log(&self, spins: &[f64], fields: &[f64], flips: &[usize]) -> f64 {
        let n = self.n();
        let mut result = 0.0;
        for f in 0..self.features {
            let weights = &self.weights[f * n..(f + 1) * n];
            for t in 0..n {
                let delta: f64 = flips
                    .iter()
                    .map(|&i| -2.0 * spins[i] * weights[self.inverse[i * n + t]])
                    .sum();
                result += log_two_cosh(fields[f * n + t] + delta);
            }
        }
        result
    }

    /// Reverse the spins at `flips` and update the cached `fields` to match.
    pub(crate) fn apply_flips(&self, spins: &mut [f64], fields: &mut [f64], flips: &[usize]) {
        let n = self.n();
        for &i in flips {
            for f in 0..self.features {
                for t in 0..n {
                    fields[f * n + t] -=
                        2.0 * spins[i] * self.weights[f * n + self.inverse[i * n + t]];
                }
            }
            spins[i] *= -1.0;
        }
    }

    /// Local energy `E_loc(s) = <s|H|psi> / <s|psi>`.
    pub(crate) fn local_energy(&self, spins: &[f64], fields: &[f64]) -> f64 {
        let base = log_fields(fields);
        let zz: f64 = self.bonds.iter().map(|&(i, j)| spins[i] * spins[j]).sum();
        match self.hamiltonian {
            // In the Marshall-rotated basis, each antiparallel bond connects to
            // the exchanged configuration with matrix element -2.
            Hamiltonian::Heisenberg => {
                zz - 2.0
                    * self
                        .bonds
                        .iter()
                        .filter(|&&(i, j)| spins[i] != spins[j])
                        .map(|&(i, j)| (self.proposed_log(spins, fields, &[i, j]) - base).exp())
                        .sum::<f64>()
            }
            Hamiltonian::Ising => {
                -zz - (0..self.n())
                    .map(|i| (self.proposed_log(spins, fields, &[i]) - base).exp())
                    .sum::<f64>()
            }
        }
    }
}

/// `log(2 cosh(z))`, written so that it does not overflow for large `|z|`.
pub(crate) fn log_two_cosh(z: f64) -> f64 {
    z.abs() + (-2.0 * z.abs()).exp().ln_1p()
}

pub(crate) fn log_fields(fields: &[f64]) -> f64 {
    fields.iter().copied().map(log_two_cosh).sum()
}
