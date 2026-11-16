//! Metropolis sampling of `|psi(s)|^2` with persistent Markov chains.
//!
//! A sweep attempts `N` single-spin flips for the Ising model, or `N/2`
//! exchanges of two opposite spins for the Heisenberg model, which keeps the
//! chain in the zero-magnetization sector. Acceptance ratios use the cached
//! hidden-unit inputs, so each proposal costs `O(features * N)`.

use crate::model::{log_fields, Hamiltonian, Model};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;

#[cfg(test)]
mod tests;

/// Number of independent Markov chains, sampled in parallel.
pub(crate) const CHAINS: usize = 8;

/// Probability of skipping a proposal. Without it, a chain that accepts every
/// single flip alternates magnetization parity on every sweep and never
/// samples half of the configurations at even sweep counts.
const LAZY_PROBABILITY: f64 = 0.05;

pub(crate) struct Chain {
    pub(crate) spins: Vec<f64>,
    rng: StdRng,
}

/// Energies, and optionally log-derivatives, of one sample per sweep.
pub(crate) struct Batch {
    pub(crate) energies: Vec<f64>,
    /// Sample-major: the derivatives of sample `s` are
    /// `gradients[s * p..(s + 1) * p]` for `p` parameters.
    pub(crate) gradients: Vec<f64>,
    pub(crate) accepted: usize,
    pub(crate) trials: usize,
}

impl Batch {
    fn empty() -> Self {
        Self {
            energies: Vec::new(),
            gradients: Vec::new(),
            accepted: 0,
            trials: 0,
        }
    }

    pub(crate) fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.trials as f64
    }
}

impl Chain {
    /// Random starting configuration; half up, half down for Heisenberg.
    pub(crate) fn new(model: &Model, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = model.n();
        let mut spins: Vec<f64> = (0..n)
            .map(|i| match model.hamiltonian {
                Hamiltonian::Heisenberg if i < n / 2 => -1.0,
                Hamiltonian::Heisenberg => 1.0,
                Hamiltonian::Ising if rng.gen_bool(0.5) => -1.0,
                Hamiltonian::Ising => 1.0,
            })
            .collect();
        spins.shuffle(&mut rng);
        Self { spins, rng }
    }

    /// One sweep; returns `(accepted, attempted)` proposal counts.
    pub(crate) fn sweep(
        &mut self,
        model: &Model,
        fields: &mut [f64],
        log_value: &mut f64,
    ) -> (usize, usize) {
        let heisenberg = model.hamiltonian == Hamiltonian::Heisenberg;
        let trials = if heisenberg { model.n() / 2 } else { model.n() };
        let mut accepted = 0;
        for _ in 0..trials {
            if self.rng.gen_bool(LAZY_PROBABILITY) {
                continue;
            }
            let i = self.rng.gen_range(0..model.n());
            let mut flips = [i, i];
            let count = if heisenberg {
                loop {
                    let j = self.rng.gen_range(0..model.n());
                    if self.spins[j] != self.spins[i] {
                        flips[1] = j;
                        break;
                    }
                }
                2
            } else {
                1
            };
            let proposed = model.proposed_log(&self.spins, fields, &flips[..count]);
            if self.rng.gen_range(0.0..1.0_f64).ln() < 2.0 * (proposed - *log_value) {
                model.apply_flips(&mut self.spins, fields, &flips[..count]);
                *log_value = proposed;
                accepted += 1;
            }
        }
        (accepted, trials)
    }

    /// Run `burn` sweeps, then record one sample after each of `count` sweeps.
    fn sample(&mut self, model: &Model, count: usize, burn: usize, derivatives: bool) -> Batch {
        let mut fields = model.fields(&self.spins);
        let mut log_value = log_fields(&fields);
        let mut batch = Batch::empty();
        batch.energies.reserve(count);
        if derivatives {
            batch.gradients.reserve(count * model.weights.len());
        }
        for _ in 0..burn {
            self.sweep(model, &mut fields, &mut log_value);
        }
        for _ in 0..count {
            let (accepted, trials) = self.sweep(model, &mut fields, &mut log_value);
            batch.accepted += accepted;
            batch.trials += trials;
            batch
                .energies
                .push(model.local_energy(&self.spins, &fields));
            if derivatives {
                batch
                    .gradients
                    .extend(model.derivatives(&self.spins, &fields));
            }
        }
        batch
    }
}

/// Sample every chain in parallel and concatenate the results chain by chain,
/// so samples `c * per_chain..(c + 1) * per_chain` come from chain `c`.
pub(crate) fn sample(
    chains: &mut [Chain],
    model: &Model,
    per_chain: usize,
    burn: usize,
    derivatives: bool,
) -> Batch {
    let batches: Vec<Batch> = chains
        .par_iter_mut()
        .map(|c| c.sample(model, per_chain, burn, derivatives))
        .collect();
    let mut out = Batch::empty();
    for batch in batches {
        out.energies.extend(batch.energies);
        out.gradients.extend(batch.gradients);
        out.accepted += batch.accepted;
        out.trials += batch.trials;
    }
    out
}

/// Mean and population variance.
pub(crate) fn mean_variance(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, variance)
}

/// Standard error of the mean, estimated from the means of consecutive blocks
/// of `block` samples. `values.len()` must be a multiple of `block` with at
/// least two blocks, so that a block never spans two chains.
pub(crate) fn block_standard_error(values: &[f64], block: usize) -> f64 {
    assert!(
        values.len().is_multiple_of(block) && values.len() >= 2 * block,
        "{} samples do not form at least two blocks of {block}",
        values.len()
    );
    let means: Vec<f64> = values.chunks(block).map(|x| mean_variance(x).0).collect();
    let (_, variance) = mean_variance(&means);
    (variance / (means.len() - 1) as f64).sqrt()
}
