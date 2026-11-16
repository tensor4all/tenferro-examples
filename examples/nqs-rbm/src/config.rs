//! Command-line options and their per-system defaults.

use crate::model::Hamiltonian;
use crate::reference::check_exact_size;
use crate::sampler::CHAINS;
use std::path::PathBuf;

#[cfg(test)]
mod tests;

/// Validation energies are blocked in groups of this many sweeps to estimate
/// the standard error.
pub(crate) const BLOCK: usize = 64;

pub(crate) const USAGE: &str = "\
usage: nqs_rbm [options]

  --system NAME            ising-chain (default), heisenberg-chain, heisenberg-square
  --size L                 sites per side (chains: 80, square: 10)
  --alpha A                filters, i.e. hidden-unit density (chains: 4, square: 16)
  --steps S                SR iterations (chains: 300, square: 200)
  --samples M              samples per SR step, a multiple of 8 (chains: 1024, square: 512)
  --seed K                 random seed (42, 43, 44 for the three systems)
  --sr-shift EPS           identity shift of the SR matrix (chains: 1e-7, square: 1e-3)
  --max-weight-step D      bound on any single weight change, 0 = none (chains: 0, square: 0.002)
  --validation-sweeps V    validation samples per chain, a multiple of 64 (2048)
  --init-weights FILE      start from a weights.csv written by an earlier run
  --output-dir DIR         write history.csv, weights.csv, validation_energies.csv, summary.json
  --energy-png FILE        plot the training energy
  --filters-png FILE       plot the trained filters
  --exact                  diagonalize the Hamiltonian and evaluate the trained state exactly
                           (small lattices only)
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum System {
    IsingChain,
    HeisenbergChain,
    HeisenbergSquare,
}

impl System {
    pub(crate) fn name(self) -> &'static str {
        match self {
            System::IsingChain => "ising-chain",
            System::HeisenbergChain => "heisenberg-chain",
            System::HeisenbergSquare => "heisenberg-square",
        }
    }

    pub(crate) fn hamiltonian(self) -> Hamiltonian {
        match self {
            System::IsingChain => Hamiltonian::Ising,
            System::HeisenbergChain | System::HeisenbergSquare => Hamiltonian::Heisenberg,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Config {
    pub(crate) system: System,
    pub(crate) size: usize,
    pub(crate) alpha: usize,
    pub(crate) steps: usize,
    pub(crate) samples: usize,
    pub(crate) seed: u64,
    pub(crate) sr_shift: f64,
    pub(crate) max_weight_step: f64,
    pub(crate) validation_sweeps: usize,
    pub(crate) init_weights: Option<PathBuf>,
    pub(crate) output_dir: Option<PathBuf>,
    pub(crate) energy_png: Option<String>,
    pub(crate) filters_png: Option<String>,
    pub(crate) exact: bool,
}

impl Config {
    /// Defaults for one of the three systems of the paper's Figure 2.
    pub(crate) fn defaults(system: System) -> Self {
        let chain = system != System::HeisenbergSquare;
        Self {
            system,
            size: if chain { 80 } else { 10 },
            alpha: if chain { 4 } else { 16 },
            steps: if chain { 300 } else { 200 },
            samples: if chain { 1024 } else { 512 },
            seed: match system {
                System::IsingChain => 42,
                System::HeisenbergChain => 43,
                System::HeisenbergSquare => 44,
            },
            sr_shift: if chain { 1e-7 } else { 1e-3 },
            max_weight_step: if chain { 0.0 } else { 0.002 },
            validation_sweeps: 2048,
            init_weights: None,
            output_dir: None,
            energy_png: None,
            filters_png: None,
            exact: false,
        }
    }

    /// Parse the arguments that follow the program name.
    pub(crate) fn from_args(args: &[String]) -> Result<Self, String> {
        let mut pairs = Vec::new();
        let mut exact = false;
        let mut iter = args.iter();
        while let Some(flag) = iter.next() {
            if flag == "--exact" {
                exact = true;
                continue;
            }
            if !flag.starts_with("--") {
                return Err(format!("unexpected argument `{flag}`"));
            }
            let value = iter
                .next()
                .ok_or_else(|| format!("missing value after `{flag}`"))?;
            if pairs.iter().any(|(f, _)| f == flag) {
                return Err(format!("`{flag}` is given more than once"));
            }
            pairs.push((flag.clone(), value.clone()));
        }
        let system = match pairs.iter().find(|(f, _)| f == "--system") {
            None => System::IsingChain,
            Some((_, name)) => match name.as_str() {
                "ising-chain" => System::IsingChain,
                "heisenberg-chain" => System::HeisenbergChain,
                "heisenberg-square" => System::HeisenbergSquare,
                _ => return Err(format!("unknown system `{name}`")),
            },
        };
        let mut config = Self::defaults(system);
        config.exact = exact;
        for (flag, value) in &pairs {
            match flag.as_str() {
                "--system" => {}
                "--size" => config.size = parse(flag, value)?,
                "--alpha" => config.alpha = parse(flag, value)?,
                "--steps" => config.steps = parse(flag, value)?,
                "--samples" => config.samples = parse(flag, value)?,
                "--seed" => config.seed = parse(flag, value)?,
                "--sr-shift" => config.sr_shift = parse(flag, value)?,
                "--max-weight-step" => config.max_weight_step = parse(flag, value)?,
                "--validation-sweeps" => config.validation_sweeps = parse(flag, value)?,
                "--init-weights" => config.init_weights = Some(value.into()),
                "--output-dir" => config.output_dir = Some(value.into()),
                "--energy-png" => config.energy_png = Some(value.clone()),
                "--filters-png" => config.filters_png = Some(value.clone()),
                _ => return Err(format!("unknown option `{flag}`")),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        let heisenberg = self.system.hamiltonian() == Hamiltonian::Heisenberg;
        if self.system == System::HeisenbergSquare {
            if self.size < 2 || !self.size.is_multiple_of(2) {
                return Err(format!(
                    "--size must be even and at least 2 for the square lattice, got {}",
                    self.size
                ));
            }
        } else if self.size < 4 || (heisenberg && !self.size.is_multiple_of(2)) {
            return Err(format!(
                "--size must be at least 4{} for a chain, got {}",
                if heisenberg { " and even" } else { "" },
                self.size
            ));
        }
        if self.alpha == 0 || self.steps == 0 {
            return Err("--alpha and --steps must be positive".into());
        }
        if self.samples == 0 || !self.samples.is_multiple_of(CHAINS) {
            return Err(format!(
                "--samples must be a positive multiple of {CHAINS}, got {}",
                self.samples
            ));
        }
        if !(self.sr_shift.is_finite() && self.sr_shift > 0.0) {
            return Err(format!(
                "--sr-shift must be positive and finite, got {}",
                self.sr_shift
            ));
        }
        if !(self.max_weight_step.is_finite() && self.max_weight_step >= 0.0) {
            return Err(format!(
                "--max-weight-step must be non-negative and finite, got {}",
                self.max_weight_step
            ));
        }
        if self.validation_sweeps < 2 * BLOCK || !self.validation_sweeps.is_multiple_of(BLOCK) {
            return Err(format!(
                "--validation-sweeps must be a multiple of {BLOCK} and at least {}, got {}",
                2 * BLOCK,
                self.validation_sweeps
            ));
        }
        if self.exact {
            check_exact_size(self.system.hamiltonian(), self.lx() * self.ly())?;
        }
        Ok(())
    }

    pub(crate) fn lx(&self) -> usize {
        self.size
    }

    pub(crate) fn ly(&self) -> usize {
        if self.system == System::HeisenbergSquare {
            self.size
        } else {
            1
        }
    }
}

fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("cannot parse `{value}` as the value of `{flag}`"))
}
