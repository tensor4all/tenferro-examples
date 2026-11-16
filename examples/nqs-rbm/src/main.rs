//! Neural-network quantum states for the ground states of lattice spin models,
//! following Carleo and Troyer, Science 355, 602 (2017),
//! <https://arxiv.org/abs/1606.02318>.
//!
//! A translation-invariant restricted Boltzmann machine is trained by
//! variational Monte Carlo with stochastic reconfiguration for one of the three
//! systems of the paper's Figure 2: the critical transverse-field Ising chain,
//! the Heisenberg chain, and the Heisenberg model on the square lattice.
//!
//! ```bash
//! cargo run --release -p nqs_rbm -- --system heisenberg-chain --steps 100
//! ```
//!
//! Run with `--help` for all options.

mod autodiff;
mod config;
mod model;
mod plot;
mod reference;
mod sampler;
mod sr;

use config::{Config, BLOCK, USAGE};
use model::Model;
use sampler::{block_standard_error, mean_variance, sample, Chain, CHAINS};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tenferro_cpu::CpuEngine;

/// Sweeps discarded before the first SR step and before validation.
const TRAINING_BURN_IN: usize = 40;
const VALIDATION_BURN_IN: usize = 400;
/// Sweeps between SR steps; the chains persist across steps.
const STEP_BURN_IN: usize = 2;
const PRINT_EVERY: usize = 20;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let config = Config::from_args(&args).map_err(|e| format!("{e}\n\n{USAGE}"))?;
    let mut model = Model::new(
        config.lx(),
        config.ly(),
        config.alpha,
        config.system.hamiltonian(),
        config.seed,
    );
    if let Some(path) = &config.init_weights {
        model.weights = read_weights(path, &model)?;
    }
    if let Some(dir) = &config.output_dir {
        if dir.join("history.csv").exists() {
            return Err(format!(
                "{} already holds a history.csv; choose a new --output-dir",
                dir.display()
            )
            .into());
        }
        fs::create_dir_all(dir)?;
        if config.init_weights.is_some() {
            save_weights(&model, &dir.join("initial_weights.csv"))?;
        }
    }
    let reference = reference::reference_energy(&model)?;
    let n = model.n();
    println!(
        "{} {}x{}: N={n}, alpha={}, {} weights, {} steps of {} samples, seed {}",
        config.system.name(),
        model.lx,
        model.ly,
        model.features,
        model.weights.len(),
        config.steps,
        config.samples,
        config.seed
    );
    match reference {
        Some(e) => println!("reference E/N = {e:.8}"),
        None => println!("no reference energy is known for this lattice"),
    }

    let mut chains: Vec<Chain> = (0..CHAINS as u64)
        .map(|i| Chain::new(&model, config.seed + 101 + i))
        .collect();
    check_autodiff(&model, &chains[0].spins)?;

    let mut history_file = match &config.output_dir {
        Some(dir) => {
            let mut file = fs::File::create(dir.join("history.csv"))?;
            writeln!(
                file,
                "iteration,energy_per_site,variance_per_site,acceptance,learning_rate,step_rms,elapsed_seconds"
            )?;
            Some(file)
        }
        None => None,
    };
    let mut energies = Vec::with_capacity(config.steps);
    let mut engine = CpuEngine::serial();
    let start = Instant::now();
    for iteration in 0..config.steps {
        let burn = if iteration == 0 {
            TRAINING_BURN_IN
        } else {
            STEP_BURN_IN
        };
        let batch = sample(&mut chains, &model, config.samples / CHAINS, burn, true);
        let (energy, variance) = mean_variance(&batch.energies);
        let delta = sr::sr_step(
            &batch.gradients,
            &batch.energies,
            model.weights.len(),
            sr::damping(iteration),
            &mut engine,
            config.sr_shift,
        )?;
        if delta.iter().any(|x| !x.is_finite()) {
            return Err(format!("non-finite SR step at iteration {iteration}").into());
        }
        let rate = sr::step_rate(&batch.gradients, &delta, config.max_weight_step);
        let step_rms = sr::rms(&delta) * rate;
        for (w, d) in model.weights.iter_mut().zip(&delta) {
            *w -= rate * d;
        }
        let (e, v) = (energy / n as f64, variance / n as f64);
        let acceptance = batch.acceptance();
        let elapsed = start.elapsed().as_secs_f64();
        energies.push(e);
        let last = iteration + 1 == config.steps;
        if let Some(file) = &mut history_file {
            writeln!(
                file,
                "{iteration},{e:.12},{v:.12},{acceptance:.6},{rate:.8},{step_rms:.8},{elapsed:.3}"
            )?;
            if iteration % PRINT_EVERY == 0 || last {
                file.flush()?;
                let dir = config
                    .output_dir
                    .as_deref()
                    .expect("history implies a directory");
                save_weights(&model, &dir.join("weights.csv"))?;
            }
        }
        if iteration % PRINT_EVERY == 0 || last {
            println!(
                "step {iteration:5}/{}: E/N={e:.8} var/N={v:.5} accept={acceptance:.3} t={elapsed:.1}s",
                config.steps
            );
        }
    }

    let mut validation_chains: Vec<Chain> = (0..CHAINS as u64)
        .map(|i| Chain::new(&model, config.seed + 100_001 + i))
        .collect();
    let validation = sample(
        &mut validation_chains,
        &model,
        config.validation_sweeps,
        VALIDATION_BURN_IN,
        false,
    );
    let (energy, variance) = mean_variance(&validation.energies);
    let (energy, variance) = (energy / n as f64, variance / n as f64);
    let stderr = block_standard_error(&validation.energies, BLOCK) / n as f64;
    println!(
        "validation: E/N = {energy:.8} +- {stderr:.8} (var/N={variance:.5}, {} samples, accept={:.3})",
        validation.energies.len(),
        validation.acceptance()
    );
    let relative_error = reference.map(|r| (energy - r) / r.abs());
    if let Some(error) = relative_error {
        println!("relative error vs reference: {error:.3e}");
    }

    let exact = if config.exact {
        let exact = reference::exact_check(&model)?;
        println!(
            "exact ({} states): ground E/N = {:.10}, trained E/N = {:.10}, relative error {:.3e}, fidelity {:.9}",
            exact.dimension,
            exact.ground_energy,
            exact.variational_energy,
            (exact.variational_energy - exact.ground_energy) / exact.ground_energy.abs(),
            exact.fidelity
        );
        Some(exact)
    } else {
        None
    };

    if let Some(dir) = &config.output_dir {
        let mut raw = fs::File::create(dir.join("validation_energies.csv"))?;
        writeln!(raw, "energy")?;
        for e in &validation.energies {
            writeln!(raw, "{e:.16}")?;
        }
        let json_number = |x: Option<f64>| x.map_or("null".to_string(), |x| x.to_string());
        let exact_fields = exact.as_ref().map_or(String::new(), |x| {
            format!(
                ",\"exact_dimension\":{},\"exact_ground_energy_per_site\":{},\"exact_variational_energy_per_site\":{},\"exact_variance_per_site\":{},\"exact_fidelity\":{}",
                x.dimension, x.ground_energy, x.variational_energy, x.variance, x.fidelity
            )
        });
        let summary = format!(
            "{{\"system\":\"{}\",\"lx\":{},\"ly\":{},\"features\":{},\"seed\":{},\"steps\":{},\"samples_per_step\":{},\"sr_identity_shift\":{},\"max_weight_step\":{},\"sr_lambda_min\":{},\"validation_samples\":{},\"energy_per_site\":{energy},\"stderr_per_site\":{stderr},\"variance_per_site\":{variance},\"acceptance\":{},\"reference_energy_per_site\":{},\"relative_energy_error\":{},\"elapsed_seconds\":{},\"spin_convention\":\"Pauli\"{exact_fields}}}\n",
            config.system.name(),
            model.lx,
            model.ly,
            model.features,
            config.seed,
            config.steps,
            config.samples,
            config.sr_shift,
            config.max_weight_step,
            sr::MIN_DAMPING,
            validation.energies.len(),
            validation.acceptance(),
            json_number(reference),
            json_number(relative_error),
            start.elapsed().as_secs_f64()
        );
        fs::write(dir.join("summary.json"), summary)?;
        println!("wrote results to {}", dir.display());
    }
    if let Some(path) = &config.energy_png {
        plot::write_energy_png(path, &energies, reference)?;
        println!("saved training energy plot to {path}");
    }
    if let Some(path) = &config.filters_png {
        plot::write_filters_png(path, &model)?;
        println!("saved filter plot to {path}");
    }
    Ok(())
}

/// Compare the hand-written log-derivatives used in training with tenferro
/// reverse-mode autodiff on one configuration, and fail if they disagree.
fn check_autodiff(model: &Model, spins: &[f64]) -> Result<(), Box<dyn Error>> {
    let expected = model.derivatives(spins, &model.fields(spins));
    let actual = autodiff::tied_gradient(model, spins)?;
    let scale = expected.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    let difference = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f64::max);
    if difference.is_nan() || difference > 1e-10 * scale {
        return Err(format!(
            "analytic log-derivatives differ from tenferro autodiff by {difference:e}"
        )
        .into());
    }
    println!("tenferro autodiff check: max |analytic - autodiff| = {difference:.1e}");
    Ok(())
}

/// Filters one per line, as written by [`save_weights`].
fn read_weights(path: &Path, model: &Model) -> Result<Vec<f64>, Box<dyn Error>> {
    let weights: Vec<f64> = fs::read_to_string(path)?
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if weights.len() != model.weights.len() || weights.iter().any(|x| !x.is_finite()) {
        return Err(format!(
            "{} must hold {} finite weights, found {}",
            path.display(),
            model.weights.len(),
            weights.len()
        )
        .into());
    }
    Ok(weights)
}

fn save_weights(model: &Model, path: &Path) -> Result<(), Box<dyn Error>> {
    let mut file = fs::File::create(path)?;
    for weights in model.weights.chunks(model.n()) {
        let line: Vec<String> = weights.iter().map(|x| format!("{x:.17}")).collect();
        writeln!(file, "{}", line.join(","))?;
    }
    Ok(())
}
