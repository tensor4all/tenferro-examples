//! Neural ODE example: fit the spiral `dz/dt = A z³`
//! (Chen, Rubanova, Bettencourt and Duvenaud, *Neural Ordinary Differential
//! Equations*, NeurIPS 2018).
//!
//! The right-hand side is an MLP applied to `z³`. A fixed-step RK4 solver steps
//! over a 10-step window and the gradient is its checkpointed discrete adjoint:
//! one RK4-step graph, with the state and the cotangent as inputs, runs 10
//! times forward and 10 times reverse per epoch. The whole gradient is one
//! program, prepared once, and the mean-absolute-error cotangent is a host-side
//! sign.
//!
//! Train with:
//!
//! ```bash
//! cargo run --release -p neural_ode
//! ```
//!
//! Flags:
//!
//! - `--epochs N` sets the number of epochs (default 20000).
//! - `--seed K` sets the master seed (default 42); the same seed reproduces
//!   the run and the printed error exactly.
//! - `--plot <path>` writes the trajectory figure.

mod data;
mod model;
mod ode;
mod plot;
mod train;

use data::{Dataset, H, N_SAMPLES, WINDOW_STEPS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use sciml_kit::optim::{step_decay_lr, Adam};
use std::error::Error;
use std::time::Instant;
use tenferro_cpu::{runtime_engine_registration, CpuEngine};
use tenferro_runtime::{
    DType, ExecutionEngine, Prepared, Runtime, TensorRead, TraceContext, TracedTensor,
};
use tenferro_tensor::Tensor;

const DEFAULT_EPOCHS: usize = 20_000;
const LR: f64 = 1e-3;
const DEFAULT_SEED: u64 = 42;
const LOG_EVERY: usize = 500;

/// Return the value following the given command-line `flag`, if present.
fn arg_value(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for window in args.windows(2) {
        if window[0] == flag {
            return Some(window[1].clone());
        }
    }
    None
}

pub(crate) fn cpu_runtime(engine: &CpuEngine) -> Result<Runtime, Box<dyn Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(engine)?)?;
    Ok(builder.build()?)
}

/// Run a prepared program once and return every output as an owned tensor.
pub(crate) fn run_prepared(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    reads: &[TensorRead<'_>],
) -> Result<Vec<Tensor>, Box<dyn Error>> {
    let outputs = runtime.run(&mut [engine], prepared, reads)?;
    let mut tensors = Vec::with_capacity(outputs.len());
    for output in outputs {
        tensors.push(output.into_tensor(engine)?);
    }
    Ok(tensors)
}

/// Compile, prepare and run `program` once, returning its first output.
///
/// A one-shot helper for the unit tests; the training and evaluation loops
/// prepare once and reuse the [`Prepared`].
#[cfg(test)]
fn run_single(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &tenferro_runtime::CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Tensor, Box<dyn Error>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    Ok(run_prepared(runtime, engine, &prepared, &reads)?.remove(0))
}

fn main() -> Result<(), Box<dyn Error>> {
    let epochs = match arg_value("--epochs") {
        Some(value) => value.parse::<usize>()?,
        None => DEFAULT_EPOCHS,
    };
    let seed = match arg_value("--seed") {
        Some(value) => value.parse::<u64>()?,
        None => DEFAULT_SEED,
    };
    let init_seed = seed;
    let window_seed = seed.wrapping_add(1);
    println!("seed: {seed} (initial values {init_seed}, windows {window_seed}), epochs: {epochs}");
    println!(
        "{N_SAMPLES} samples on [0, 25], solver step {H:.6}; \
         {WINDOW_STEPS}-step windows, batch {}",
        data::BATCH
    );

    let startup = Instant::now();
    let data = Dataset::generate();
    println!(
        "generated the reference trajectory in {:.3} s",
        startup.elapsed().as_secs_f64()
    );

    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &model::MLP_SIZES, TracedTensor::tanh)?;
    let mut params = net.initial_values(&mut StdRng::seed_from_u64(init_seed))?;
    let mut rng_window = StdRng::seed_from_u64(window_seed);

    let compiled = Instant::now();
    let program = train::build(&trace, &net, data::BATCH)?;
    println!(
        "built and compiled the training program in {:.3} s",
        compiled.elapsed().as_secs_f64()
    );

    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine)?;
    let mut optim = Adam::new(LR);
    let mut final_loss = f64::INFINITY;
    let mut loss_history = Vec::with_capacity(epochs);

    // `prepare` binds a shape, not the values that follow: a batch of first
    // windows is enough, and it must not advance the window RNG.
    let (prepare_z0, _) = data.minibatch(&[0; data::BATCH])?;
    let at = Instant::now();
    let prepared = train::prepare(&runtime, &mut engine, &program, &params, &prepare_z0)?;
    let prepare_time = at.elapsed().as_secs_f64();
    println!("prepared the training program in {prepare_time:.3} s");

    let trained = Instant::now();
    for epoch in 0..epochs {
        let starts: Vec<usize> = (0..data::BATCH)
            .map(|_| rng_window.gen_range(0..N_SAMPLES - WINDOW_STEPS))
            .collect();
        let (z0, target_tensors) = data.minibatch(&starts)?;
        let targets: Vec<Vec<f64>> = target_tensors
            .iter()
            .map(|target| Ok(target.as_slice::<f64>()?.to_vec()))
            .collect::<Result<_, Box<dyn Error>>>()?;

        let result =
            train::window_loss_and_grad(&runtime, &mut engine, &prepared, &params, &z0, &targets)?;
        final_loss = result.loss;
        loss_history.push(final_loss);

        let grads: Vec<Tensor> = result
            .grads
            .into_iter()
            .zip(&params)
            .map(|(values, param)| Tensor::from_vec_col_major(param.shape().to_vec(), values))
            .collect::<Result<Vec<_>, _>>()?;
        optim.set_lr(step_decay_lr(epoch, epochs, LR));
        optim.step(&mut params, &grads);

        if epoch % LOG_EVERY == 0 {
            println!("epoch {epoch}: loss={final_loss:.6e}");
        }
    }
    let training = trained.elapsed().as_secs_f64();
    let per_epoch = 1000.0 * training / epochs.max(1) as f64;
    println!("final loss after {epochs} epochs: {final_loss:.6e}");
    println!("training took {training:.1} s ({per_epoch:.2} ms per epoch)");

    // Evaluate the full rollout with a forward-only program: the same 10-step
    // window is chained on the host, feeding its last state back as the next
    // input. 100 windows cover 1000 steps; the trajectory has 999 intervals, so
    // the final state is computed but not recorded.
    let z_eval = trace.input_concrete_shape(DType::F64, &[1, 2])?;
    let eval_states = ode::rollout(&|z| model::rhs(&net, z), &z_eval, H, WINDOW_STEPS)?;
    let eval_refs: Vec<&TracedTensor> = eval_states.iter().collect();
    let eval_compiled = Instant::now();
    let eval_program = tenferro_runtime::GraphCompiler::new().compile_many(&eval_refs)?;
    println!(
        "built and compiled the evaluation program in {:.3} s",
        eval_compiled.elapsed().as_secs_f64()
    );

    let windows = (N_SAMPLES - 1).div_ceil(WINDOW_STEPS);
    let mut samples: Vec<[f64; 2]> = Vec::with_capacity(N_SAMPLES);
    samples.push(data.state(0));
    let mut z_host = Tensor::from_vec_col_major(vec![1, 2], data.state(0).to_vec())?;
    let eval_prepared = {
        let mut inputs: Vec<&Tensor> = params.iter().collect();
        inputs.push(&z_host);
        let reads: Vec<TensorRead<'_>> = inputs
            .iter()
            .map(|&tensor| TensorRead::from_tensor(tensor))
            .collect();
        runtime.prepare(&eval_program, &reads, &[engine.target()])?
    };
    for _ in 0..windows {
        let outputs = {
            let mut inputs: Vec<&Tensor> = params.iter().collect();
            inputs.push(&z_host);
            let reads: Vec<TensorRead<'_>> = inputs
                .iter()
                .map(|&tensor| TensorRead::from_tensor(tensor))
                .collect();
            run_prepared(&runtime, &mut engine, &eval_prepared, &reads)?
        };
        let next: Vec<f64> = outputs
            .last()
            .expect("one output per window step")
            .as_slice::<f64>()?
            .to_vec();
        for state in &outputs {
            if samples.len() < N_SAMPLES {
                let values = state.as_slice::<f64>()?;
                samples.push([values[0], values[1]]);
            }
        }
        z_host = Tensor::from_vec_col_major(vec![1, 2], next)?;
    }
    assert_eq!(
        samples.len(),
        N_SAMPLES,
        "the evaluation must record every sample time"
    );

    // `Dataset::states` is column-major, and so is the plot.
    let mut trajectory = vec![0.0; N_SAMPLES * 2];
    for (index, sample) in samples.iter().enumerate() {
        trajectory[index] = sample[0];
        trajectory[N_SAMPLES + index] = sample[1];
    }

    let truth = data.states();
    let mut error = 0.0;
    let mut norm = 0.0;
    for (&predicted, &reference) in trajectory.iter().zip(truth) {
        error += (predicted - reference).powi(2);
        norm += reference * reference;
    }
    let relative_l2 = (error / norm).sqrt();
    println!("trajectory relative L2 error: {relative_l2:.6e}");
    // The pointwise relative L2 is dominated by phase drift late in the
    // rollout; the per-time errors show how it accumulates.
    for &index in &[N_SAMPLES / 5 - 1, N_SAMPLES - 1] {
        let err = (0..2)
            .map(|c| (trajectory[c * N_SAMPLES + index] - truth[c * N_SAMPLES + index]).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("  state error at t={:.2}: {err:.3e}", index as f64 * H);
    }

    if let Some(path) = arg_value("--plot") {
        let times: Vec<f64> = (0..N_SAMPLES).map(|i| i as f64 * H).collect();
        // One loss per minibatch is noisy; the figure shows block means.
        let block = 50;
        let curve: Vec<f64> = loss_history
            .chunks(block)
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .collect();
        plot::write_figure_png(&path, &times, truth, &trajectory, &curve)?;
        println!("saved figure to {path}");
    }
    Ok(())
}
