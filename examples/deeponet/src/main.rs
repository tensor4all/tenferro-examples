//! DeepONet example: learn the antiderivative operator
//! `G(u)(y) = ∫_0^y u(s) ds` on `[0, 1]` (Lu et al., *Nature Machine
//! Intelligence* 3, 218 (2021)).
//!
//! A branch network reads the values of `u` at 100 fixed sensors and a trunk
//! network reads the query point `y`. Their `p`-wide outputs are contracted,
//! and a scalar output bias `b0` is added. One traced loss, one reverse pass
//! for every parameter gradient, one compiled program, and one `Prepared` for
//! the whole training loop.
//!
//! Train with:
//!
//! ```bash
//! cargo run --release -p deeponet
//! ```
//!
//! Flags:
//!
//! - `--epochs N` sets the number of epochs (default 10000).
//! - `--seed K` sets the master seed (default 42); the same seed reproduces
//!   the run and the printed test error exactly.
//! - `--plot <path>` writes the test-function figure.

mod data;
mod model;
mod plot;

use data::{SENSORS, Y_POINTS};
use model::DeepONet;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sciml_kit::func::Params;
use sciml_kit::optim::Adam;
use std::error::Error;
use std::time::Instant;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::{runtime_engine_registration, CpuEngine};
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Prepared, Runtime, TensorRead, TraceContext,
    TracedTensor,
};
use tenferro_tensor::Tensor;

const N_TRAIN: usize = 1000;
const N_TEST: usize = 200;
const BATCH: usize = 64;
const DEFAULT_EPOCHS: usize = 10_000;
const LR: f64 = 1e-3;
const DEFAULT_SEED: u64 = 42;
const LOG_EVERY: usize = 100;

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

fn cpu_runtime(engine: &CpuEngine) -> Result<Runtime, Box<dyn Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(engine)?)?;
    Ok(builder.build()?)
}

/// Run a prepared program once and return every output as an owned tensor.
fn run_prepared(
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

/// Relative L2 error over all entries of two column-major tensors.
fn relative_l2(pred: &Tensor, truth: &Tensor) -> Result<f64, Box<dyn Error>> {
    let pred = pred.as_slice::<f64>()?;
    let truth = truth.as_slice::<f64>()?;
    let mut error = 0.0;
    let mut norm = 0.0;
    for (&p, &t) in pred.iter().zip(truth.iter()) {
        error += (p - t).powi(2);
        norm += t * t;
    }
    Ok((error / norm).sqrt())
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
    // Independent streams: the test set is disjoint from the training set and
    // does not depend on how the training batches are drawn.
    let train_seed = seed;
    let test_seed = seed.wrapping_add(1);
    let init_seed = seed.wrapping_add(2);
    let batch_seed = seed.wrapping_add(3);
    println!(
        "seed: {seed} (train {train_seed}, test {test_seed}, init {init_seed}, batches {batch_seed})"
    );
    println!(
        "training functions: {N_TRAIN}, test functions: {N_TEST}, sensors: {SENSORS}, \
         y points: {Y_POINTS}, batch: {BATCH}"
    );

    let startup = Instant::now();
    let mut engine = CpuEngine::serial();
    let (train, test) = data::generate(N_TRAIN, N_TEST, train_seed, test_seed, &mut engine)?;
    println!("generated data in {:.3} s", startup.elapsed().as_secs_f64());

    let trace = TraceContext::new();
    let net = DeepONet::new(&trace, SENSORS)?;
    let mut rng_init = StdRng::seed_from_u64(init_seed);
    let mut params = net.initial_values(&mut rng_init)?;
    let mut rng_batch = StdRng::seed_from_u64(batch_seed);

    // Parameters are created first, then the data placeholders, so the
    // compiled program's input order is [params..., u, y, target] (#6
    // convention 2; tenferro 0.11 has no explicit-order entry point for
    // several outputs, tenferro-rs#2642).
    let u = trace.input_concrete_shape(DType::F64, &[BATCH, SENSORS])?;
    let y = trace.input_concrete_shape(DType::F64, &[Y_POINTS, 1])?;
    let target = trace.input_concrete_shape(DType::F64, &[BATCH, Y_POINTS])?;
    let pred = net.forward(&u, &y)?;
    let diff = pred.sub(&target)?;
    let loss = diff
        .mul(&diff)?
        .reduce_sum(Some(&[0, 1]))?
        .scale_real(1.0 / (BATCH * Y_POINTS) as f64)?;

    let placeholders = net.placeholders();
    let (loss_value, grads) = loss.value_and_grad(&placeholders)?;
    let grads: Vec<TracedTensor> = grads
        .into_iter()
        .map(|grad| grad.expect("every parameter reaches the loss"))
        .collect();
    let mut program_outputs: Vec<&TracedTensor> = Vec::with_capacity(1 + grads.len());
    program_outputs.push(&loss_value);
    program_outputs.extend(grads.iter());

    let compiled = Instant::now();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&program_outputs)?;
    println!(
        "built and compiled the training program in {:.3} s",
        compiled.elapsed().as_secs_f64()
    );

    let runtime = cpu_runtime(&engine)?;
    let y_tensor = Tensor::from_vec_col_major(vec![Y_POINTS, 1], data::y_grid())?;
    let mut optim = Adam::new(LR);
    let mut prepared: Option<Prepared> = None;
    let mut final_loss = f64::INFINITY;

    let trained = Instant::now();
    for epoch in 0..epochs {
        let indices: Vec<usize> = (0..BATCH)
            .map(|_| rng_batch.gen_range(0..N_TRAIN))
            .collect();
        let (u_batch, target_batch) = train.minibatch(&indices)?;

        let mut inputs: Vec<&Tensor> = Vec::with_capacity(params.len() + 3);
        inputs.extend(params.iter());
        inputs.extend([&u_batch, &y_tensor, &target_batch]);
        let reads: Vec<TensorRead<'_>> = inputs
            .iter()
            .map(|&tensor| TensorRead::from_tensor(tensor))
            .collect();

        if prepared.is_none() {
            prepared = Some(runtime.prepare(&program, &reads, &[engine.target()])?);
        }
        let outputs = run_prepared(&runtime, &mut engine, prepared.as_ref().unwrap(), &reads)?;
        final_loss = outputs[0].as_slice::<f64>()?[0];
        optim.step(&mut params, &outputs[1..]);

        if epoch % LOG_EVERY == 0 {
            println!("epoch {epoch}: loss={final_loss:.6e}");
        }
    }
    println!("final loss after {epochs} epochs: {final_loss:.6e}");
    println!("training took {:.1} s", trained.elapsed().as_secs_f64());

    // Evaluate the held-out functions with a separate forward-only program;
    // its shapes differ from the minibatch, so it gets its own `Prepared`.
    let all: Vec<usize> = (0..N_TEST).collect();
    let (u_test_tensor, target_test_tensor) = test.minibatch(&all)?;
    let u_test = trace.input_concrete_shape(DType::F64, &[N_TEST, SENSORS])?;
    let y_test = trace.input_concrete_shape(DType::F64, &[Y_POINTS, 1])?;
    let pred_test = net.forward(&u_test, &y_test)?;
    let eval_program = compiler.compile_many(&[&pred_test])?;

    let mut eval_inputs: Vec<&Tensor> = params.iter().collect();
    eval_inputs.extend([&u_test_tensor, &y_tensor]);
    let eval_reads: Vec<TensorRead<'_>> = eval_inputs
        .iter()
        .map(|&tensor| TensorRead::from_tensor(tensor))
        .collect();
    let eval_prepared = runtime.prepare(&eval_program, &eval_reads, &[engine.target()])?;
    let predictions = run_prepared(&runtime, &mut engine, &eval_prepared, &eval_reads)?.remove(0);
    let relative = relative_l2(&predictions, &target_test_tensor)?;
    println!("test relative L2 error: {relative:.6e}");

    if let Some(path) = arg_value("--plot") {
        plot::write_test_png(
            &path,
            &data::sensor_grid(),
            &data::y_grid(),
            u_test_tensor.as_slice::<f64>()?,
            target_test_tensor.as_slice::<f64>()?,
            predictions.as_slice::<f64>()?,
            N_TEST,
        )?;
        println!("saved test-function figure to {path}");
    }
    Ok(())
}
