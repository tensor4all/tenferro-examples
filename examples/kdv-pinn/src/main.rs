//! Physics-informed neural network (PINN) example for the Korteweg–de Vries (KdV)
//! equation.
//!
//! This example trains a small multi-layer perceptron to approximate the KdV
//! single-soliton solution on the spatial domain `x ∈ [-5, 5]` and time domain
//! `t ∈ [0, 1]`. The loss combines the PDE residual in the interior, the initial
//! condition at `t = 0`, and exact-solution Dirichlet boundary data at `x = ±5`.
//!
//! Train the example with:
//!
//! ```bash
//! cargo run --release -p kdv_pinn
//! ```
//!
//! Optional outputs are controlled by command-line flags:
//!
//! - `--epochs N` sets the number of training epochs (default 3000).
//! - `--seed K` sets the random seed (default 42); the same seed reproduces a
//!   run exactly.
//! - `--gif <path>` writes an animated comparison of the predicted and analytic
//!   solution over time.
//! - `--loss-png <path>` writes the training-loss curve (log-scale y-axis).
//!
//! ```bash
//! cargo run --release -p kdv_pinn -- --gif kdv_pinn.gif --loss-png loss.png
//! ```

mod loss;
mod pde;
mod plot;
mod sampler;

use sampler::Sampler;
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use sciml_kit::optim::{step_decay_lr, Adam};
use std::time::Instant;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::{runtime_engine_registration, CpuEngine};
use tenferro_runtime::{
    CompiledGraph, DType, ExecutionEngine, GraphCompiler, Prepared, Runtime, TensorRead,
    TraceContext, TracedTensor,
};
use tenferro_tensor::Tensor;

/// Return the value following the given command-line `flag`, if present.
///
/// For example, with arguments `--gif out.gif`, `arg_value("--gif")` returns
/// `Some("out.gif")`. Returns `None` when the flag is absent or has no value.
fn arg_value(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for window in args.windows(2) {
        if window[0] == flag {
            return Some(window[1].clone());
        }
    }
    None
}

fn cpu_runtime(engine: &CpuEngine) -> Result<Runtime, Box<dyn std::error::Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(engine)?)?;
    Ok(builder.build()?)
}

/// The order in which `program` consumes an explicit input for `tensors`.
///
/// `compile_many` takes its input order from the trace context, and tenferro
/// 0.11 has no entrance that accepts an explicit order for several outputs at
/// once (tenferro-rs#2642), so each declared placeholder is mapped back to its
/// semantic input index instead of assuming that order. Constant leaves that
/// carry data — `t_zero` and the `ones` tangents — are semantic inputs too, but
/// they keep their bound value, and the runtime consumes no explicit input for
/// them.
///
/// (`CompiledGraph::input_key_index` is `#[doc(hidden)]` in tenferro 0.11; it is
/// the only way to name a program's inputs, which is the same #2642 gap.)
fn bind_order(program: &CompiledGraph, tensors: &[&TracedTensor]) -> Vec<usize> {
    let slot = |tensor: &TracedTensor| {
        let key = tensor
            .input_key()
            .expect("a declared placeholder has an input key");
        program
            .input_key_index(&key)
            .expect("a declared placeholder is a program input")
    };
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&index| slot(tensors[index]));
    order
}

/// The explicit inputs of a program run, given tensors in placeholder order.
fn ordered_reads<'a>(tensors: &[&'a Tensor], order: &[usize]) -> Vec<TensorRead<'a>> {
    order
        .iter()
        .map(|&index| TensorRead::from_tensor(tensors[index]))
        .collect()
}

/// Run a prepared program once and return every output as an owned tensor.
fn run_prepared(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    reads: &[TensorRead<'_>],
) -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
    let outputs = runtime.run(&mut [engine], prepared, reads)?;
    let mut tensors = Vec::with_capacity(outputs.len());
    for output in outputs {
        tensors.push(output.into_tensor(engine)?);
    }
    Ok(tensors)
}

/// Compile, prepare and run `output` once, returning its first output.
///
/// A one-shot helper for the unit tests; the training loop prepares once and
/// reuses the [`Prepared`].
#[cfg(test)]
fn run_single(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    Ok(run_prepared(runtime, engine, &prepared, &reads)?.remove(0))
}

const N_IC: usize = 128;
const N_COL: usize = 1024;
const N_BC: usize = 128;
const N_EVAL: usize = 100;
const LAMBDA_PDE: f64 = 1.0;
const LAMBDA_IC: f64 = 1.0;
const LAMBDA_BC: f64 = 1.0;
const LR: f64 = 0.001;
/// Default number of training epochs; override with `--epochs N`.
const DEFAULT_EPOCHS: usize = 3000;
/// Default random seed; override with `--seed K`.
const DEFAULT_SEED: u64 = 42;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rand::SeedableRng;

    let epochs = match arg_value("--epochs") {
        Some(value) => value.parse::<usize>()?,
        None => DEFAULT_EPOCHS,
    };
    let seed = match arg_value("--seed") {
        Some(value) => value.parse::<u64>()?,
        None => DEFAULT_SEED,
    };
    println!("seed: {} ({} epochs)", seed, epochs);

    let started = Instant::now();
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &[2, 64, 64, 1], TracedTensor::tanh)?;
    // Parameters first: their placeholders are the leading program inputs.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut params = net.initial_values(&mut rng)?;

    let x_ic = trace.input_concrete_shape(DType::F64, &[N_IC, 1])?;
    let u_ic_true = trace.input_concrete_shape(DType::F64, &[N_IC, 1])?;

    // A rank-0 zero expanded to the column shape: a constant that is not an AD
    // seed, so it needs no data-carrying leaf (see `pde::ones_like`).
    let t_zero = trace
        .from_vec_col_major(vec![], vec![0.0_f64])?
        .broadcast_in_dim(&[N_IC, 1], &[])?;
    let xt_ic = TracedTensor::stack(&[&x_ic, &t_zero], 1)?.reshape([N_IC, 2])?;
    let u_ic = net.forward(&xt_ic)?;

    let x_col = trace.input_concrete_shape(DType::F64, &[N_COL, 1])?;
    let t_col = trace.input_concrete_shape(DType::F64, &[N_COL, 1])?;
    let xt_col = TracedTensor::stack(&[&x_col, &t_col], 1)?.reshape([N_COL, 2])?;
    let u_col = net.forward(&xt_col)?;

    let x_bc = trace.input_concrete_shape(DType::F64, &[N_BC, 1])?;
    let t_bc = trace.input_concrete_shape(DType::F64, &[N_BC, 1])?;
    let u_bc_true = trace.input_concrete_shape(DType::F64, &[N_BC, 1])?;
    let xt_bc = TracedTensor::stack(&[&x_bc, &t_bc], 1)?.reshape([N_BC, 2])?;
    let u_bc = net.forward(&xt_bc)?;

    let residual = pde::kdv_residual(&u_col, &x_col, &t_col)?;
    let total_loss = loss::total_loss(
        &residual, &u_ic, &u_ic_true, &u_bc, &u_bc_true, N_COL, N_IC, N_BC, LAMBDA_PDE, LAMBDA_IC,
        LAMBDA_BC,
    )?;

    // One reverse pass gives every parameter gradient, and the loss value comes
    // with them, so an epoch needs one program and one run.
    let placeholders = net.placeholders();
    let (loss_value, grads) = total_loss.value_and_grad(&placeholders)?;
    let grads: Vec<TracedTensor> = grads
        .into_iter()
        .map(|grad| grad.expect("every parameter reaches the loss"))
        .collect();
    let mut outputs: Vec<&TracedTensor> = Vec::with_capacity(1 + grads.len());
    outputs.push(&loss_value);
    outputs.extend(grads.iter());

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&outputs)?;

    // The tensors a run binds, in placeholder order: parameters, then data.
    let data_placeholders: [&TracedTensor; 7] =
        [&x_col, &t_col, &x_ic, &u_ic_true, &x_bc, &t_bc, &u_bc_true];
    let bound: Vec<&TracedTensor> = placeholders
        .iter()
        .copied()
        .chain(data_placeholders)
        .collect();
    let order = bind_order(&program, &bound);
    println!(
        "built and compiled the training program in {:.3} s",
        started.elapsed().as_secs_f64()
    );

    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine)?;
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut optim = Adam::new(LR);

    let mut final_loss = f64::INFINITY;
    let mut loss_history: Vec<f64> = Vec::with_capacity(epochs);
    // Every epoch has the same input shapes, so one prepared program serves
    // all of them.
    let mut prepared: Option<Prepared> = None;
    for epoch in 0..epochs {
        let (x_col_tensor, t_col_tensor) = sampler.collocation(N_COL, &mut rng);
        let (x_ic_tensor, u_ic_tensor) = sampler.initial(N_IC, &mut rng);
        let (x_bc_tensor, t_bc_tensor, u_bc_tensor) = sampler.boundary(N_BC, &mut rng);

        let mut inputs: Vec<&Tensor> = Vec::with_capacity(bound.len());
        inputs.extend(params.iter());
        inputs.extend([
            &x_col_tensor,
            &t_col_tensor,
            &x_ic_tensor,
            &u_ic_tensor,
            &x_bc_tensor,
            &t_bc_tensor,
            &u_bc_tensor,
        ]);
        let reads = ordered_reads(&inputs, &order);

        if prepared.is_none() {
            prepared = Some(runtime.prepare(&program, &reads, &[engine.target()])?);
        }
        let outputs = run_prepared(&runtime, &mut engine, prepared.as_ref().unwrap(), &reads)?;

        final_loss = outputs[0].as_slice::<f64>()?[0];
        loss_history.push(final_loss);

        optim.set_lr(step_decay_lr(epoch, epochs, LR));
        optim.step(&mut params, &outputs[1..]);

        if epoch % 50 == 0 {
            println!("epoch {}: loss={:.6e}", epoch, final_loss);
        }
    }
    println!("final loss after {} epochs: {:.6e}", epochs, final_loss);

    if let Some(loss_png_path) = arg_value("--loss-png") {
        plot::write_loss_png(&loss_png_path, &loss_history).expect("write loss png failed");
        println!("saved loss curve to {}", loss_png_path);
    }

    // Evaluation grid at t = 0.5, and the same program for the GIF frames.
    let x_eval = trace.input_concrete_shape(DType::F64, &[N_EVAL, 1])?;
    let t_eval = trace.input_concrete_shape(DType::F64, &[N_EVAL, 1])?;
    let xt_eval = TracedTensor::stack(&[&x_eval, &t_eval], 1)?.reshape([N_EVAL, 2])?;
    let u_eval = net.forward(&xt_eval)?;
    let eval_program = compiler.compile_many(&[&u_eval])?;

    let eval_bound: Vec<&TracedTensor> = placeholders
        .iter()
        .copied()
        .chain([&x_eval, &t_eval])
        .collect();
    let eval_order = bind_order(&eval_program, &eval_bound);

    let mut x_eval_data = Vec::with_capacity(N_EVAL);
    let mut u_true_data = Vec::with_capacity(N_EVAL);
    for i in 0..N_EVAL {
        let x = -5.0 + 10.0 * (i as f64) / (N_EVAL as f64 - 1.0);
        let t = 0.5;
        x_eval_data.push(x);
        u_true_data.push(2.0 * (1.0 / ((x - 4.0 * t).cosh())).powi(2));
    }
    let x_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], x_eval_data.clone())?;
    let t_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], vec![0.5_f64; N_EVAL])?;
    let u_true_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], u_true_data)?;

    let mut eval_inputs: Vec<&Tensor> = Vec::with_capacity(params.len() + 2);
    eval_inputs.extend(params.iter());
    eval_inputs.extend([&x_eval_tensor, &t_eval_tensor]);
    let eval_reads = ordered_reads(&eval_inputs, &eval_order);
    let eval_prepared = runtime.prepare(&eval_program, &eval_reads, &[engine.target()])?;

    let u_pred = run_prepared(&runtime, &mut engine, &eval_prepared, &eval_reads)?.remove(0);
    let pred = u_pred.as_slice::<f64>().expect("predicted solution data");
    let truth = u_true_tensor.as_slice::<f64>().expect("true solution data");
    let l2_error = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        .sqrt()
        / truth.iter().map(|t| t.powi(2)).sum::<f64>().sqrt();
    println!("L2 relative error at t=0.5: {:.6e}", l2_error);

    if let Some(gif_path) = arg_value("--gif") {
        const N_FRAMES: usize = 30;
        let mut frames: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::with_capacity(N_FRAMES);
        for frame in 0..N_FRAMES {
            let t = frame as f64 / (N_FRAMES as f64 - 1.0);
            let t_frame_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], vec![t; N_EVAL])?;
            let mut frame_inputs: Vec<&Tensor> = Vec::with_capacity(params.len() + 2);
            frame_inputs.extend(params.iter());
            frame_inputs.extend([&x_eval_tensor, &t_frame_tensor]);
            let frame_reads = ordered_reads(&frame_inputs, &eval_order);

            let u_pred =
                run_prepared(&runtime, &mut engine, &eval_prepared, &frame_reads)?.remove(0);
            let pred = u_pred
                .as_slice::<f64>()
                .expect("predicted frame data")
                .to_vec();

            let analytic: Vec<f64> = x_eval_data
                .iter()
                .map(|&x| 2.0 * (1.0 / ((x - 4.0 * t).cosh())).powi(2))
                .collect();

            frames.push((t, analytic, pred));
            if frame % 10 == 0 {
                println!("rendered frame {} / {}", frame, N_FRAMES - 1);
            }
        }
        plot::write_comparison_gif(&gif_path, &x_eval_data, &frames)
            .expect("write comparison gif failed");
        println!("saved animation to {}", gif_path);
    }
    Ok(())
}
