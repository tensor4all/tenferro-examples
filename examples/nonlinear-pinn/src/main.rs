//! Physics-informed neural networks for two nonlinear PDEs, viscous Burgers and
//! Allen-Cahn, trained with Adam and with Adam followed by L-BFGS
//! (tenferro-examples#9).
//!
//! ```bash
//! cargo run --release -p nonlinear_pinn -- --problem burgers --optimizer adam-lbfgs
//! ```
//!
//! Options:
//!
//! - `--problem burgers|allen-cahn` (default `burgers`).
//! - `--optimizer adam|adam-lbfgs` (default `adam-lbfgs`). `lm`
//!   (Levenberg-Marquardt) is not wired up yet: `pinn-gauss-newton`'s step
//!   is local to that example (see the README).
//! - `--adam-steps N` (default 10000) and `--lbfgs-steps N` (default 10000,
//!   used by `adam-lbfgs`).
//! - `--seed K` (default 42): network initialization and collocation points.
//! - `--plot <path>`: reference and prediction at four times.
//!
//! The collocation, initial and boundary points are drawn once and stay fixed,
//! so the loss is a deterministic function of the parameters, which L-BFGS
//! needs. One compiled program returns the loss and every parameter gradient;
//! Adam steps and the L-BFGS two-loop recursion and line search run on the host
//! over its outputs. The relative L2 error is measured against a pseudo-spectral
//! reference (`reference.rs`, written with `tenferro-fft`) on a 256 x 101 grid.

mod plot;
mod problem;
mod reference;

use problem::Problem;
use rand::{Rng, SeedableRng};
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use sciml_kit::optim::{flatten, step_decay_lr, unflatten, Adam, Lbfgs, LbfgsStep};
use std::error::Error;
use std::time::Instant;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::{runtime_engine_registration, CpuEngine};
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Prepared, Runtime, TensorRead, TraceContext,
    TracedTensor,
};
use tenferro_tensor::Tensor;

/// Collocation points in the interior.
const N_F: usize = 8000;
/// Initial-condition points.
const N_IC: usize = 256;
/// Boundary points per side (Burgers only).
const N_BC: usize = 64;
/// Hidden layers and their width.
const DEPTH: usize = 6;
const WIDTH: usize = 32;
const ADAM_LR: f64 = 1e-3;
const LBFGS_MEMORY: usize = 50;
/// Evaluation grid: every `n / 256`-th reference point, at `t = 0, 0.01, ..., 1`.
const N_EVAL_X: usize = 256;

fn arg_value(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn parsed<T: std::str::FromStr>(flag: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    Ok(match arg_value(flag) {
        Some(value) => value.parse()?,
        None => default,
    })
}

/// Initial-condition weight: Allen-Cahn needs a stronger pull to its initial
/// data (Wang, Sankaran and Perdikaris 2022 use 100); Burgers uses 1.
fn ic_weight(problem: Problem) -> f64 {
    match problem {
        Problem::Burgers => 1.0,
        Problem::AllenCahn => 100.0,
    }
}

/// Grid size and time step of the spectral reference.
fn reference_resolution(problem: Problem) -> (usize, f64) {
    match problem {
        Problem::Burgers => (4096, 1e-4),
        Problem::AllenCahn => (2048, 1e-4),
    }
}

fn column(data: Vec<f64>) -> Tensor {
    let n = data.len();
    Tensor::from_vec_col_major(vec![n, 1], data).expect("column data")
}

/// Mean of the squares of a column, as a traced scalar.
fn mean_square(r: &TracedTensor, n: usize) -> tenferro_runtime::Result<TracedTensor> {
    r.mul(r)?.reduce_sum(None)?.scale_real(1.0 / n as f64)
}

fn run(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>, Box<dyn Error>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let mut tensors = Vec::new();
    for output in runtime.run(&mut [&mut *engine], prepared, &reads)? {
        tensors.push(output.into_tensor(engine)?);
    }
    Ok(tensors)
}

fn main() -> Result<(), Box<dyn Error>> {
    let problem_name: String = parsed("--problem", "burgers".to_string())?;
    let problem = Problem::parse(&problem_name)
        .ok_or_else(|| format!("unknown problem {problem_name:?}: use burgers or allen-cahn"))?;
    let optimizer: String = parsed("--optimizer", "adam-lbfgs".to_string())?;
    let use_lbfgs = match optimizer.as_str() {
        "adam" => false,
        "adam-lbfgs" => true,
        "lm" => {
            return Err(
                "Levenberg-Marquardt is not wired up yet: pinn-gauss-newton's \
                        step is local to that example (tenferro-examples#9)"
                    .into(),
            )
        }
        other => return Err(format!("unknown optimizer {other:?}: use adam or adam-lbfgs").into()),
    };
    let adam_steps: usize = parsed("--adam-steps", 10_000)?;
    let lbfgs_steps: usize = if use_lbfgs {
        parsed("--lbfgs-steps", 10_000)?
    } else {
        0
    };
    let seed: u64 = parsed("--seed", 42)?;
    println!(
        "{}: {optimizer}, {adam_steps} Adam steps, {lbfgs_steps} L-BFGS steps, seed {seed}",
        problem.name()
    );

    // Reference first: it is independent of training and fixes the error grid.
    let started = Instant::now();
    let (n_ref, dt_ref) = reference_resolution(problem);
    let reference = reference::solve(problem, n_ref, dt_ref, 0.01, 1.0);
    println!(
        "spectral reference ({n_ref} points, dt {dt_ref:e}) in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    // Parameters first: their placeholders are the leading program inputs, in
    // the order of `net.placeholders()`.
    let trace = TraceContext::new();
    let mut sizes = vec![problem.input_width()];
    sizes.extend([WIDTH; DEPTH]);
    sizes.push(1);
    let net = Mlp::new(&trace, &sizes, TracedTensor::tanh)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut params = net.initial_values(&mut rng)?;

    // Fixed training points.
    let mut uniform = |lo: f64, hi: f64, n: usize| -> Vec<f64> {
        (0..n).map(|_| rng.gen_range(lo..hi)).collect()
    };
    let x_f_data = uniform(-1.0, 1.0, N_F);
    let t_f_data = uniform(0.0, 1.0, N_F);
    let x_ic_data = uniform(-1.0, 1.0, N_IC);
    let t_bc_data = uniform(0.0, 1.0, 2 * N_BC);
    let u_ic_data: Vec<f64> = x_ic_data.iter().map(|&x| problem.initial(x)).collect();

    let x_f = trace.input_concrete_shape(DType::F64, &[N_F, 1])?;
    let t_f = trace.input_concrete_shape(DType::F64, &[N_F, 1])?;
    let x_ic = trace.input_concrete_shape(DType::F64, &[N_IC, 1])?;
    let u_ic_true = trace.input_concrete_shape(DType::F64, &[N_IC, 1])?;
    // A jvp tangent must carry data in tenferro 0.11, so the ones are a
    // full-shape constant leaf (as in kdv-pinn, tenferro-rs#2642).
    let ones = trace.from_vec_col_major(vec![N_F, 1], vec![1.0_f64; N_F])?;
    let t_zero = trace
        .from_vec_col_major(vec![], vec![0.0_f64])?
        .broadcast_in_dim(&[N_IC, 1], &[])?;

    let u_f = net.forward(&problem.features(&x_f, &t_f)?)?;
    let residual = problem.residual(&u_f, &x_f, &t_f, &ones)?;
    let u_ic = net.forward(&problem.features(&x_ic, &t_zero)?)?;
    let mut loss = mean_square(&residual, N_F)?
        .add(&mean_square(&u_ic.sub(&u_ic_true)?, N_IC)?.scale_real(ic_weight(problem))?)?;
    let mut data: Vec<Tensor> = vec![
        column(x_f_data),
        column(t_f_data),
        column(x_ic_data),
        column(u_ic_data),
    ];
    if problem == Problem::Burgers {
        // u(±1, t) = 0: half the points on each side
        let x_bc = trace.input_concrete_shape(DType::F64, &[2 * N_BC, 1])?;
        let t_bc = trace.input_concrete_shape(DType::F64, &[2 * N_BC, 1])?;
        let u_bc = net.forward(&problem.features(&x_bc, &t_bc)?)?;
        loss = loss.add(&mean_square(&u_bc, 2 * N_BC)?)?;
        let sides = (0..2 * N_BC)
            .map(|i| if i < N_BC { -1.0 } else { 1.0 })
            .collect();
        data.extend([column(sides), column(t_bc_data)]);
    }

    let placeholders = net.placeholders();
    let (loss_value, grads) = loss.value_and_grad(&placeholders)?;
    let grads: Vec<TracedTensor> = grads
        .into_iter()
        .map(|grad| grad.expect("every parameter reaches the loss"))
        .collect();
    let mut outputs = vec![&loss_value];
    outputs.extend(grads.iter());
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&outputs)?;

    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&engine)?)?;
    let runtime = builder.build()?;
    // program inputs: the parameters, then the data in placeholder order
    let first_reads: Vec<TensorRead<'_>> = params
        .iter()
        .chain(&data)
        .map(TensorRead::from_tensor)
        .collect();
    let prepared = runtime.prepare(&program, &first_reads, &[engine.target()])?;
    drop(first_reads);
    let mut evaluate = |params: &[Tensor]| -> Result<Vec<Tensor>, Box<dyn Error>> {
        let inputs: Vec<&Tensor> = params.iter().chain(&data).collect();
        run(&runtime, &mut engine, &prepared, &inputs)
    };

    let n_params: usize = params
        .iter()
        .map(|p| p.shape().iter().product::<usize>())
        .sum();
    println!(
        "{n_params} parameters; program built, compiled and prepared in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    // Adam.
    let training = Instant::now();
    let mut adam = Adam::new(ADAM_LR);
    let mut last_loss = f64::NAN;
    for step in 0..adam_steps {
        let outputs = evaluate(&params)?;
        last_loss = outputs[0].as_slice::<f64>()?[0];
        adam.set_lr(step_decay_lr(step, adam_steps, ADAM_LR));
        adam.step(&mut params, &outputs[1..]);
        if step % 1000 == 0 {
            println!("adam {step}: loss {last_loss:.4e}");
        }
    }
    let adam_seconds = training.elapsed().as_secs_f64();
    // each step evaluated the loss before its update: one more run gives the loss of
    // the final parameters
    if adam_steps > 0 {
        last_loss = evaluate(&params)?[0].as_slice::<f64>()?[0];
    }
    println!("adam done: loss {last_loss:.4e}, {adam_seconds:.1} s");

    // L-BFGS on the flat parameter vector.
    let mut evaluations = 0;
    if lbfgs_steps > 0 {
        let mut x = flatten(&params);
        // Tensor is not Clone in tenferro 0.11: same-shape buffers for the trials
        let mut scratch = params
            .iter()
            .map(|p| {
                Tensor::from_vec_col_major(p.shape().to_vec(), flatten(std::slice::from_ref(p)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut eval_flat = |x: &[f64]| -> Result<(f64, Vec<f64>), Box<dyn Error>> {
            unflatten(x, &mut scratch);
            let outputs = evaluate(&scratch)?;
            Ok((outputs[0].as_slice::<f64>()?[0], flatten(&outputs[1..])))
        };
        let (mut f, mut g) = eval_flat(&x)?;
        let mut lbfgs = Lbfgs::new(LBFGS_MEMORY);
        for step in 0..lbfgs_steps {
            match lbfgs.step(&mut x, &mut f, &mut g, &mut eval_flat)? {
                LbfgsStep::Accepted { evaluations: used } => evaluations += used,
                LbfgsStep::LineSearchFailed { evaluations: used } => {
                    evaluations += used;
                    println!("l-bfgs {step}: line search failed, stopping");
                    break;
                }
            }
            if step % 1000 == 0 {
                println!("l-bfgs {step}: loss {f:.4e}");
            }
        }
        unflatten(&x, &mut params);
        last_loss = f;
        println!(
            "l-bfgs done: loss {last_loss:.4e}, {evaluations} evaluations, {:.1} s",
            training.elapsed().as_secs_f64() - adam_seconds
        );
    }
    let train_seconds = training.elapsed().as_secs_f64();

    // Evaluation on the reference grid.
    let stride = n_ref / N_EVAL_X;
    let x_grid: Vec<f64> = (0..N_EVAL_X).map(|j| reference.x[j * stride]).collect();
    let n_t = reference.t.len();
    let n_eval = N_EVAL_X * n_t;
    let x_eval = trace.input_concrete_shape(DType::F64, &[n_eval, 1])?;
    let t_eval = trace.input_concrete_shape(DType::F64, &[n_eval, 1])?;
    let u_eval = net.forward(&problem.features(&x_eval, &t_eval)?)?;
    let eval_program = compiler.compile_many(&[&u_eval])?;
    // column-major [x, t] grid: x fastest
    let x_eval_data = column((0..n_eval).map(|i| x_grid[i % N_EVAL_X]).collect());
    let t_eval_data = column((0..n_eval).map(|i| reference.t[i / N_EVAL_X]).collect());
    let eval_inputs: Vec<&Tensor> = params.iter().chain([&x_eval_data, &t_eval_data]).collect();
    let eval_reads: Vec<TensorRead<'_>> = eval_inputs
        .iter()
        .map(|t| TensorRead::from_tensor(t))
        .collect();
    let eval_prepared = runtime.prepare(&eval_program, &eval_reads, &[engine.target()])?;
    let prediction = run(&runtime, &mut engine, &eval_prepared, &eval_inputs)?.remove(0);
    let prediction = prediction.as_slice::<f64>()?;
    let (mut error, mut norm) = (0.0, 0.0);
    for s in 0..n_t {
        for j in 0..N_EVAL_X {
            let truth = reference.u[s][j * stride];
            error += (prediction[s * N_EVAL_X + j] - truth).powi(2);
            norm += truth * truth;
        }
    }
    let relative_l2 = (error / norm).sqrt();
    println!(
        "{} {optimizer}: final loss {last_loss:.4e}, relative L2 error {relative_l2:.4e}, \
         training {train_seconds:.1} s",
        problem.name()
    );

    if let Some(path) = arg_value("--plot") {
        let snapshots: Vec<(Vec<f64>, Vec<f64>)> = [25, 50, 75, 100]
            .iter()
            .map(|&s| {
                let truth = (0..N_EVAL_X).map(|j| reference.u[s][j * stride]).collect();
                let pred = prediction[s * N_EVAL_X..(s + 1) * N_EVAL_X].to_vec();
                (truth, pred)
            })
            .collect();
        plot::write_snapshots_png(&path, &x_grid, &snapshots)?;
        println!("saved snapshots to {path}");
    }
    Ok(())
}
