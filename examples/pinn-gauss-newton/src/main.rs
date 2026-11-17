//! Physics-informed neural network for the Poisson equation, trained with
//! Levenberg-Marquardt and with Adam in one binary.
//!
//! The problem has an exact solution, so the training error is measured against
//! it instead of against a reference run:
//!
//! - one dimension: `-u'' = pi^2 sin(pi x)` on `(0, 1)`, `u = sin(pi x)`;
//! - two dimensions: `-Laplace u = 2 pi^2 sin(pi x) sin(pi y)` on the unit
//!   square, `u = sin(pi x) sin(pi y)`,
//!
//! both with zero Dirichlet data on the boundary.
//!
//! The Jacobian of the residual vector is taken in one reverse pass with the
//! parameters fed a second time as a **per-point copy** (`theta_b`, shape
//! `[out, in, N]`): `grad(sum(r), theta_b)` returns one partial per point, which
//! is exactly the transposed Jacobian. Levenberg-Marquardt then solves
//! `(J^T J + lambda I) delta = -J^T r` inside the graph, together with the
//! candidate residual at `theta + delta`, so one program run and one read-back
//! give the host everything the accept/reject test needs.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release -p pinn_gauss_newton -- --dim 1 --optimizer lm
//! cargo run --release -p pinn_gauss_newton -- --dim 2 --optimizer adam
//! cargo run --release -p pinn_gauss_newton -- --optimizer both --plot loss.png
//! ```
//!
//! Flags:
//!
//! - `--dim 1|2` (default 1): the problem dimension.
//! - `--optimizer adam|lm|both` (default `lm`).
//! - `--epochs N`: steps for the optimizer that runs — Adam epochs or
//!   Levenberg-Marquardt iterations, applied to both when `--optimizer both`
//!   is given; defaults differ per optimizer (150 iterations, 2000 epochs).
//! - `--seed K` (default 42): points and initial parameters.
//! - `--plot <path>`: writes the loss curves (x = seconds, y = loss, log axis).

mod lm;
mod network;
mod plot;
mod poisson;

use std::error::Error;
use std::time::Instant;

use network::Model;
use plotters::style::{BLUE, RED};
use rand::rngs::StdRng;
use rand::SeedableRng;
use sciml_kit::func::Params;
use sciml_kit::optim::{step_decay_lr, Adam};
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_runtime::{
    CompiledGraph, DType, ExecutionEngine, GraphCompiler, Prepared, Result as TrResult, Runtime,
    TensorRead, TraceContext, TracedTensor,
};
use tenferro_tensor::Tensor;

/// Width of the two hidden layers.
const WIDTH: usize = 32;
/// Collocation points of one run.
const N_COL: usize = 2048;
/// Boundary points per edge in two dimensions (four edges in total).
const N_BC_2D: usize = 32;
/// Initial damping factor of Levenberg-Marquardt.
const LAMBDA_INIT: f64 = 1e-3;
/// Default iterations of a Levenberg-Marquardt run.
const DEFAULT_LM_STEPS: usize = 150;
/// Default epochs of an Adam run.
const DEFAULT_ADAM_EPOCHS: usize = 2000;
/// Base learning rate of the Adam baseline.
const ADAM_LR: f64 = 1e-3;
/// Evaluation points per axis (one dimension: points, two: `n × n` grid).
const N_EVAL_1D: usize = 200;
const N_EVAL_2D: usize = 60;
/// Default seed of the point set and the initial parameters.
const DEFAULT_SEED: u64 = 42;

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

/// A runtime with the CPU engine and the linalg extension installed.
///
/// The Cholesky factorization and the triangular solve are `tenferro-linalg`
/// extension operations, so a runtime that prepares a program containing one
/// needs the extension module as well as the engine.
fn cpu_runtime(engine: &CpuEngine) -> Result<Runtime, Box<dyn Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(engine)?)?;
    builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
        runtime_engine_id()?,
    )?)?;
    Ok(builder.build()?)
}

/// A compiled program and where each of its inputs comes from.
///
/// The graph folds some declared placeholders into constants and appends the
/// inputs `tenferro-ad` introduced (one per `jvp` seed it imported, one per
/// reverse-pass cotangent), so input slots are matched to the caller's handles
/// by key. The slots no handle names are the AD inputs, in program order.
struct Program {
    graph: CompiledGraph,
    slots: Vec<Option<usize>>,
}

impl Program {
    /// Compile `outputs` and record which program input each handle names.
    fn new(outputs: &[&TracedTensor], handles: &[&TracedTensor]) -> Result<Self, Box<dyn Error>> {
        let graph = GraphCompiler::new().compile_many(outputs)?;
        let mut slots = Vec::with_capacity(graph.input_count());
        for key in graph.input_keys() {
            slots.push(
                handles
                    .iter()
                    .position(|handle| handle.input_key().as_ref() == Some(key)),
            );
        }
        Ok(Self { graph, slots })
    }

    /// The reads of one run: `values` in handle order, `extras` in program order.
    fn reads<'a>(&self, values: &[&'a Tensor], extras: &[&'a Tensor]) -> Vec<TensorRead<'a>> {
        assert_eq!(
            self.slots.iter().filter(|slot| slot.is_none()).count(),
            extras.len(),
            "one AD input per program input no handle names"
        );
        let mut extras = extras.iter();
        let mut reads = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let value: &Tensor = match slot {
                Some(index) => values.get(*index).expect("a value for every handle"),
                None => extras.next().expect("a value for every AD input"),
            };
            reads.push(TensorRead::from_tensor(value));
        }
        reads
    }
}

/// Run a prepared program once and return every output as an owned tensor.
fn run_prepared(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    reads: &[TensorRead<'_>],
) -> TrResult<Vec<Tensor>> {
    let outputs = runtime.run(&mut [engine], prepared, reads)?;
    let mut tensors = Vec::with_capacity(outputs.len());
    for output in outputs {
        tensors.push(output.into_tensor(engine)?);
    }
    Ok(tensors)
}

/// Relative L2 error of `pred` against `truth`, both column-major.
fn relative_l2(pred: &[f64], truth: &[f64]) -> f64 {
    let error: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let norm: f64 = truth.iter().map(|t| t * t).sum();
    (error / norm).sqrt()
}

/// What one training run produced.
struct Report {
    steps: usize,
    seconds: f64,
    final_loss: f64,
    history: Vec<(f64, f64)>,
    /// Damping raises that answered a failing factorization (LM only).
    retries: usize,
    /// Rejected steps (LM only).
    rejections: usize,
}

/// What one optimizer run needs besides its programs.
struct Run<'a> {
    /// The collocation and boundary points, `[dim, N]`.
    points: &'a Tensor,
    /// The AD inputs of the program, in program order.
    extras: &'a [&'a Tensor],
    /// The plain-parameter model, which provides the initial values.
    model: &'a Model,
    /// Seed of the initial parameters.
    seed: u64,
    /// Adam epochs or Levenberg-Marquardt iterations.
    steps: usize,
}

/// Train with Adam and return the parameters and the report.
fn train_adam(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &Program,
    run: &Run,
) -> Result<(Vec<Tensor>, Report), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(run.seed);
    let mut theta = run.model.initial_values(&mut rng)?;

    let first_reads = {
        let mut inputs: Vec<&Tensor> = theta.iter().collect();
        inputs.push(run.points);
        program.reads(&inputs, run.extras)
    };
    let prepared = runtime.prepare(&program.graph, &first_reads, &[engine.target()])?;

    let mut optim = Adam::new(ADAM_LR);
    let mut history = Vec::with_capacity(run.steps);
    let mut final_loss = f64::NAN;
    let started = Instant::now();
    for epoch in 0..run.steps {
        let reads = {
            let mut inputs: Vec<&Tensor> = theta.iter().collect();
            inputs.push(run.points);
            program.reads(&inputs, run.extras)
        };
        let outputs = run_prepared(runtime, engine, &prepared, &reads)?;
        final_loss = outputs[0].as_slice::<f64>()?[0];
        optim.set_lr(step_decay_lr(epoch, run.steps, ADAM_LR));
        optim.step(&mut theta, &outputs[1..]);
        history.push((started.elapsed().as_secs_f64(), final_loss));
        if epoch % 200 == 0 {
            println!("adam epoch {epoch}: loss={final_loss:.6e}");
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "adam: final loss={final_loss:.6e} after {} epochs in {seconds:.1} s",
        run.steps
    );
    Ok((
        theta,
        Report {
            steps: run.steps,
            seconds,
            final_loss,
            history,
            retries: 0,
            rejections: 0,
        },
    ))
}

/// Train with Levenberg-Marquardt and return the parameters and the report.
///
/// Every iteration makes one host copy of the parameters into the per-point
/// layout, runs the step program once, and applies the host decision. A failing
/// Cholesky factorization, reported as a numerical error, raises the damping
/// factor and retries the same iteration.
fn train_lm(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &Program,
    candidate_program: &Program,
    candidate_extras: &[&Tensor],
    run: &Run,
) -> Result<(Vec<Tensor>, Report), Box<dyn Error>> {
    let n = run.points.shape()[1];
    let mut rng = StdRng::seed_from_u64(run.seed);
    let mut theta = run.model.initial_values(&mut rng)?;
    let mut lambda = LAMBDA_INIT;
    let mut retries = 0;
    let mut rejections = 0;

    let initial_tiles = network::tile(&theta, n)?;
    let initial_lambda = Tensor::from_vec_col_major(vec![], vec![lambda])?;
    let first_reads = {
        let mut inputs: Vec<&Tensor> = initial_tiles.iter().collect();
        inputs.push(run.points);
        inputs.push(&initial_lambda);
        program.reads(&inputs, run.extras)
    };
    let prepared = runtime.prepare(&program.graph, &first_reads, &[engine.target()])?;
    let candidate_reads = {
        let mut inputs: Vec<&Tensor> = initial_tiles.iter().collect();
        inputs.push(run.points);
        candidate_program.reads(&inputs, candidate_extras)
    };
    let candidate_prepared = runtime.prepare(
        &candidate_program.graph,
        &candidate_reads,
        &[engine.target()],
    )?;

    let mut history = Vec::with_capacity(run.steps);
    let mut final_loss = f64::NAN;
    let mut stalled = 0;
    let mut iteration = 0;
    let started = Instant::now();
    while iteration < run.steps {
        let tiles = network::tile(&theta, n)?;
        let lambda_tensor = Tensor::from_vec_col_major(vec![], vec![lambda])?;
        let reads = {
            let mut inputs: Vec<&Tensor> = tiles.iter().collect();
            inputs.push(run.points);
            inputs.push(&lambda_tensor);
            program.reads(&inputs, run.extras)
        };

        let outputs = match run_prepared(runtime, engine, &prepared, &reads) {
            Ok(outputs) => outputs,
            Err(error) if lm::is_numerical_failure(&error) && lambda < lm::LAMBDA_MAX => {
                // J^T J + lambda I is not positive definite at this damping;
                // raise it and repeat the same iteration.
                lambda = (lambda * 10.0).min(lm::LAMBDA_MAX);
                retries += 1;
                println!(
                    "lm iteration {iteration}: factorization failed, lambda -> {lambda:.1e}, retry"
                );
                continue;
            }
            Err(error) => return Err(error.into()),
        };

        let delta = outputs[0].as_slice::<f64>()?;
        let loss = outputs[1].as_slice::<f64>()?[0];
        let predicted = outputs[2].as_slice::<f64>()?[0];

        // The candidate is one residual evaluation at theta + delta.
        let candidate = lm::stepped(&theta, delta)?;
        let candidate_tiles = network::tile(&candidate, n)?;
        let candidate_reads = {
            let mut inputs: Vec<&Tensor> = candidate_tiles.iter().collect();
            inputs.push(run.points);
            candidate_program.reads(&inputs, candidate_extras)
        };
        let loss_candidate = run_prepared(runtime, engine, &candidate_prepared, &candidate_reads)?
            .remove(0)
            .as_slice::<f64>()?[0];

        let decision = lm::decide(lambda, loss, loss_candidate, predicted);
        if decision.accept {
            theta = candidate;
            final_loss = loss_candidate;
            stalled = 0;
        } else {
            final_loss = loss;
            rejections += 1;
            stalled += 1;
        }
        let previous = lambda;
        lambda = decision.next_lambda;
        history.push((started.elapsed().as_secs_f64(), final_loss));
        println!(
            "lm iteration {iteration}: loss={loss:.6e} candidate={loss_candidate:.6e} \
             rho={:.2} {} lambda {:.1e} -> {:.1e}",
            decision.rho,
            if decision.accept { "accept" } else { "reject" },
            previous,
            lambda
        );
        iteration += 1;
        if stalled >= 3 && lambda >= lm::LAMBDA_MAX {
            println!(
                "lm: no progress with lambda at its cap after {iteration} iterations, stopping"
            );
            break;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "lm: final loss={final_loss:.6e} after {iteration} iterations in {seconds:.1} s \
         ({rejections} rejected, {retries} factorization retries, lambda {lambda:.1e})"
    );
    Ok((
        theta,
        Report {
            steps: iteration,
            seconds,
            final_loss,
            history,
            retries,
            rejections,
        },
    ))
}
fn evaluate(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &Program,
    points: &Tensor,
    params: &[Tensor],
    exact: &[f64],
) -> Result<f64, Box<dyn Error>> {
    let reads = {
        let mut inputs: Vec<&Tensor> = params.iter().collect();
        inputs.push(points);
        program.reads(&inputs, &[])
    };
    let prepared = runtime.prepare(&program.graph, &reads, &[engine.target()])?;
    let prediction = run_prepared(runtime, engine, &prepared, &reads)?.remove(0);
    Ok(relative_l2(prediction.as_slice::<f64>()?, exact))
}

fn main() -> Result<(), Box<dyn Error>> {
    let dim = match arg_value("--dim") {
        Some(value) => value.parse::<usize>()?,
        None => 1,
    };
    if dim != 1 && dim != 2 {
        return Err("--dim must be 1 or 2".into());
    }
    let optimizer = arg_value("--optimizer").unwrap_or_else(|| "lm".to_string());
    if !matches!(optimizer.as_str(), "lm" | "adam" | "both") {
        return Err("--optimizer must be adam, lm or both".into());
    }
    let seed = match arg_value("--seed") {
        Some(value) => value.parse::<u64>()?,
        None => DEFAULT_SEED,
    };
    let epochs = arg_value("--epochs")
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if epochs == Some(0) {
        return Err("--epochs must be at least 1".into());
    }
    let wants_adam = optimizer == "adam" || optimizer == "both";
    let wants_lm = optimizer == "lm" || optimizer == "both";
    let adam_epochs = epochs.unwrap_or(DEFAULT_ADAM_EPOCHS);
    let lm_iterations = epochs.unwrap_or(DEFAULT_LM_STEPS);

    let sizes = [dim, WIDTH, WIDTH, 1];
    let points = poisson::points(dim, N_COL, N_BC_2D, seed)?;

    // Each program gets its own trace: an AD transform walks the whole trace
    // it is applied in, and a `jvp` cannot cross a linalg extension op, so the
    // damped solve and the differentiated residual never share one.
    let templates = TraceContext::new();
    let plain = Model::new(&templates, &sizes, None)?;
    // The seed data is the same for every program, so one set serves them all.
    let seeds = poisson::Seeds::new(&templates, dim, points.len())?;

    let mut rng = StdRng::seed_from_u64(seed);
    let initial = plain.initial_values(&mut rng)?;
    let parameters: usize = initial
        .iter()
        .map(|tensor| tensor.shape().iter().product::<usize>())
        .sum();
    println!(
        "dim {dim}: {} points ({} collocation + {} boundary), {parameters} parameters, seed {seed}",
        points.len(),
        points.n_col,
        points.n_bc
    );
    println!(
        "per-step parameter copy: {:.1} MB (N*P*8 = {} * {parameters} * 8 bytes; \
         an LM iteration copies twice, for the step and for the candidate)",
        (points.len() * parameters) as f64 * 8.0 / 1e6,
        points.len()
    );

    // tenferro-ad turns every `jvp` seed it imports and every reverse-pass
    // cotangent into a program input, so a run binds them after the declared
    // inputs: each Laplace seed once per nested `jvp` call, in the order
    // `poisson::seed_inputs` gives, then the cotangent.
    let cotangent = Tensor::from_vec_col_major(vec![], vec![1.0])?;
    let seed_reads = poisson::seed_inputs(seeds.values());
    let mut differentiated = seed_reads.clone();
    differentiated.push(&cotangent);

    let compiled = Instant::now();
    let adam_program = if wants_adam {
        let trace = TraceContext::new();
        let model = Model::new(&trace, &sizes, None)?;
        let x = trace.input_concrete_shape(DType::F64, &[dim, points.len()])?;
        let program_seeds = poisson::Seeds::new(&trace, dim, points.len())?;
        let params = model.placeholders();
        let residual = poisson::residual(
            &model,
            &params,
            &x,
            &program_seeds,
            dim,
            points.n_col,
            points.n_bc,
        )?;
        let loss = residual.reduce_sum_squares(None)?.scale_real(0.5)?;
        let (loss_value, grads) = loss.value_and_grad(&params)?;
        let mut outputs: Vec<&TracedTensor> = vec![&loss_value];
        let grads: Vec<TracedTensor> = grads
            .into_iter()
            .map(|grad| grad.expect("every parameter reaches the residual"))
            .collect();
        outputs.extend(grads.iter());
        let mut handles = params.clone();
        handles.push(&x);
        Some(Program::new(&outputs, &handles)?)
    } else {
        None
    };

    let lm_program = if wants_lm {
        let trace = TraceContext::new();
        let model = Model::new(&trace, &sizes, Some(points.len()))?;
        let x = trace.input_concrete_shape(DType::F64, &[dim, points.len()])?;
        let lambda = trace.input_concrete_shape(DType::F64, &[])?;
        let program_seeds = poisson::Seeds::new(&trace, dim, points.len())?;
        let params = model.placeholders();
        let residual = poisson::residual(
            &model,
            &params,
            &x,
            &program_seeds,
            dim,
            points.n_col,
            points.n_bc,
        )?;
        let jacobian = poisson::jacobian_transposed(&residual, &params)?;
        let step = lm::solve_step(&jacobian, &residual, &lambda)?;
        let loss = residual.reduce_sum_squares(None)?.scale_real(0.5)?;
        let mut handles = params.clone();
        handles.extend([&x, &lambda]);
        Some(Program::new(
            &[&step.delta, &loss, &step.predicted],
            &handles,
        )?)
    } else {
        None
    };

    // The candidate residual runs as its own program: a second differentiated
    // residual in the same trace would have to cross the damped solve, which
    // `tenferro-ad` refuses (`UnsupportedAdRule` on the linalg op).
    let lm_candidate_program = if wants_lm {
        let trace = TraceContext::new();
        let model = Model::new(&trace, &sizes, Some(points.len()))?;
        let x = trace.input_concrete_shape(DType::F64, &[dim, points.len()])?;
        let program_seeds = poisson::Seeds::new(&trace, dim, points.len())?;
        let params = model.placeholders();
        let residual = poisson::residual(
            &model,
            &params,
            &x,
            &program_seeds,
            dim,
            points.n_col,
            points.n_bc,
        )?;
        let loss = residual.reduce_sum_squares(None)?.scale_real(0.5)?;
        let mut handles = params.clone();
        handles.push(&x);
        Some(Program::new(&[&loss], &handles)?)
    } else {
        None
    };

    let (evaluation_points, exact) =
        poisson::evaluation(dim, if dim == 1 { N_EVAL_1D } else { N_EVAL_2D })?;
    let evaluation_program = {
        let trace = TraceContext::new();
        let model = Model::new(&trace, &sizes, None)?;
        let x_eval =
            trace.input_concrete_shape(DType::F64, &[dim, evaluation_points.shape()[1]])?;
        let params = model.placeholders();
        let prediction = model.forward(&params, &x_eval)?;
        let mut handles = params.clone();
        handles.push(&x_eval);
        Program::new(&[&prediction], &handles)?
    };
    println!(
        "built and compiled the programs in {:.3} s",
        compiled.elapsed().as_secs_f64()
    );

    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine)?;

    let mut curves: Vec<(plotters::style::RGBColor, Vec<(f64, f64)>)> = Vec::new();
    if let Some(program) = &adam_program {
        let (theta, report) = train_adam(
            &runtime,
            &mut engine,
            program,
            &Run {
                points: &points.tensor,
                extras: &differentiated,
                model: &plain,
                seed,
                steps: adam_epochs,
            },
        )?;
        let rel_l2 = evaluate(
            &runtime,
            &mut engine,
            &evaluation_program,
            &evaluation_points,
            &theta,
            &exact,
        )?;
        println!(
            "summary: optimizer=adam dim={dim} steps={} seconds={:.2} loss={:.6e} rel_l2={:.6e}",
            report.steps, report.seconds, report.final_loss, rel_l2
        );
        curves.push((BLUE, report.history));
    }
    if let (Some(program), Some(candidate_program)) = (&lm_program, &lm_candidate_program) {
        let (theta, report) = train_lm(
            &runtime,
            &mut engine,
            program,
            candidate_program,
            &seed_reads,
            &Run {
                points: &points.tensor,
                extras: &differentiated,
                model: &plain,
                seed,
                steps: lm_iterations,
            },
        )?;
        let rel_l2 = evaluate(
            &runtime,
            &mut engine,
            &evaluation_program,
            &evaluation_points,
            &theta,
            &exact,
        )?;
        println!(
            "summary: optimizer=lm dim={dim} steps={} seconds={:.2} loss={:.6e} rel_l2={:.6e} \
             rejections={} factorization_retries={}",
            report.steps,
            report.seconds,
            report.final_loss,
            rel_l2,
            report.rejections,
            report.retries
        );
        curves.push((RED, report.history));
    }

    if let Some(path) = arg_value("--plot") {
        let references: Vec<(plotters::style::RGBColor, &[(f64, f64)])> = curves
            .iter()
            .map(|(colour, history)| (*colour, history.as_slice()))
            .collect();
        plot::write_loss_png(&path, &references)?;
        println!("saved loss curves to {path}");
    }
    Ok(())
}
