//! Tests for the point set, the forcing term and the Jacobian.

use super::*;
use crate::network::{tile, Model};
use sciml_kit::func::Params;
use std::error::Error;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, TensorRead, TraceContext, TracedTensor,
};

/// Run a single-output program on one explicit input and return its first output.
fn run_with_x(
    program: &tenferro_runtime::CompiledGraph,
    x: &Tensor,
) -> std::result::Result<Tensor, Box<dyn Error>> {
    let reads = [TensorRead::from_tensor(x)];
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    Ok(crate::run_prepared(&runtime, &mut engine, &prepared, &reads)?.remove(0))
}

/// Run one program whose inputs are all bound and return its outputs.
fn run_constants(
    program: &tenferro_runtime::CompiledGraph,
) -> std::result::Result<Vec<Tensor>, Box<dyn Error>> {
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let prepared = runtime.prepare(program, &[], &[engine.target()])?;
    Ok(crate::run_prepared(&runtime, &mut engine, &prepared, &[])?)
}

/// Collocation points are interior and boundary points are on the boundary.
#[test]
fn points_sit_where_they_should() -> std::result::Result<(), Box<dyn Error>> {
    for dim in [1usize, 2] {
        let points = points(dim, 16, 3, 7)?;
        let data = points.tensor.as_slice::<f64>()?;
        assert_eq!(points.tensor.shape(), &[dim, points.len()]);
        assert_eq!(points.n_col, 16);
        assert_eq!(points.n_bc, if dim == 1 { 2 } else { 12 });
        for j in 0..points.n_col {
            for d in 0..dim {
                let coordinate = data[d + dim * j];
                assert!((0.0..1.0).contains(&coordinate), "collocation {coordinate}");
            }
        }
        for j in points.n_col..points.len() {
            let on_boundary = (0..dim).any(|d| {
                let coordinate = data[d + dim * j];
                coordinate == 0.0 || coordinate == 1.0
            });
            assert!(on_boundary, "column {j} is not on the boundary");
        }
    }
    Ok(())
}

/// The same seed gives the same points, a different seed different ones.
#[test]
fn points_are_reproducible() -> std::result::Result<(), Box<dyn Error>> {
    let first = points(2, 8, 2, 21)?;
    let again = points(2, 8, 2, 21)?;
    let other = points(2, 8, 2, 22)?;
    assert_eq!(
        first.tensor.as_slice::<f64>()?,
        again.tensor.as_slice::<f64>()?
    );
    assert_ne!(
        first.tensor.as_slice::<f64>()?,
        other.tensor.as_slice::<f64>()?
    );
    Ok(())
}

/// The exact solution vanishes on the boundary and equals sin(pi x) inside.
#[test]
fn exact_solution_matches_the_problem() -> std::result::Result<(), Box<dyn Error>> {
    let (grid, exact) = evaluation(1, 5)?;
    let data = grid.as_slice::<f64>()?;
    assert_eq!(exact[0], 0.0);
    assert!(exact[4].abs() < 1e-15, "u(1) = {}", exact[4]);
    for j in 0..5 {
        let expected = (std::f64::consts::PI * data[j]).sin();
        assert!((exact[j] - expected).abs() < 1e-15);
    }
    let (_, exact_2d) = evaluation(2, 3)?;
    assert_eq!(exact_2d.len(), 9);
    assert!(
        (exact_2d[4] - 1.0).abs() < 1e-15,
        "u(1/2, 1/2) = {}",
        exact_2d[4]
    );
    Ok(())
}

/// The forcing term equals the analytic `pi^2 sin(pi x)` / `2 pi^2 sin sin`.
#[test]
fn forcing_matches_the_analytic_rhs() -> std::result::Result<(), Box<dyn Error>> {
    let pi = std::f64::consts::PI;
    for dim in [1usize, 2] {
        let points = points(dim, 9, 2, 5)?;
        let n = points.len();
        let trace = TraceContext::new();
        let x = trace.input_concrete_shape(DType::F64, &[dim, n])?;
        let program = GraphCompiler::new().compile_many(&[&forcing(&x, dim, n)?])?;
        let value = run_with_x(&program, &points.tensor)?;
        let got = value.as_slice::<f64>()?;
        let data = points.tensor.as_slice::<f64>()?;
        assert_eq!(value.shape(), &[1, n]);
        for j in 0..n {
            let sine = (pi * data[dim * j]).sin();
            let expected = if dim == 1 {
                pi * pi * sine
            } else {
                2.0 * pi * pi * sine * (pi * data[dim * j + 1]).sin()
            };
            assert!(
                (got[j] - expected).abs() < 1e-12,
                "dim {dim} column {j}: {} vs {expected}",
                got[j]
            );
        }
    }
    Ok(())
}

/// The batched Jacobian of the residual equals per-parameter JVPs.
///
/// This is the check the example exists to justify: `grad(sum(r), theta_b)`
/// with a per-point parameter copy must reproduce, entry by entry, the
/// directional derivatives taken one parameter at a time by forward mode.
///
/// Everything is baked as data-carrying constants, so the program needs no
/// binding: a `jvp` seed must carry data, and a program whose seeds are bound
/// takes no reads at all.
#[test]
fn batched_jacobian_matches_per_parameter_jvps() -> std::result::Result<(), Box<dyn Error>> {
    for dim in [1usize, 2] {
        let points = points(dim, 5, 2, 11)?;
        let n = points.len();
        let trace = TraceContext::new();
        let sizes = [dim, 2, 2, 1];
        let plain = Model::new(&trace, &sizes, None)?;
        let point = Model::new(&trace, &sizes, Some(n))?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let theta = plain.initial_values(&mut rng)?;
        let tiled = crate::network::tile(&theta, n)?;
        let constant = |tensor: &Tensor| -> std::result::Result<TracedTensor, Box<dyn Error>> {
            Ok(trace
                .from_vec_col_major(tensor.shape().to_vec(), tensor.as_slice::<f64>()?.to_vec())?)
        };
        let plain_values: Vec<TracedTensor> = theta
            .iter()
            .map(&constant)
            .collect::<std::result::Result<_, _>>()?;
        let point_values: Vec<TracedTensor> = tiled
            .iter()
            .map(&constant)
            .collect::<std::result::Result<_, _>>()?;
        let x = constant(&points.tensor)?;

        let plain_params: Vec<&TracedTensor> = plain_values.iter().collect();
        let point_params: Vec<&TracedTensor> = point_values.iter().collect();
        let seeds = Seeds::new(&trace, dim, n)?;
        let residual_plain = residual(
            &plain,
            &plain_params,
            &x,
            &seeds,
            dim,
            points.n_col,
            points.n_bc,
        )?;
        let residual_point = residual(
            &point,
            &point_params,
            &x,
            &seeds,
            dim,
            points.n_col,
            points.n_bc,
        )?;
        let jacobian = jacobian_transposed(&residual_point, &point_params)?;

        // One JVP per scalar parameter entry, each with a data-carrying seed.
        let mut columns = Vec::new();
        for (param, value) in plain_params.iter().zip(&theta) {
            let shape = value.shape().to_vec();
            let len: usize = shape.iter().product();
            for index in 0..len {
                let mut seed = vec![0.0_f64; len];
                seed[index] = 1.0;
                let seed = trace.from_vec_col_major(shape.clone(), seed)?;
                columns.push(residual_plain.jvp(param, &seed)?);
            }
        }

        let mut outputs: Vec<&TracedTensor> = vec![&jacobian];
        outputs.extend(columns.iter());
        let program = GraphCompiler::new().compile_many(&outputs)?;
        let results = run_constants(&program)?;

        let jacobian = results[0].as_slice::<f64>()?;
        assert_eq!(results[0].shape(), &[columns.len(), n]);
        let scale = jacobian
            .iter()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()))
            .max(1.0);
        for column in 0..columns.len() {
            let tangent = results[1 + column].as_slice::<f64>()?;
            assert_eq!(tangent.len(), n);
            for j in 0..n {
                let batched = jacobian[column + columns.len() * j];
                assert!(
                    (batched - tangent[j]).abs() <= 1e-10 * scale,
                    "dim {dim} entry ({column}, {j}): batched {batched} vs jvp {}",
                    tangent[j]
                );
            }
        }
    }
    Ok(())
}

/// The two-dimensional residual matches finite differences.
///
/// This is the check that the seeds bound to the program's AD inputs reach the
/// coordinate they belong to: a swapped pair would mix `u_xx` with `u_yy` and
/// still run.
#[test]
fn residual_matches_finite_differences_2d() -> std::result::Result<(), Box<dyn Error>> {
    let dim = 2;
    let n_col = 2048;
    let sizes = [dim, 32, 32, 1];
    let points = points(dim, n_col, 32, 3)?;
    let n = points.len();
    let h = 1e-4;
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);

    // The residual program, with per-point parameters and the Laplace seeds.
    let trace = TraceContext::new();
    let tiled_model = Model::new(&trace, &sizes, Some(n))?;
    let params = tiled_model.placeholders();
    let x = trace.input_concrete_shape(DType::F64, &[dim, n])?;
    let seeds = Seeds::new(&trace, dim, n)?;
    let r = residual(&tiled_model, &params, &x, &seeds, dim, n_col, points.n_bc)?;
    let mut handles = params.clone();
    handles.push(&x);
    let program = crate::Program::new(&[&r], &handles)?;

    // The same weights as plain parameters, for the finite-difference field.
    // Shift the biases away from zero: an all-zero bias could be folded away
    // without changing the network, and this check is meant to rule that out.
    let plain = {
        let trace = TraceContext::new();
        let model = Model::new(&trace, &sizes, None)?;
        let mut values = model.initial_values(&mut rng)?;
        for (index, tensor) in values.iter_mut().enumerate().skip(1).step_by(2) {
            let shifted: Vec<f64> = tensor
                .as_slice::<f64>()?
                .iter()
                .map(|value| value + 0.25 * index as f64)
                .collect();
            *tensor = Tensor::from_vec_col_major(tensor.shape().to_vec(), shifted)?;
        }
        values
    };
    let tiled = tile(&plain, n)?;

    let data = points.tensor.as_slice::<f64>()?;
    let columns_of = |start: usize, len: usize, shift: f64, axis: usize| -> Tensor {
        let mut values = vec![0.0_f64; dim * len];
        for j in 0..len {
            for a in 0..dim {
                let offset = if a == axis { shift } else { 0.0 };
                values[a + dim * j] = data[a + dim * (start + j)] + offset;
            }
        }
        Tensor::from_vec_col_major(vec![dim, len], values).expect("a column slice")
    };
    let field = |points: &Tensor| -> std::result::Result<Vec<f64>, Box<dyn Error>> {
        let trace = TraceContext::new();
        let model = Model::new(&trace, &sizes, None)?;
        let params = model.placeholders();
        let count = points.shape()[1];
        let grid = trace.input_concrete_shape(DType::F64, &[dim, count])?;
        let u = model.forward(&params, &grid)?;
        let mut handles = params.clone();
        handles.push(&grid);
        let u_program = crate::Program::new(&[&u], &handles)?;
        let mut values: Vec<&Tensor> = plain.iter().collect();
        values.push(points);
        let reads = u_program.reads(&values, &[]);
        let mut engine = CpuEngine::serial();
        let runtime = crate::cpu_runtime(&engine)?;
        let prepared = runtime.prepare(&u_program.graph, &reads, &[engine.target()])?;
        let output = crate::run_prepared(&runtime, &mut engine, &prepared, &reads)?.remove(0);
        Ok(output.as_slice::<f64>()?.to_vec())
    };

    let inside = columns_of(0, n_col, 0.0, 0);
    let u0 = field(&inside)?;
    let ux_plus = field(&columns_of(0, n_col, h, 0))?;
    let ux_minus = field(&columns_of(0, n_col, -h, 0))?;
    let uy_plus = field(&columns_of(0, n_col, h, 1))?;
    let uy_minus = field(&columns_of(0, n_col, -h, 1))?;
    let on_boundary = field(&columns_of(n_col, points.n_bc, 0.0, 0))?;

    let extra_values = seed_inputs(seeds.values());
    let mut values: Vec<&Tensor> = tiled.iter().collect();
    values.push(&points.tensor);
    let reads = program.reads(&values, &extra_values);
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let prepared = runtime.prepare(&program.graph, &reads, &[engine.target()])?;
    let output = crate::run_prepared(&runtime, &mut engine, &prepared, &reads)?.remove(0);
    let residual = output.as_slice::<f64>()?;

    let pde_scale = 1.0 / (n_col as f64).sqrt();
    let bc_scale = 1.0 / (points.n_bc as f64).sqrt();
    let pi = std::f64::consts::PI;
    let mut worst: f64 = 0.0;
    for j in 0..n_col {
        let laplacian =
            (ux_plus[j] + ux_minus[j] + uy_plus[j] + uy_minus[j] - 4.0 * u0[j]) / (h * h);
        let (px, py) = (data[dim * j], data[1 + dim * j]);
        let forcing = 2.0 * pi * pi * (pi * px).sin() * (pi * py).sin();
        let expected = (-laplacian - forcing) * pde_scale;
        worst = worst.max((expected - residual[j]).abs());
    }
    for j in 0..points.n_bc {
        let expected = on_boundary[j] * bc_scale;
        worst = worst.max((expected - residual[n_col + j]).abs());
    }
    assert!(
        worst < 1e-6,
        "the residual disagrees with the field by {worst}"
    );

    // The damped step adds the reverse pass, so check that the loss it reports
    // is the square of the same residual the residual program produced.
    let step_trace = TraceContext::new();
    let step_model = Model::new(&step_trace, &sizes, Some(n))?;
    let step_params = step_model.placeholders();
    let step_x = step_trace.input_concrete_shape(DType::F64, &[dim, n])?;
    let step_lambda = step_trace.input_concrete_shape(DType::F64, &[])?;
    let step_seeds = Seeds::new(&step_trace, dim, n)?;
    let step_residual = crate::poisson::residual(
        &step_model,
        &step_params,
        &step_x,
        &step_seeds,
        dim,
        n_col,
        points.n_bc,
    )?;
    let jacobian = jacobian_transposed(&step_residual, &step_params)?;
    let step = crate::lm::solve_step(&jacobian, &step_residual, &step_lambda)?;
    let step_loss = step_residual.reduce_sum_squares(None)?.scale_real(0.5)?;
    let mut step_handles = step_params.clone();
    step_handles.extend([&step_x, &step_lambda]);
    let step_program =
        crate::Program::new(&[&step.delta, &step_loss, &step.predicted], &step_handles)?;
    let cotangent = Tensor::from_vec_col_major(vec![], vec![1.0])?;
    let mut step_extras = seed_inputs(step_seeds.values());
    step_extras.push(&cotangent);
    let lambda_value = Tensor::from_vec_col_major(vec![], vec![1e-3])?;
    let mut step_values: Vec<&Tensor> = tiled.iter().collect();
    step_values.push(&points.tensor);
    step_values.push(&lambda_value);
    let step_reads = step_program.reads(&step_values, &step_extras);
    let mut step_engine = CpuEngine::serial();
    let step_runtime = crate::cpu_runtime(&step_engine)?;
    let step_prepared =
        step_runtime.prepare(&step_program.graph, &step_reads, &[step_engine.target()])?;
    let step_outputs =
        crate::run_prepared(&step_runtime, &mut step_engine, &step_prepared, &step_reads)?;
    let reported = step_outputs[1].as_slice::<f64>()?[0];
    let host: f64 = 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
    assert!(
        (reported - host).abs() < 1e-9 * host.abs().max(1.0),
        "the step program reports loss {reported}, the residual squares to {host}"
    );
    Ok(())
}

/// The loss gradient matches finite differences.
#[test]
fn loss_gradient_matches_finite_differences() -> std::result::Result<(), Box<dyn Error>> {
    let dim = 2;
    let n_col = 9;
    let sizes = [dim, 5, 5, 1];
    let points = points(dim, n_col, 2, 5)?;
    let n = points.len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(13);
    let trace = TraceContext::new();
    let model = Model::new(&trace, &sizes, None)?;
    let params = model.placeholders();
    let theta = model.initial_values(&mut rng)?;
    let x = trace.input_concrete_shape(DType::F64, &[dim, n])?;
    let seeds = Seeds::new(&trace, dim, n)?;
    let r = residual(&model, &params, &x, &seeds, dim, points.n_col, points.n_bc)?;
    let loss = r.reduce_sum_squares(None)?.scale_real(0.5)?;
    let (loss_value, grads) = loss.value_and_grad(&params)?;
    let mut outputs: Vec<&TracedTensor> = vec![&loss_value];
    let grads: Vec<TracedTensor> = grads
        .into_iter()
        .map(|grad| grad.expect("every parameter reaches the residual"))
        .collect();
    outputs.extend(grads.iter());
    let mut handles = params.clone();
    handles.push(&x);
    let program = crate::Program::new(&outputs, &handles)?;

    let cotangent = Tensor::from_vec_col_major(vec![], vec![1.0])?;
    let mut extras = seed_inputs(seeds.values());
    extras.push(&cotangent);
    let evaluate = |values: &[&Tensor]| -> std::result::Result<Vec<Tensor>, Box<dyn Error>> {
        let mut inputs: Vec<&Tensor> = values.to_vec();
        inputs.push(&points.tensor);
        let reads = program.reads(&inputs, &extras);
        let mut engine = CpuEngine::serial();
        let runtime = crate::cpu_runtime(&engine)?;
        let prepared = runtime.prepare(&program.graph, &reads, &[engine.target()])?;
        Ok(crate::run_prepared(
            &runtime,
            &mut engine,
            &prepared,
            &reads,
        )?)
    };

    let base: Vec<&Tensor> = theta.iter().collect();
    let outputs = evaluate(&base)?;
    let value = outputs[0].as_slice::<f64>()?[0];
    let gradient: Vec<f64> = outputs[1..]
        .iter()
        .flat_map(|tensor| tensor.as_slice::<f64>().expect("gradient data").to_vec())
        .collect();

    let mut shift = StdRng::seed_from_u64(17);
    let direction: Vec<f64> = (0..gradient.len())
        .map(|_| shift.gen::<f64>() - 0.5)
        .collect();
    let step = 1e-6;
    let moved = |sign: f64| -> std::result::Result<f64, Box<dyn Error>> {
        let mut inner = Vec::with_capacity(theta.len());
        let mut offset = 0;
        for tensor in &theta {
            let values = tensor.as_slice::<f64>()?;
            let shifted: Vec<f64> = values
                .iter()
                .enumerate()
                .map(|(index, value)| value + sign * step * direction[offset + index])
                .collect();
            offset += values.len();
            inner.push(Tensor::from_vec_col_major(
                tensor.shape().to_vec(),
                shifted,
            )?);
        }
        let refs: Vec<&Tensor> = inner.iter().collect();
        Ok(evaluate(&refs)?[0].as_slice::<f64>()?[0])
    };
    let finite_difference = (moved(1.0)? - moved(-1.0)?) / (2.0 * step);
    let directional: f64 = gradient
        .iter()
        .zip(direction.iter())
        .map(|(g, d)| g * d)
        .sum();
    assert!(value.is_finite(), "the loss is {value}");
    assert!(
        (directional - finite_difference).abs() < 1e-5 * directional.abs().max(1.0),
        "the gradient {directional} disagrees with finite differences {finite_difference}"
    );
    Ok(())
}
