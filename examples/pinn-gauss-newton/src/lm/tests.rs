//! Tests for the damped solve, the accept/reject decision and the candidate
//! parameters.

use super::*;
use std::error::Error;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, TensorRead, TraceContext};

/// The accept/reject and damping table.
#[test]
fn rho_accept_reject_table() {
    // Good model, large gain ratio: accept and shrink the damping.
    let decision = decide(1.0, 10.0, 1.0, 9.0);
    assert!(decision.accept);
    assert!((decision.rho - 1.0).abs() < 1e-15);
    assert!((decision.next_lambda - 1.0 / 3.0).abs() < 1e-15);

    // Middling gain ratio: accept and keep the damping.
    let decision = decide(1.0, 10.0, 6.0, 8.0);
    assert!(decision.accept);
    assert_eq!(decision.next_lambda, 1.0);

    // Poor but positive gain ratio: accept and grow the damping.
    let decision = decide(1.0, 10.0, 9.6, 8.0);
    assert!(decision.accept);
    assert_eq!(decision.next_lambda, 2.0);

    // Negative gain ratio: reject and grow the damping tenfold.
    let decision = decide(1.0, 10.0, 12.0, 8.0);
    assert!(!decision.accept);
    assert!(decision.rho < 0.0);
    assert_eq!(decision.next_lambda, 10.0);

    // No predicted reduction (and no information): reject.
    let decision = decide(1.0, 10.0, 1.0, 0.0);
    assert!(!decision.accept);
    assert_eq!(decision.rho, f64::NEG_INFINITY);
    assert_eq!(decision.next_lambda, 10.0);

    // A factorization that produced NaNs instead of an error: reject too.
    let decision = decide(1.0, f64::NAN, 1.0, f64::NAN);
    assert!(!decision.accept);
    assert_eq!(decision.next_lambda, 10.0);

    // The damping stays inside its bounds.
    assert_eq!(decide(5e5, 10.0, 12.0, 8.0).next_lambda, LAMBDA_MAX);
    assert_eq!(decide(1e-9, 10.0, 1.0, 9.0).next_lambda, LAMBDA_MIN);
    assert_eq!(decide(LAMBDA_MAX, 10.0, 12.0, 8.0).next_lambda, LAMBDA_MAX);
}

/// Run one graph with only constant inputs and return its outputs.
fn run_constants(
    program: &tenferro_runtime::CompiledGraph,
) -> std::result::Result<Vec<tenferro_tensor::Tensor>, Box<dyn Error>> {
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let prepared = runtime.prepare(program, &[], &[engine.target()])?;
    Ok(crate::run_prepared(&runtime, &mut engine, &prepared, &[])?)
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// The step solves `(J^T J + lambda I) delta = -J^T r`, damped and undamped.
#[test]
fn step_solves_the_damped_system() -> std::result::Result<(), Box<dyn Error>> {
    let (p, n) = (6usize, 10usize);
    // A is the Jacobian of a linear residual, [n, p] column-major: ten rows of
    // six integer coefficients, the first six upper triangular with the
    // diagonal 2..7, so the columns are independent and `J^T J` is positive
    // definite with room to spare (its Cholesky factor's diagonal runs from
    // 2.6 to 6.7, and integers are exact in `f64`). The undamped case below is
    // only meaningful against a matrix that really is definite: a draw that
    // lands within an ulp of singular makes Cholesky fail at `lambda = 0` on
    // whichever platform rounds the last bit differently.
    let rows = [
        [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 4.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 5.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 7.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    ];
    let a: Vec<f64> = (0..n * p).map(|index| rows[index % n][index / n]).collect();
    // The right-hand side is integer-valued too, so nothing here depends on the
    // platform's `sin`/`cos`.
    let theta: Vec<f64> = (0..p).map(|index| 0.4 - 0.05 * index as f64).collect();
    let b: Vec<f64> = (0..n)
        .map(|index| 0.25 * (index % 5) as f64 - 0.5)
        .collect();
    let r: Vec<f64> = (0..n)
        .map(|row| (0..p).map(|col| a[row + n * col] * theta[col]).sum::<f64>() - b[row])
        .collect();
    // J^T is [p, n]: entry (col, row) = A[row, col].
    let jt: Vec<f64> = (0..p * n)
        .map(|index| {
            let (col, row) = (index % p, index / p);
            a[row + n * col]
        })
        .collect();
    let normal: Vec<f64> = (0..p * p)
        .map(|index| {
            let (row, col) = (index % p, index / p);
            (0..n).map(|k| a[k + n * row] * a[k + n * col]).sum::<f64>()
        })
        .collect();
    let gradient: Vec<f64> = (0..p)
        .map(|row| (0..n).map(|k| a[k + n * row] * r[k]).sum::<f64>())
        .collect();

    let trace = TraceContext::new();
    let jt = trace.from_vec_col_major(vec![p, n], jt)?;
    let r = trace.from_vec_col_major(vec![n, 1], r)?;
    for lambda_value in [0.0, 0.5] {
        let lambda = trace.from_vec_col_major(vec![], vec![lambda_value])?;
        let step = solve_step(&jt, &r, &lambda)?;
        let program = GraphCompiler::new().compile_many(&[&step.delta, &step.predicted])?;
        let outputs = run_constants(&program)?;

        let delta = outputs[0].as_slice::<f64>()?;
        assert_eq!(delta.len(), p);
        assert!(
            delta.iter().all(|value| value.is_finite()),
            "lambda {lambda_value}: the step is not finite"
        );
        let residual: Vec<f64> = (0..p)
            .map(|row| {
                (0..p)
                    .map(|col| {
                        let diagonal = if row == col { lambda_value } else { 0.0 };
                        (normal[row + p * col] + diagonal) * delta[col]
                    })
                    .sum::<f64>()
                    + gradient[row]
            })
            .collect();
        assert!(
            norm(&residual) <= 1e-10 * norm(&gradient),
            "lambda {lambda_value}: |(H + lambda I) delta + g| = {}",
            norm(&residual)
        );
        assert!(outputs[1].as_slice::<f64>()?[0] > 0.0);
    }
    Ok(())
}

/// A factorization that cannot produce numbers is reported as a numerical
/// failure, and the same prepared program runs once the inputs are finite.
///
/// The damping factor is a program input, so the retry re-prepares nothing: it
/// binds new values to the same prepared program.
#[test]
fn numerical_failure_is_classified_and_retryable() -> std::result::Result<(), Box<dyn Error>> {
    let (p, n) = (3usize, 4usize);
    let trace = TraceContext::new();
    let jt = trace.input_concrete_shape(tenferro_runtime::DType::F64, &[p, n])?;
    let r = trace.input_concrete_shape(tenferro_runtime::DType::F64, &[n, 1])?;
    let lambda = trace.input_concrete_shape(tenferro_runtime::DType::F64, &[])?;
    let step = solve_step(&jt, &r, &lambda)?;
    let program = GraphCompiler::new().compile_many(&[&step.delta, &step.predicted])?;

    let finite_jt = tenferro_tensor::Tensor::from_vec_col_major(
        vec![p, n],
        (0..p * n).map(|index| 0.5 + 0.3 * index as f64).collect(),
    )?;
    let broken_jt = tenferro_tensor::Tensor::from_vec_col_major(vec![p, n], vec![f64::NAN; p * n])?;
    let r_values =
        tenferro_tensor::Tensor::from_vec_col_major(vec![n, 1], vec![1.0, -1.0, 0.5, 0.25])?;
    let zero = tenferro_tensor::Tensor::from_vec_col_major(vec![], vec![0.0])?;
    let damped = tenferro_tensor::Tensor::from_vec_col_major(vec![], vec![1e-2])?;

    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let reads = [
        TensorRead::from_tensor(&broken_jt),
        TensorRead::from_tensor(&r_values),
        TensorRead::from_tensor(&zero),
    ];
    let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;
    match crate::run_prepared(&runtime, &mut engine, &prepared, &reads) {
        Err(error) => assert!(
            is_numerical_failure(&error),
            "expected a numerical failure, got {error}"
        ),
        Ok(outputs) => {
            let delta = outputs[0].as_slice::<f64>()?;
            let predicted = outputs[1].as_slice::<f64>()?[0];
            assert!(
                delta.iter().any(|value| !value.is_finite()) || predicted <= 0.0,
                "a non-finite Jacobian gave a usable step: {delta:?}, predicted {predicted}"
            );
        }
    }

    let reads = [
        TensorRead::from_tensor(&finite_jt),
        TensorRead::from_tensor(&r_values),
        TensorRead::from_tensor(&damped),
    ];
    let outputs = crate::run_prepared(&runtime, &mut engine, &prepared, &reads)?;
    let delta = outputs[0].as_slice::<f64>()?;
    assert!(delta.iter().all(|value| value.is_finite()), "{delta:?}");
    assert!(outputs[1].as_slice::<f64>()?[0] > 0.0);
    Ok(())
}

/// `stepped` walks the parameters in order and adds the matching entry.
#[test]
fn stepped_adds_in_parameter_order() -> std::result::Result<(), Box<dyn Error>> {
    let params = vec![
        tenferro_tensor::Tensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0])?,
        tenferro_tensor::Tensor::from_vec_col_major(vec![1], vec![3.0])?,
    ];
    let stepped = stepped(&params, &[0.5, -1.0, 2.0])?;
    assert_eq!(stepped[0].as_slice::<f64>()?, &[1.5, 1.0]);
    assert_eq!(stepped[1].as_slice::<f64>()?, &[5.0]);
    // The input list is left alone, so a rejected step keeps the old values.
    assert_eq!(params[0].as_slice::<f64>()?, &[1.0, 2.0]);
    Ok(())
}
