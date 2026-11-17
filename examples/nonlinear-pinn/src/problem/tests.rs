use super::*;
use tenferro_cpu::{runtime_engine_registration, CpuEngine};
use tenferro_runtime::{DType, ExecutionEngine, GraphCompiler, Runtime, TensorRead, TraceContext};
use tenferro_tensor::Tensor;

/// Compile `build(x, t, ones)` for columns `x`, `t` and run it once.
fn evaluate(
    x: &[f64],
    t: &[f64],
    build: impl Fn(&TracedTensor, &TracedTensor, &TracedTensor) -> Result<TracedTensor>,
) -> Vec<f64> {
    let n = x.len();
    let trace = TraceContext::new();
    let x_p = trace.input_concrete_shape(DType::F64, &[n, 1]).unwrap();
    let t_p = trace.input_concrete_shape(DType::F64, &[n, 1]).unwrap();
    let ones = trace
        .from_vec_col_major(vec![n, 1], vec![1.0_f64; n])
        .unwrap();
    let out = build(&x_p, &t_p, &ones).unwrap();
    let program = GraphCompiler::new().compile_many(&[&out]).unwrap();
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&engine).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let data = [
        Tensor::from_vec_col_major(vec![n, 1], x.to_vec()).unwrap(),
        Tensor::from_vec_col_major(vec![n, 1], t.to_vec()).unwrap(),
    ];
    let reads: Vec<TensorRead<'_>> = data.iter().map(TensorRead::from_tensor).collect();
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .unwrap();
    let result = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .unwrap()
        .remove(0)
        .into_tensor(&mut engine)
        .unwrap();
    result.as_slice::<f64>().unwrap().to_vec()
}

/// `u = sin(pi x) exp(-t)`: `u_t = -u`, `u_x = pi cos(pi x) exp(-t)`,
/// `u_xx = -pi² u`.
#[test]
fn residuals_match_the_analytic_derivatives() {
    let pi = std::f64::consts::PI;
    let x = [-0.9, -0.3, 0.1, 0.7];
    let t = [0.0, 0.25, 0.5, 1.0];
    for problem in [Problem::Burgers, Problem::AllenCahn] {
        let residual = evaluate(&x, &t, |x, t, ones| {
            let u = x.scale_real(pi)?.sin()?.mul(&t.neg()?.exp()?)?;
            problem.residual(&u, x, t, ones)
        });
        for i in 0..x.len() {
            let u = (pi * x[i]).sin() * (-t[i]).exp();
            let u_x = pi * (pi * x[i]).cos() * (-t[i]).exp();
            let expected = match problem {
                Problem::Burgers => -u + u * u_x + NU * pi * pi * u,
                Problem::AllenCahn => -u + EPSILON * pi * pi * u + 5.0 * u * u * u - 5.0 * u,
            };
            assert!(
                (residual[i] - expected).abs() < 1e-12,
                "{problem:?} at x={}: {} vs {expected}",
                x[i],
                residual[i]
            );
        }
    }
}

#[test]
fn allen_cahn_features_are_periodic_in_x() {
    let features = evaluate(&[-1.0, 1.0], &[0.3, 0.3], |x, t, _| {
        Problem::AllenCahn.features(x, t)
    });
    // column-major [2, 3]: rows are the two points
    for column in 0..3 {
        let (left, right) = (features[2 * column], features[2 * column + 1]);
        assert!(
            (left - right).abs() < 1e-15,
            "feature {column}: {left} vs {right}"
        );
    }
}

#[test]
fn initial_data_and_coefficients() {
    assert!((Problem::Burgers.initial(0.5) + 1.0).abs() < 1e-15);
    assert!((Problem::AllenCahn.initial(1.0) + 1.0).abs() < 1e-15);
    assert_eq!(Problem::parse("allen-cahn"), Some(Problem::AllenCahn));
    assert_eq!(Problem::parse("kdv"), None);
    assert_eq!(Problem::Burgers.diffusion(), NU);
}
