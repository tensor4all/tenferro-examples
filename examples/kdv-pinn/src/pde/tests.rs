use super::*;
use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{
    CompiledGraph, DType, ExecutionEngine, GraphCompiler, Result, TracedTensor,
};
use tenferro_tensor::Tensor;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn grad(output: &TracedTensor, input: &TracedTensor) -> Result<TracedTensor> {
    output.grad(input)
}

fn eval(tensor: &TracedTensor, bindings: &[(&TracedTensor, &Tensor)]) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let specs: Vec<(&TracedTensor, DType, &[usize])> = bindings
        .iter()
        .map(|(p, t)| (*p, t.dtype(), t.shape()))
        .collect();
    let program = compiler.compile_with_input_specs(tensor, &specs).unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine).unwrap();
    let inputs: Vec<&Tensor> = bindings.iter().map(|(_, tensor)| *tensor).collect();
    crate::run_single(&runtime, &mut engine, &program, &inputs).unwrap()
}

#[test]
fn kdv_residual_of_zero_solution_is_zero() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[2, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[2, 1])?;
    let zero = trace.from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    let u = x.mul(&x)?.mul(&x)?.mul(&t)?.mul(&zero)?;
    let r = kdv_residual(&u, &x, &t)?;

    let x_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    let t_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    let out = eval(&r, &[(&x, &x_tensor), (&t, &t_tensor)]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
    Ok(())
}

#[test]
fn jvp_higher_order_derivatives_of_cube() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let y = x.mul(&x)?.mul(&x)?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let y_x = y.jvp(&x, &ones)?;
    let y_xx = y_x.jvp(&x, &ones)?;
    let y_xxx = y_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
    let out_xx = eval(&y_xx, &[(&x, &x_tensor)]);
    // The third derivative of x^3 is constant, so it does not depend on `x`
    // and `x` must not be declared (`compile_with_input_specs` rejects it).
    let out_xxx = eval(&y_xxx, &[]);
    let xx = out_xx.as_slice::<f64>().unwrap();
    let xxx = out_xxx.as_slice::<f64>().unwrap();
    assert!((xx[0] - 6.0).abs() < 1e-6);
    assert!((xx[1] - 12.0).abs() < 1e-6);
    assert!((xx[2] - 18.0).abs() < 1e-6);
    assert!((xxx[0] - 6.0).abs() < 1e-6);
    assert!((xxx[1] - 6.0).abs() < 1e-6);
    assert!((xxx[2] - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn jvp_gives_elementwise_derivative() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let y = x.mul(&x)?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let dy_dx = y.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
    let out = eval(&dy_dx, &[(&x, &x_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[1] - 4.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn jvp_mixed_variable_second_derivative() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let u = z.exp()?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&u_xx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    let u_val = eval(&u, &[(&x, &x_tensor), (&t, &t_tensor)])
        .as_slice::<f64>()
        .unwrap()
        .to_vec();
    for i in 0..3 {
        assert!((data[i] - u_val[i]).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn jvp_pow_second_derivative() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let two = trace.from_vec_col_major(vec![3, 1], vec![2.0_f64; 3])?;
    let u = z.pow(&two)?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&u_xx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    for &v in out.as_slice::<f64>().unwrap() {
        assert!((v - 2.0).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn jvp_division_third_derivative() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let one = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u = one.div(&z.add(&one)?)?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;
    let u_xxx = u_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-0.5_f64, 0.0_f64, 0.5_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.0_f64, 0.0_f64])?;
    let out = eval(&u_xxx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    for (i, &actual) in data.iter().enumerate().take(3) {
        let xi = -0.5 + i as f64 * 0.5;
        let expected = -6.0 / (xi + 1.0).powi(4);
        assert!((actual - expected).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn jvp_pow_division_third_derivative() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let one = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let two = trace.from_vec_col_major(vec![3, 1], vec![2.0_f64; 3])?;
    let u = one.div(&z.add(&one)?.pow(&two)?)?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;
    let u_xxx = u_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-0.5_f64, 0.0_f64, 0.5_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.0_f64, 0.0_f64])?;
    let out = eval(&u_xxx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    for (i, &actual) in data.iter().enumerate().take(3) {
        let xi = -0.5 + i as f64 * 0.5;
        let expected = -24.0 / (xi + 1.0).powi(5);
        assert!((actual - expected).abs() < 1e-4);
    }
    Ok(())
}

#[test]
fn jvp_scale_real_third_derivative() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let three = trace.from_vec_col_major(vec![3, 1], vec![3.0_f64; 3])?;
    let u = z.scale_real(2.0)?.pow(&three)?;
    let ones = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;
    let u_xxx = u_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&u_xxx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    for &v in out.as_slice::<f64>().unwrap() {
        assert!((v - 48.0).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn kdv_residual_of_exact_solution_is_small() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let one = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let two = trace.from_vec_col_major(vec![3, 1], vec![2.0_f64; 3])?;
    let exp_z = z.exp()?;
    let neg_z = z.neg()?;
    let exp_neg_z = neg_z.exp()?;
    let cosh = exp_z.add(&exp_neg_z)?.div(&two)?;
    let sech2 = one.div(&cosh.pow(&two)?)?;
    let u = sech2.scale_real(2.0)?;
    let r = kdv_residual(&u, &x, &t)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&r, &[(&x, &x_tensor), (&t, &t_tensor)]);
    for &v in out.as_slice::<f64>().unwrap() {
        assert!(v.abs() < 1e-3, "residual of exact solution too large: {v}");
    }
    Ok(())
}

#[test]
fn third_derivative_of_cube() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let y = x.mul(&x)?.mul(&x)?.reduce_sum(Some(&[0, 1]))?;
    let y_x = grad(&y, &x)?;
    let y_xx = grad(&y_x.reduce_sum(Some(&[0, 1]))?, &x)?;
    let y_xxx = grad(&y_xx.reduce_sum(Some(&[0, 1]))?, &x)?;

    // The third derivative of x^3 is constant, so it does not depend on `x`
    // and `x` must not be declared (`compile_with_input_specs` rejects it).
    let result = eval(&y_xxx, &[]);
    let data = result.as_slice::<f64>().unwrap();
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 6.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
    Ok(())
}

/// One program for the loss and every gradient (`value_and_grad` +
/// `compile_many`, the training path) must agree with one program per
/// parameter gradient (the previous path). A wrong input binding order would
/// change these numbers, so this also guards `crate::bind_order`.
#[test]
fn value_and_grad_matches_per_parameter_grads() -> TestResult {
    use crate::loss;
    use rand::SeedableRng;
    use sciml_kit::func::Params;
    use sciml_kit::nn::Mlp;

    fn run_outputs(program: &CompiledGraph, tensors: &[&Tensor], order: &[usize]) -> Vec<Tensor> {
        let mut engine = tenferro_cpu::CpuEngine::serial();
        let runtime = crate::cpu_runtime(&engine).unwrap();
        let reads = crate::ordered_reads(tensors, order);
        let prepared = runtime
            .prepare(program, &reads, &[engine.target()])
            .unwrap();
        crate::run_prepared(&runtime, &mut engine, &prepared, &reads).unwrap()
    }

    let trace = tenferro_runtime::TraceContext::new();
    let net = Mlp::new(&trace, &[2, 8, 1], TracedTensor::tanh)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let params = net.initial_values(&mut rng)?;

    let x = trace.input_concrete_shape(DType::F64, &[4, 1])?;
    let t = trace.input_concrete_shape(DType::F64, &[4, 1])?;
    let u_true = trace.input_concrete_shape(DType::F64, &[4, 1])?;
    let xt = TracedTensor::stack(&[&x, &t], 1)?.reshape([4, 2])?;
    let u = net.forward(&xt)?;
    let residual = kdv_residual(&u, &x, &t)?;
    let total = loss::total_loss(&residual, &u, &u_true, &u, &u_true, 4, 4, 4, 1.0, 1.0, 1.0)?;

    let placeholders = net.placeholders();
    let bound: Vec<&TracedTensor> = placeholders
        .iter()
        .copied()
        .chain([&x, &t, &u_true])
        .collect();
    let (loss_value, per_parameter_options) = total.value_and_grad(&placeholders)?;
    let per_parameter_grads: Vec<TracedTensor> = per_parameter_options
        .into_iter()
        .map(|grad| grad.expect("every parameter reaches the loss"))
        .collect();

    let mut outputs: Vec<&TracedTensor> = vec![&loss_value];
    outputs.extend(per_parameter_grads.iter());
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&outputs)?;
    let order = crate::bind_order(&program, &bound);

    let x_tensor = Tensor::from_vec_col_major(vec![4, 1], vec![-1.0_f64, 0.0, 1.0, 2.0])?;
    let t_tensor = Tensor::from_vec_col_major(vec![4, 1], vec![0.0_f64, 0.5, 1.0, 0.2])?;
    let u_true_tensor = Tensor::from_vec_col_major(vec![4, 1], vec![0.4_f64, 0.1, 0.0, 0.3])?;
    let mut tensors: Vec<&Tensor> = params.iter().collect();
    tensors.extend([&x_tensor, &t_tensor, &u_true_tensor]);
    let single = run_outputs(&program, &tensors, &order);

    // The previous path: one compiled program per parameter gradient.
    for (index, placeholder) in placeholders.iter().enumerate() {
        let gradient = total.grad(placeholder)?;
        let program = compiler.compile_many(&[&gradient])?;
        let order = crate::bind_order(&program, &bound);
        let expected = run_outputs(&program, &tensors, &order).remove(0);
        let actual = single[index + 1].as_slice::<f64>()?;
        let expected = expected.as_slice::<f64>()?;
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-9, "gradient {index} differs: {a} vs {e}");
        }
    }

    // `value_and_grad` reports the same loss as a plain compile of the loss.
    let plain = compiler.compile_many(&[&total])?;
    let order = crate::bind_order(&plain, &bound);
    let loss_plain = run_outputs(&plain, &tensors, &order);
    assert!((single[0].as_slice::<f64>()?[0] - loss_plain[0].as_slice::<f64>()?[0]).abs() < 1e-12);
    Ok(())
}

/// A `jvp` tangent must be a data-carrying leaf, so the rank-0 constant +
/// `broadcast_in_dim` construction of the umbrella conventions cannot be used
/// for the ones. This pins the reason `ones_like` keeps a full-shape constant;
/// if tenferro starts accepting an operation result here, this test fails and
/// the ones can be simplified.
#[test]
fn jvp_rejects_a_broadcast_tangent() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 1])?;
    let u = x.mul(&x)?;
    let one = trace.from_vec_col_major(vec![], vec![1.0_f64])?;
    let ones = one.broadcast_in_dim(&[3, 1], &[])?;
    let error = u
        .jvp(&x, &ones)
        .expect_err("a broadcast tangent carries no data");
    assert!(
        format!("{error}").contains("concrete tensor data"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// `bind_order` must follow the program's input order, not the order the caller
/// listed the placeholders in. The constants the graph binds itself keep no
/// explicit input, so they do not shift the result either.
#[test]
fn bind_order_follows_the_program_not_the_argument_order() -> TestResult {
    let trace = tenferro_runtime::TraceContext::new();
    let a = trace.input_concrete_shape(DType::F64, &[2])?;
    let b = trace.input_concrete_shape(DType::F64, &[2])?;
    // A data-carrying constant created between the two placeholders: a bound
    // program input that consumes no explicit read.
    let c = trace.from_vec_col_major(vec![2], vec![1.0_f64, 1.0])?;
    let y = a.add(&b)?.mul(&c)?;
    let program = GraphCompiler::new().compile_many(&[&y])?;

    assert_eq!(
        crate::bind_order(&program, &[&a, &b]),
        vec![0, 1],
        "declaration order is program order here"
    );
    assert_eq!(
        crate::bind_order(&program, &[&b, &a]),
        vec![1, 0],
        "listing b first must still bind a before b"
    );
    Ok(())
}
