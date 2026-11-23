// Runtime setup follows tenferro-fft 0.12 tests/support/mod.rs conventions.
use super::*;
use rand::{rngs::StdRng, SeedableRng};
use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{CompiledGraph, DType, ExecutionEngine, GraphCompiler, Runtime};
use tenferro_tensor::{Complex, TensorRead};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn run(
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> std::result::Result<Vec<Tensor>, Box<dyn std::error::Error>> {
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    let runtime = builder.build()?;
    let reads: Vec<_> = inputs.iter().map(|&t| TensorRead::from_tensor(t)).collect();
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    let outputs = runtime.run(&mut [&mut engine], &prepared, &reads)?;
    Ok(outputs
        .into_iter()
        .map(|output| output.into_tensor(&mut engine))
        .collect::<std::result::Result<_, _>>()?)
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|value| value.is_finite()));
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f64::max);
    assert!(
        max_error <= tolerance,
        "maximum absolute error {max_error:e}, tolerance {tolerance:e}"
    );
}

#[test]
fn real_parameter_prefix_and_complex_suffix_bind_at_multiple_resolutions() -> TestResult {
    let trace = TraceContext::new();
    let net = Fno::new(&trace, 2, 3)?;
    let placeholders = net.placeholders();
    let mut values = net.initial_values(&mut StdRng::seed_from_u64(42))?;
    assert_eq!(net.real_param_count(), 14);
    assert_eq!(placeholders.len(), 18);
    assert_eq!(values.len(), placeholders.len());
    for (index, (placeholder, value)) in placeholders.iter().zip(&values).enumerate() {
        assert_eq!(
            placeholder.try_concrete_shape().as_deref(),
            Some(value.shape())
        );
        let expected_dtype = if index < net.real_param_count() {
            DType::F64
        } else {
            DType::C64
        };
        assert_eq!(placeholder.dtype(), expected_dtype);
        assert_eq!(value.dtype(), expected_dtype);
    }
    for value in &mut values[..net.real_param_count()] {
        value.as_slice_mut::<f64>()?.fill(0.0);
    }
    // The last projection bias makes every output 0.75, independent of resolution.
    values[net.real_param_count() - 1]
        .as_slice_mut::<f64>()?
        .fill(0.75);
    for n in [8, 9, 12] {
        let x = trace.input_concrete_shape(DType::F64, &[n, 2, 1])?;
        let y = net.forward(&x, n, 2)?;
        let program = GraphCompiler::new().compile(&y)?;
        let input = Tensor::from_vec_col_major(vec![n, 2, 1], vec![0.3_f64; n * 2])?;
        let mut inputs: Vec<_> = values.iter().collect();
        inputs.push(&input);
        let output = run(&program, &inputs)?.remove(0);
        assert_eq!(output.shape(), &[n, 2, 1]);
        assert_close(output.as_slice::<f64>()?, &vec![0.75; n * 2], 1e-12);
    }
    Ok(())
}

#[test]
fn spectral_modes_mix_channels_and_discard_the_high_frequencies() -> TestResult {
    let modes = 3;
    for n in [8, 9] {
        let trace = TraceContext::new();
        let weight = trace.input_concrete_shape(DType::C64, &[modes, 2, 2])?;
        let x = trace.input_concrete_shape(DType::F64, &[n, 2, 2])?;
        let y = spectral(&x, &weight, n, modes)?;
        let program = GraphCompiler::new().compile(&y)?;
        // Column-major [mode, input channel, output channel].
        let weights = Tensor::from_vec_col_major(
            vec![modes, 2, 2],
            vec![
                Complex::new(1.0, 4.0),
                Complex::new(2.0, 0.5),
                Complex::new(0.3, -0.4),
                Complex::new(0.5, -3.0),
                Complex::new(-0.2, 0.6),
                Complex::new(1.2, 0.5),
                Complex::new(-0.25, 2.0),
                Complex::new(-0.4, 0.3),
                Complex::new(0.1, 0.2),
                Complex::new(2.0, -5.0),
                Complex::new(1.0, -0.7),
                Complex::new(-0.8, 0.6),
            ],
        )?;
        let mut input = Vec::with_capacity(n * 4);
        let mut expected = Vec::with_capacity(n * 4);
        for channel in 0..2 {
            for batch in 0..2 {
                let a = 0.3 + 0.2 * batch as f64;
                let d = -0.25 + 0.1 * batch as f64;
                for j in 0..n {
                    let t = std::f64::consts::TAU * j as f64 / n as f64;
                    if channel == 0 {
                        input.push(a + t.cos() + 0.4 * t.sin() + 0.8 * (3.0 * t).cos());
                        expected.push(
                            a + 0.5 * d + 2.08 * t.cos() + 0.34 * t.sin() + 0.84 * (2.0 * t).cos()
                                - 0.35 * (2.0 * t).sin(),
                        );
                    } else {
                        input.push(
                            d + 0.7 * (2.0 * t).cos() - 0.2 * t.sin() + 0.3 * (3.0 * t).sin(),
                        );
                        expected.push(
                            -0.25 * a + 2.0 * d
                                - 0.14 * t.cos()
                                - 0.66 * t.sin()
                                - 0.56 * (2.0 * t).cos()
                                - 0.42 * (2.0 * t).sin(),
                        );
                    }
                }
            }
        }
        let input = Tensor::from_vec_col_major(vec![n, 2, 2], input)?;
        let output = run(&program, &[&weights, &input])?.remove(0);
        assert_eq!(output.shape(), &[n, 2, 2]);
        assert_close(output.as_slice::<f64>()?, &expected, 2e-12);
    }
    Ok(())
}

#[test]
fn complex_spectral_gradient_matches_real_and_imaginary_finite_differences() -> TestResult {
    let modes = 3;
    let step = 1e-4;
    for n in [8, 9] {
        let trace = TraceContext::new();
        let weight = trace.input_concrete_shape(DType::C64, &[modes, 2, 2])?;
        let x = trace.input_concrete_shape(DType::F64, &[n, 2, 2])?;
        let target = trace.input_concrete_shape(DType::F64, &[n, 2, 2])?;
        let error = spectral(&x, &weight, n, modes)?.sub(&target)?;
        let loss = error
            .mul(&error)?
            .reduce_sum(Some(&[0, 1, 2]))?
            .scale_real(1.0 / (n * 4) as f64)?;
        let ad = AdContext::builder()
            .with_semantic_extension_rules(tenferro_fft::semantic_ad_rules()?)?
            .build()?;
        let grad = ad.grad(&loss, &weight)?;
        let program = GraphCompiler::new().compile_many(&[&loss, &grad])?;
        let base: Vec<_> = (0..modes * 4)
            .map(|i| {
                let t = i as f64 * 0.37;
                Complex::new(0.3 + t.sin(), -0.2 + (1.3 * t).cos())
            })
            .collect();
        let mut weights = Tensor::from_vec_col_major(vec![modes, 2, 2], base.clone())?;
        let input = Tensor::from_vec_col_major(
            vec![n, 2, 2],
            (0..n * 4)
                .map(|i| {
                    let t = 0.43 * i as f64;
                    0.4 + t.sin() + 0.3 * (1.7 * t).cos()
                })
                .collect(),
        )?;
        let truth = Tensor::from_vec_col_major(
            vec![n, 2, 2],
            (0..n * 4)
                .map(|i| {
                    let t = 0.29 * i as f64;
                    -0.1 + (1.2 * t).cos() - 0.6 * t.sin()
                })
                .collect(),
        )?;
        let output = run(&program, &[&weights, &input, &truth])?;
        let gradient = output[1].as_slice::<Complex<f64>>()?;
        let mut largest_error: f64 = 0.0;
        let mut interior_imaginary_gradient: f64 = 0.0;
        for index in 0..base.len() {
            for imaginary in [false, true] {
                let unit = if imaginary {
                    Complex::new(0.0, 1.0)
                } else {
                    Complex::new(1.0, 0.0)
                };
                weights.as_slice_mut::<Complex<f64>>()?[index] = base[index] + step * unit;
                let plus = run(&program, &[&weights, &input, &truth])?[0].as_slice::<f64>()?[0];
                weights.as_slice_mut::<Complex<f64>>()?[index] = base[index] - step * unit;
                let minus = run(&program, &[&weights, &input, &truth])?[0].as_slice::<f64>()?[0];
                weights.as_slice_mut::<Complex<f64>>()?[index] = base[index];
                let finite_difference = (plus - minus) / (2.0 * step);
                // Re(conj(g) * dw): g.re and g.im are the two real derivatives.
                let actual = if imaginary {
                    gradient[index].im
                } else {
                    gradient[index].re
                };
                assert!(actual.is_finite() && finite_difference.is_finite());
                largest_error = largest_error.max((actual - finite_difference).abs());
                if imaginary && index % modes != 0 {
                    interior_imaginary_gradient = interior_imaginary_gradient.max(actual.abs());
                }
                if imaginary && index % modes == 0 {
                    assert!(actual.abs() < 1e-12, "DC imaginary gradient {actual:e}");
                }
            }
        }
        assert!(
            interior_imaginary_gradient > 1e-3,
            "phase-sensitive fixture became degenerate"
        );
        assert!(
            largest_error < 2e-8,
            "n={n}: maximum gradient error {largest_error:e}"
        );
    }
    Ok(())
}

#[test]
fn invalid_width_and_mode_counts_are_errors() -> TestResult {
    let trace = TraceContext::new();
    assert!(Fno::new(&trace, 0, 3).is_err());
    assert!(Fno::new(&trace, 2, 0).is_err());
    let x = trace.input_concrete_shape(DType::F64, &[8, 2, 2])?;
    let weights = trace.input_concrete_shape(DType::C64, &[6, 2, 2])?;
    assert!(spectral(&x, &weights, 8, 6).is_err());
    Ok(())
}
