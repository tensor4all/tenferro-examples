//! Tests for the two parameter layouts of the model.

use super::*;
use rand::SeedableRng;
use sciml_kit::func::Params;
use std::error::Error;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{DType, ExecutionEngine, GraphCompiler, TensorRead, TraceContext};

/// The plain shapes of `sizes`: one weight and one bias per layer.
fn plain_shapes(sizes: &[usize]) -> Vec<Vec<usize>> {
    sizes
        .windows(2)
        .flat_map(|pair| [vec![pair[1], pair[0]], vec![pair[1]]])
        .collect()
}

/// The per-point initial values are the plain ones, tiled along the point axis.
#[test]
fn per_point_initial_values_tile_the_plain_ones() -> std::result::Result<(), Box<dyn Error>> {
    let trace = TraceContext::new();
    let sizes = [2, 3, 3, 1];
    let plain = Model::new(&trace, &sizes, None)?;
    let point = Model::new(&trace, &sizes, Some(5))?;
    let plain_values = plain.initial_values(&mut rand::rngs::StdRng::seed_from_u64(4))?;
    let point_values = point.initial_values(&mut rand::rngs::StdRng::seed_from_u64(4))?;

    assert_eq!(plain_values.len(), point_values.len());
    for ((plain, point), shape) in plain_values
        .iter()
        .zip(point_values.iter())
        .zip(plain_shapes(&sizes))
    {
        assert_eq!(plain.shape().to_vec(), shape);
        let mut tiled = shape.clone();
        tiled.push(5);
        assert_eq!(point.shape().to_vec(), tiled);
        let plain_data = plain.as_slice::<f64>()?;
        let point_data = point.as_slice::<f64>()?;
        for (index, value) in point_data.iter().enumerate() {
            assert_eq!(*value, plain_data[index % plain_data.len()]);
        }
    }
    Ok(())
}

/// Both layouts evaluate the same function: feeding the tiled parameters to the
/// per-point model gives the plain model's output for every point.
#[test]
fn per_point_forward_matches_the_plain_forward() -> std::result::Result<(), Box<dyn Error>> {
    let (dim, points) = (2usize, 6usize);
    let trace = TraceContext::new();
    let sizes = [dim, 3, 3, 1];
    let plain = Model::new(&trace, &sizes, None)?;
    let point = Model::new(&trace, &sizes, Some(points))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(12);
    let theta = plain.initial_values(&mut rng)?;
    let tiled = tile(&theta, points)?;

    let x = trace.input_concrete_shape(DType::F64, &[dim, points])?;
    let plain_output = plain.forward(&plain.placeholders(), &x)?;
    let point_output = point.forward(&point.placeholders(), &x)?;
    let program = GraphCompiler::new().compile_many(&[&plain_output, &point_output])?;

    // Creation order: plain parameters, per-point parameters, x.
    let mut inputs: Vec<&Tensor> = theta.iter().collect();
    inputs.extend(tiled.iter());
    let x_values = Tensor::from_vec_col_major(
        vec![dim, points],
        (0..dim * points)
            .map(|index| 0.1 + 0.05 * (index % dim) as f64 + 0.13 * (index / dim) as f64)
            .collect(),
    )?;
    inputs.push(&x_values);
    let reads: Vec<TensorRead<'_>> = inputs
        .iter()
        .map(|&tensor| TensorRead::from_tensor(tensor))
        .collect();
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;
    let outputs = crate::run_prepared(&runtime, &mut engine, &prepared, &reads)?;

    let expected = outputs[0].as_slice::<f64>()?;
    let actual = outputs[1].as_slice::<f64>()?;
    assert_eq!(outputs[0].shape(), &[1, points]);
    assert_eq!(expected.len(), actual.len());
    for (index, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() <= 1e-14, "entry {index}: {e} vs {a}");
    }
    Ok(())
}

/// `tile` repeats each plain value along a new trailing axis.
#[test]
fn tile_repeats_along_a_new_axis() -> std::result::Result<(), Box<dyn Error>> {
    let plain = Tensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let tiled = tile(&[plain], 3)?;
    assert_eq!(tiled[0].shape(), &[2, 2, 3]);
    let data = tiled[0].as_slice::<f64>()?;
    for (index, value) in data.iter().enumerate() {
        assert_eq!(*value, [1.0, 2.0, 3.0, 4.0][index % 4]);
    }
    Ok(())
}

/// A parameter list that does not match the layer count is rejected.
#[test]
#[should_panic(expected = "one weight and one bias per layer")]
fn forward_checks_the_parameter_count() {
    let trace = TraceContext::new();
    let model = Model::new(&trace, &[1, 2, 1], None).unwrap();
    let params = model.placeholders();
    let x = trace.input_concrete_shape(DType::F64, &[4, 1]).unwrap();
    let _ = model.forward(&params[..2], &x);
}
