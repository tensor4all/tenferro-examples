use super::*;
use crate::run_single;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sciml_kit::func::Params;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{DType, GraphCompiler, TraceContext};
use tenferro_tensor::Tensor;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[test]
fn example_network_has_the_issue_layers() {
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &MLP_SIZES, TracedTensor::tanh).unwrap();
    let placeholders = net.placeholders();
    let shapes: Vec<Vec<usize>> = placeholders
        .iter()
        .map(|placeholder| placeholder.try_concrete_shape().unwrap())
        .collect();
    assert_eq!(
        shapes,
        vec![vec![2, 50], vec![50], vec![50, 2], vec![2]],
        "two layers give a weight and a bias each"
    );

    let values = net.initial_values(&mut StdRng::seed_from_u64(42)).unwrap();
    assert_eq!(values.len(), placeholders.len());
    for (value, placeholder) in values.iter().zip(&placeholders) {
        assert_eq!(
            Some(value.shape().to_vec()),
            placeholder.try_concrete_shape()
        );
    }
}

#[test]
fn rhs_is_the_network_applied_to_the_cube() -> TestResult {
    let trace = TraceContext::new();
    // A two-layer net has no hidden layer, so `forward` is the linear map
    // `x W`; a weight buffer holding `Aᵀ` gives `Wᵀ z³ = A z³`.
    let net = Mlp::new(&trace, &[2, 2], TracedTensor::tanh)?;
    let mut values = net.initial_values(&mut StdRng::seed_from_u64(0))?;
    values[0]
        .as_slice_mut::<f64>()?
        .copy_from_slice(&[-0.1, 2.0, -2.0, -0.1]);

    let z = trace.input_concrete_shape(DType::F64, &[1, 2])?;
    let out = rhs(&net, &z)?;
    let program = GraphCompiler::new().compile_many(&[&out])?;
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;

    let z_tensor = Tensor::from_vec_col_major(vec![1, 2], vec![2.0, -1.0])?;
    let mut inputs: Vec<&Tensor> = values.iter().collect();
    inputs.push(&z_tensor);
    let result = run_single(&runtime, &mut engine, &program, &inputs)?;
    let data = result.as_slice::<f64>()?;
    assert!((data[0] + 2.8).abs() < 1e-12, "got {data:?}");
    assert!((data[1] + 15.9).abs() < 1e-12, "got {data:?}");
    Ok(())
}
