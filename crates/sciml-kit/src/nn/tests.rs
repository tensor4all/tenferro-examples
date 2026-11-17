use super::*;
use crate::func::Params;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tenferro_runtime::{CompiledGraph, GraphCompiler, TracedTensor};

#[test]
fn mlp_forward_shape() {
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &[2, 16, 16, 1], TracedTensor::tanh).unwrap();
    let x = trace.input_concrete_shape(DType::F64, &[4, 2]).unwrap();
    let y = net.forward(&x).unwrap();
    assert_eq!(y.rank(), 2);
    assert_eq!(y.try_concrete_shape(), Some(vec![4, 1]));
}

#[test]
fn activation_is_a_parameter() {
    fn identity(x: &TracedTensor) -> Result<TracedTensor> {
        Ok(x.clone())
    }
    fn operation_count(program: &CompiledGraph) -> usize {
        program.program().operations().count()
    }

    let trace = TraceContext::new();
    let x = trace.input_concrete_shape(DType::F64, &[3, 2]).unwrap();
    let with_tanh = Mlp::new(&trace, &[2, 4, 1], TracedTensor::tanh).unwrap();
    let with_identity = Mlp::new(&trace, &[2, 4, 1], identity).unwrap();
    let tanh_program = GraphCompiler::new()
        .compile_many(&[&with_tanh.forward(&x).unwrap()])
        .unwrap();
    let identity_program = GraphCompiler::new()
        .compile_many(&[&with_identity.forward(&x).unwrap()])
        .unwrap();
    assert!(
        operation_count(&tanh_program) > operation_count(&identity_program),
        "the activation passed to `Mlp::new` must reach the graph"
    );
}

#[test]
fn placeholders_are_program_inputs() {
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &[2, 8, 1], TracedTensor::tanh).unwrap();
    let x = trace.input_concrete_shape(DType::F64, &[5, 2]).unwrap();
    let y = net.forward(&x).unwrap();
    let program = GraphCompiler::new().compile_many(&[&y]).unwrap();

    let mut slots: Vec<usize> = net
        .placeholders()
        .iter()
        .map(|placeholder| {
            let key = placeholder
                .input_key()
                .expect("a parameter placeholder has an input key");
            program
                .input_key_index(&key)
                .expect("a parameter is a program input")
        })
        .collect();
    assert_eq!(slots.len(), 4, "two layers give a weight and a bias each");
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(slots.len(), 4, "every parameter has its own input slot");
}

#[test]
fn initial_values_match_placeholders_in_order() {
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &[2, 8, 1], TracedTensor::tanh).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let placeholders = net.placeholders();
    let values = net.initial_values(&mut rng).unwrap();
    assert_eq!(values.len(), placeholders.len());
    for (value, placeholder) in values.iter().zip(placeholders.iter()) {
        assert_eq!(
            Some(value.shape().to_vec()),
            placeholder.try_concrete_shape(),
            "initial value shape must match its placeholder"
        );
    }
}

#[test]
fn xavier_weights_are_bounded_and_biases_are_zero() {
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &[2, 8, 1], TracedTensor::tanh).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let values = net.initial_values(&mut rng).unwrap();

    let mut saw_nonzero_weight = false;
    for (index, value) in values.iter().enumerate() {
        if index % 2 == 1 {
            // Bias.
            assert!(
                value.as_slice::<f64>().unwrap().iter().all(|&b| b == 0.0),
                "biases start at zero"
            );
            continue;
        }
        // Weight, initialized uniformly in ±sqrt(6 / (fan_in + fan_out)).
        let shape = value.shape();
        let bound = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
        for &w in value.as_slice::<f64>().unwrap() {
            assert!(w.abs() <= bound, "weight {w} outside ±{bound}");
            saw_nonzero_weight |= w != 0.0;
        }
    }
    assert!(saw_nonzero_weight, "weights are randomized, not zeros");
}
