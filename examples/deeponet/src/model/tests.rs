use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn forward_contracts_the_two_networks() {
    let trace = TraceContext::new();
    let net = DeepONet::new(&trace, 8).unwrap();
    let u = trace.input_concrete_shape(DType::F64, &[5, 8]).unwrap();
    let y = trace.input_concrete_shape(DType::F64, &[11, 1]).unwrap();
    let pred = net.forward(&u, &y).unwrap();
    assert_eq!(pred.try_concrete_shape(), Some(vec![5, 11]));
}

#[test]
fn b0_is_the_last_parameter_with_a_zero_initial_value() {
    let trace = TraceContext::new();
    let net = DeepONet::new(&trace, 4).unwrap();
    let placeholders = net.placeholders();
    let mut rng = StdRng::seed_from_u64(42);
    let values = net.initial_values(&mut rng).unwrap();

    assert_eq!(placeholders.len(), values.len());
    // Three layers per network, each a weight and a bias, plus b0.
    assert_eq!(placeholders.len(), 3 * 2 + 3 * 2 + 1);

    let b0 = placeholders.last().unwrap();
    assert_eq!(b0.try_concrete_shape(), Some(vec![1]));
    let b0_value = values.last().unwrap().as_slice::<f64>().unwrap();
    assert_eq!(b0_value, &[0.0]);
}
