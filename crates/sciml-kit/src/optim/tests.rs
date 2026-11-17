use super::*;
use tenferro_tensor::Tensor;

fn tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape, data).unwrap()
}

#[test]
fn sgd_updates_correctly() {
    let mut params = [tensor(vec![2], vec![1.0, 2.0])];
    let grads = [tensor(vec![2], vec![0.5, 1.0])];
    Sgd::new(0.1).step(&mut params, &grads);
    let data = params[0].as_slice::<f64>().unwrap();
    assert!((data[0] - 0.95).abs() < 1e-12);
    assert!((data[1] - 1.9).abs() < 1e-12);
}

#[test]
fn adam_first_step_is_bias_corrected() {
    // With a constant gradient the first bias-corrected step is
    // `-lr * g / (|g| + eps)`, i.e. `-lr` in the direction of `g`.
    let mut params = [tensor(vec![2], vec![1.0, 2.0])];
    let grads = [tensor(vec![2], vec![0.5, 1.0])];
    Adam::new(0.1).step(&mut params, &grads);
    let data = params[0].as_slice::<f64>().unwrap();
    assert!((data[0] - (1.0 - 0.1)).abs() < 1e-6, "got {}", data[0]);
    assert!((data[1] - (2.0 - 0.1)).abs() < 1e-6, "got {}", data[1]);
}

#[test]
fn adam_set_lr_changes_step_size() {
    // After lowering the learning rate to zero, a step must leave the
    // parameters unchanged, proving `set_lr` overrides the constructor value.
    let mut params = [tensor(vec![2], vec![1.0, 2.0])];
    let grads = [tensor(vec![2], vec![0.5, 1.0])];
    let mut optim = Adam::new(0.1);
    optim.set_lr(0.0);
    optim.step(&mut params, &grads);
    let data = params[0].as_slice::<f64>().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-12);
    assert!((data[1] - 2.0).abs() < 1e-12);
}

#[test]
fn adam_matches_a_step_by_step_reference() {
    // Two steps with the same gradient move the parameter by exactly `lr` each
    // time: Adam's bias correction makes the first two steps of a constant
    // gradient equal. A moment buffer holding the *corrected* moment instead of
    // the raw one gives 0.876 here instead of 0.8 (end-to-end: it made the PINN
    // diverge at ~10x the intended step size).
    let mut params = [tensor(vec![1], vec![1.0])];
    let mut optim = Adam::new(0.1);
    for step in 1..=2 {
        optim.step(&mut params, &[tensor(vec![1], vec![0.5])]);
        let expected = 1.0 - 0.1 * step as f64;
        let actual = params[0].as_slice::<f64>().unwrap()[0];
        assert!(
            (actual - expected).abs() < 1e-6,
            "step {step}: expected {expected}, got {actual}"
        );
    }
    assert_eq!(optim.m.len(), 1);
    assert_eq!(optim.v.len(), 1);
}

#[test]
fn adam_handles_complex_parameters() {
    // The second moment is `g * conj(g)`, so after one step with a purely
    // imaginary gradient the parameter moves by `-lr` along that direction:
    // `p -= lr * g / (|g| + eps)`.
    let mut params = [Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(0.5, -1.0)],
    )
    .unwrap()];
    let grads = [Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 2.0), Complex64::new(0.0, -1.0)],
    )
    .unwrap()];
    Adam::new(0.1).step(&mut params, &grads);
    let data = params[0].as_slice::<Complex64>().unwrap();
    assert_eq!(params[0].dtype(), DType::C64);
    assert!((data[0].re - 1.0).abs() < 1e-6, "got {:?}", data[0]);
    assert!((data[0].im - 0.9).abs() < 1e-6, "got {:?}", data[0]);
    assert!((data[1].re - 0.5).abs() < 1e-6, "got {:?}", data[1]);
    assert!((data[1].im + 0.9).abs() < 1e-6, "got {:?}", data[1]);
}

#[test]
#[should_panic(expected = "same length")]
fn sgd_rejects_a_parameter_and_gradient_length_mismatch() {
    let mut params = [tensor(vec![2], vec![1.0, 2.0])];
    let grads = [tensor(vec![1], vec![0.5])];
    Sgd::new(0.1).step(&mut params, &grads);
}

#[test]
#[should_panic(expected = "moment buffer")]
fn adam_rejects_a_parameter_that_changes_length() {
    let mut optim = Adam::new(0.1);
    let mut first = [tensor(vec![2], vec![1.0, 2.0])];
    optim.step(&mut first, &[tensor(vec![2], vec![0.5, 1.0])]);
    let mut second = [tensor(vec![1], vec![1.0])];
    optim.step(&mut second, &[tensor(vec![1], vec![0.5])]);
}
