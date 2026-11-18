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

#[test]
fn step_decay_lr_has_three_stages() {
    let base = 1.0e-3;
    let total = 3000;
    // First half: full base rate.
    assert!((step_decay_lr(0, total, base) - base).abs() < 1e-18);
    assert!((step_decay_lr(total / 2 - 1, total, base) - base).abs() < 1e-18);
    // Second quarter: halved.
    assert!((step_decay_lr(total / 2, total, base) - base * 0.5).abs() < 1e-18);
    assert!((step_decay_lr(3 * total / 4 - 1, total, base) - base * 0.5).abs() < 1e-18);
    // Final quarter: quartered.
    assert!((step_decay_lr(3 * total / 4, total, base) - base * 0.25).abs() < 1e-18);
    assert!((step_decay_lr(total - 1, total, base) - base * 0.25).abs() < 1e-18);
}

fn rosenbrock(x: &[f64]) -> Result<(f64, Vec<f64>), ()> {
    let (a, b) = (x[0], x[1]);
    let f = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
    let g = vec![
        -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
        200.0 * (b - a * a),
    ];
    Ok((f, g))
}

#[test]
fn lbfgs_minimizes_rosenbrock() {
    let mut x = vec![-1.2, 1.0];
    let (mut f, mut g) = rosenbrock(&x).unwrap();
    let mut optimizer = Lbfgs::new(5);
    let mut iterations = 0;
    while f > 1e-20 && iterations < 200 {
        let outcome = optimizer.step(&mut x, &mut f, &mut g, rosenbrock).unwrap();
        assert!(
            matches!(outcome, LbfgsStep::Accepted { .. }),
            "{outcome:?} at {x:?}"
        );
        iterations += 1;
    }
    // the textbook start converges in a few dozen iterations
    assert!(iterations < 100, "{iterations} iterations");
    assert!(
        (x[0] - 1.0).abs() < 1e-8 && (x[1] - 1.0).abs() < 1e-8,
        "{x:?}"
    );
}

#[test]
fn lbfgs_first_direction_is_unit_steepest_descent() {
    let direction = Lbfgs::new(3).direction(&[3.0, -4.0]);
    assert!((direction[0] + 0.6).abs() < 1e-15 && (direction[1] - 0.8).abs() < 1e-15);
}

#[test]
fn lbfgs_is_exact_on_a_quadratic_after_one_pair() {
    // f = x·x / 2 in 2-D: one curvature pair gives H = I, so the second
    // step lands on the minimum.
    let quadratic = |x: &[f64]| -> Result<(f64, Vec<f64>), ()> {
        Ok((0.5 * (x[0] * x[0] + x[1] * x[1]), x.to_vec()))
    };
    let mut x = vec![3.0, 4.0];
    let (mut f, mut g) = quadratic(&x).unwrap();
    let mut optimizer = Lbfgs::new(2);
    optimizer.step(&mut x, &mut f, &mut g, quadratic).unwrap();
    optimizer.step(&mut x, &mut f, &mut g, quadratic).unwrap();
    assert!(f < 1e-28, "{f}");
}

#[test]
fn lbfgs_reports_a_failed_line_search_and_keeps_the_iterate() {
    // A loss that never decreases along the reported descent direction: every
    // trial fails the Armijo condition.
    let stubborn = |_: &[f64]| -> Result<(f64, Vec<f64>), ()> { Ok((2.0, vec![1.0, 0.0])) };
    let mut x = vec![0.5, -0.5];
    let (mut f, mut g) = (1.0, vec![1.0, 0.0]);
    let mut optimizer = Lbfgs::new(3);
    let outcome = optimizer.step(&mut x, &mut f, &mut g, stubborn).unwrap();
    assert_eq!(outcome, LbfgsStep::LineSearchFailed { evaluations: 30 });
    assert_eq!((x, f, g), (vec![0.5, -0.5], 1.0, vec![1.0, 0.0]));
}

#[test]
fn lbfgs_propagates_an_evaluation_error() {
    let failing = |_: &[f64]| -> Result<(f64, Vec<f64>), &'static str> { Err("engine") };
    let mut x = vec![1.0];
    let (mut f, mut g) = (1.0, vec![1.0]);
    let result = Lbfgs::new(1).step(&mut x, &mut f, &mut g, failing);
    assert_eq!(result, Err("engine"));
    assert_eq!(x, vec![1.0]);
}

#[test]
fn flatten_and_unflatten_round_trip_in_parameter_order() {
    let mut params = [tensor(vec![2], vec![1.0, 2.0]), tensor(vec![1], vec![3.0])];
    assert_eq!(flatten(&params), vec![1.0, 2.0, 3.0]);
    unflatten(&[4.0, 5.0, 6.0], &mut params);
    assert_eq!(params[0].as_slice::<f64>().unwrap(), &[4.0, 5.0]);
    assert_eq!(params[1].as_slice::<f64>().unwrap(), &[6.0]);
}
