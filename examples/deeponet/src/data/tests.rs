use super::*;

#[test]
fn trapezoid_is_exact_for_a_linear_field() {
    let h = 1.0 / 10.0;
    let u: Vec<f64> = (0..=10).map(|j| j as f64 * h).collect();
    let g = cumulative_trapezoid(&u, h);
    for (j, &value) in g.iter().enumerate() {
        let y = j as f64 * h;
        assert!((value - 0.5 * y * y).abs() < 1e-12);
    }
}

#[test]
fn cholesky_factor_reproduces_the_jittered_covariance() {
    let mut engine = CpuEngine::serial();
    let xs: Vec<f64> = (0..9).map(|i| i as f64 / 8.0).collect();
    let n = xs.len();
    let k = covariance(&xs);
    let l = cholesky_factor(&xs, &mut engine).unwrap();
    for j in 0..n {
        for i in 0..n {
            let mut reproduced = 0.0;
            for p in 0..n {
                reproduced += l[i + p * n] * l[j + p * n];
            }
            let expected = k[i + j * n] + if i == j { JITTER } else { 0.0 };
            assert!(
                (reproduced - expected).abs() < 1e-12,
                "({i},{j}): {reproduced} vs {expected}"
            );
        }
    }
}

#[test]
fn sampling_is_reproducible_and_seeded() {
    let mut engine = CpuEngine::serial();
    let (train_a, test_a) = generate(2, 2, 42, 43, &mut engine).unwrap();
    let (train_b, test_b) = generate(2, 2, 42, 43, &mut engine).unwrap();
    assert_eq!(train_a.u, train_b.u);
    assert_eq!(train_a.g, train_b.g);
    assert_eq!(test_a.u, test_b.u);
    assert_eq!(test_a.g, test_b.g);
    assert_ne!(train_a.u, test_a.u, "the two streams must draw differently");
}

#[test]
fn test_set_is_independent_of_the_training_count() {
    let mut engine = CpuEngine::serial();
    let (_, test_a) = generate(3, 2, 42, 43, &mut engine).unwrap();
    let (_, test_b) = generate(7, 2, 42, 43, &mut engine).unwrap();
    assert_eq!(test_a.u, test_b.u);
    assert_eq!(test_a.g, test_b.g);
}

#[test]
fn minibatch_gathers_functions_in_column_major_order() {
    let mut engine = CpuEngine::serial();
    let (train, _) = generate(2, 1, 42, 43, &mut engine).unwrap();
    let (u, g) = train.minibatch(&[1, 0]).unwrap();
    let u = u.as_slice::<f64>().unwrap();
    let g = g.as_slice::<f64>().unwrap();
    for i in 0..SENSORS {
        assert_eq!(u[i * 2], train.u[1 + i * 2]);
        assert_eq!(u[1 + i * 2], train.u[i * 2]);
    }
    for j in 0..Y_POINTS {
        assert_eq!(g[j * 2], train.g[1 + j * 2]);
        assert_eq!(g[1 + j * 2], train.g[j * 2]);
    }
}
