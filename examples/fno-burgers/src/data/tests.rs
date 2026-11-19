use super::*;
use rand::SeedableRng;

fn cole_hopf(n: usize, t: f64) -> Vec<f64> {
    let a = 0.6 * (-NU * std::f64::consts::TAU.powi(2) * t).exp();
    (0..n)
        .map(|j| {
            let x = std::f64::consts::TAU * j as f64 / n as f64;
            2.0 * NU * std::f64::consts::TAU * a * x.sin() / (1.0 + a * x.cos())
        })
        .collect()
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn solver_preserves_the_spatial_mean() {
    let n = 128;
    let initial: Vec<f64> = (0..n)
        .map(|j| 0.2 + (std::f64::consts::TAU * j as f64 / n as f64).sin())
        .collect();
    let actual = Solver::new(n, REFERENCE_DT)
        .unwrap()
        .solve(&initial, 500)
        .unwrap();
    let error = (actual.iter().sum::<f64>() / n as f64 - 0.2).abs();
    assert!(error < 1e-12, "mean drift: {error:e}");
}

#[test]
fn solver_matches_the_exact_cole_hopf_solution_at_t_one() {
    let n = 128;
    let actual = Solver::new(n, REFERENCE_DT)
        .unwrap()
        .solve(&cole_hopf(n, 0.0), 500)
        .unwrap();
    let error = max_error(&actual, &cole_hopf(n, 1.0));
    assert!(error < 2e-6, "Cole-Hopf max error: {error:e}");
}

#[test]
fn reference_timestep_converges_to_cole_hopf() {
    let n = 128;
    let initial = cole_hopf(n, 0.0);
    let exact = cole_hopf(n, 0.2);
    let mut errors = Vec::new();
    for steps in [50, 100, 200] {
        let actual = Solver::new(n, 0.2 / steps as f64)
            .unwrap()
            .solve(&initial, steps)
            .unwrap();
        errors.push(max_error(&actual, &exact));
    }
    eprintln!("Cole-Hopf dt=0.004/0.002/0.001 max errors: {errors:?}");
    assert!(errors[1] < 0.4 * errors[0], "errors: {errors:?}");
    assert!(errors[2] < 0.4 * errors[1], "errors: {errors:?}");
    // Allow roundoff headroom over the documented 2.14e-10 reference residual.
    assert!(errors[1] < 1e-9, "default dt error: {:e}", errors[1]);
}

#[test]
fn gaussian_reference_converges_when_the_timestep_is_halved() {
    let mut coarse_solver = Solver::new(REFERENCE_N, REFERENCE_DT).unwrap();
    let initial = sample_field(&mut coarse_solver, &mut StdRng::seed_from_u64(42)).unwrap();
    let coarse = coarse_solver.solve(&initial, 500).unwrap();
    let fine = Solver::new(REFERENCE_N, REFERENCE_DT / 2.0)
        .unwrap()
        .solve(&initial, 1000)
        .unwrap();
    let difference = coarse
        .iter()
        .zip(&fine)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let norm = fine.iter().map(|v| v * v).sum::<f64>();
    let relative_error = (difference / norm).sqrt();
    eprintln!("Gaussian N=1024 dt=0.002 vs 0.001 relative L2 residual: {relative_error:e}");
    assert!(
        relative_error < 1e-7,
        "Gaussian timestep residual: {relative_error:e}"
    );
}

#[test]
fn initial_high_modes_are_filtered_before_the_quadratic_product() {
    let n = 64;
    let initial: Vec<f64> = (0..n)
        .map(|j| {
            let x = std::f64::consts::TAU * j as f64 / n as f64;
            (24.0 * x).cos() + (26.0 * x).cos()
        })
        .collect();
    let actual = Solver::new(n, 0.001).unwrap().solve(&initial, 1).unwrap();
    let error = actual.iter().copied().map(f64::abs).fold(0.0, f64::max);
    assert!(error < 1e-12, "initial high-mode alias: {error:e}");
}

#[test]
fn gaussian_fields_have_zero_mean_and_independent_seeded_streams() {
    let mut solver = Solver::new(REFERENCE_N, REFERENCE_DT).unwrap();
    let mut first = StdRng::seed_from_u64(42);
    let a = sample_field(&mut solver, &mut first).unwrap();
    let next = sample_field(&mut solver, &mut first).unwrap();
    let repeated = sample_field(&mut solver, &mut StdRng::seed_from_u64(42)).unwrap();
    let other = sample_field(&mut solver, &mut StdRng::seed_from_u64(43)).unwrap();
    assert_eq!(a, repeated);
    assert_ne!(a, next);
    assert_ne!(a, other);
    let mean = a.iter().sum::<f64>() / a.len() as f64;
    assert!(mean.abs() < 1e-12, "Gaussian field mean: {mean:e}");
}

#[test]
fn dataset_pairs_are_reproducible_and_nested_across_resolutions() {
    let dataset = Dataset::generate(2, 42).unwrap();
    let repeated = Dataset::generate(1, 42).unwrap();
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.initial[..REFERENCE_N], repeated.initial);
    assert_eq!(dataset.final_values[..REFERENCE_N], repeated.final_values);
    let (fine_u, fine_v) = dataset.batch(&[1, 0], 512).unwrap();
    assert_eq!(fine_u.shape(), &[512, 2, 1]);
    assert_eq!(fine_v.shape(), &[512, 2, 1]);
    for n in [128, 256] {
        let (u, v) = dataset.batch(&[1, 0], n).unwrap();
        let u = u.as_slice::<f64>().unwrap();
        let v = v.as_slice::<f64>().unwrap();
        let fine_u = fine_u.as_slice::<f64>().unwrap();
        let fine_v = fine_v.as_slice::<f64>().unwrap();
        for batch in 0..2 {
            for j in 0..n {
                assert_eq!(u[j + n * batch], fine_u[j * (512 / n) + 512 * batch]);
                assert_eq!(v[j + n * batch], fine_v[j * (512 / n) + 512 * batch]);
                let sample = 1 - batch;
                assert_eq!(
                    u[j + n * batch],
                    dataset.initial[sample * REFERENCE_N + j * (REFERENCE_N / n)]
                );
            }
        }
    }
}

#[test]
fn rms_uses_actual_dataset_values() {
    let dataset = Dataset {
        initial: vec![3.0; REFERENCE_N],
        final_values: vec![0.25; REFERENCE_N],
    };
    assert_eq!(dataset.rms(), (3.0, 0.25));
}

#[test]
fn rms_rejects_empty_input_or_target_storage() {
    for (initial, final_values) in [(vec![], vec![1.0]), (vec![1.0], vec![])] {
        let dataset = Dataset {
            initial,
            final_values,
        };
        assert!(std::panic::catch_unwind(|| dataset.rms()).is_err());
    }
}

#[test]
fn dataset_rejects_invalid_counts_grids_and_indices() {
    assert!(Dataset::generate(0, 42).is_err());
    assert!(Dataset::generate(usize::MAX, 42).is_err());
    let dataset = Dataset {
        initial: vec![1.0; REFERENCE_N],
        final_values: vec![0.5; REFERENCE_N],
    };
    for n in [0, 127, 2048] {
        assert!(dataset.batch(&[0], n).is_err(), "accepted n={n}");
    }
    assert!(dataset.batch(&[1], 128).is_err());
    assert!(dataset.batch(&[], 128).is_err());
}
