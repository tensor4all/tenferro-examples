use super::*;

/// Residual of `(S + lambda diag(S) + shift I) x = F`, computed directly from
/// the sample covariances.
fn max_residual(g: &[f64], e: &[f64], p: usize, lambda: f64, shift: f64, x: &[f64]) -> f64 {
    let ns = e.len();
    let means: Vec<f64> = (0..p)
        .map(|i| (0..ns).map(|s| g[s * p + i]).sum::<f64>() / ns as f64)
        .collect();
    let mean_e = e.iter().sum::<f64>() / ns as f64;
    (0..p)
        .map(|i| {
            let cov = |j: usize| {
                (0..ns)
                    .map(|s| (g[s * p + i] - means[i]) * (g[s * p + j] - means[j]))
                    .sum::<f64>()
                    / ns as f64
            };
            let lhs = (0..p).map(|j| cov(j) * x[j]).sum::<f64>() + (lambda * cov(i) + shift) * x[i];
            let rhs = (0..ns)
                .map(|s| (g[s * p + i] - means[i]) * (e[s] - mean_e))
                .sum::<f64>()
                / ns as f64;
            (lhs - rhs).abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn parameter_space_sr_solves_covariance_equations() -> Result<(), Box<dyn Error>> {
    let p = 3;
    let gradients = vec![
        0.2, 0.7, -0.3, 0.4, -0.2, 0.9, -0.1, 0.5, 0.2, 0.3, 0.8, -0.7, -0.6, 0.1, 0.4,
    ];
    let energies = vec![-1.2, -0.8, -1.1, -0.5, -1.5];
    let step = sr_step(
        &gradients,
        &energies,
        p,
        0.07,
        &mut CpuEngine::serial(),
        1e-3,
    )?;
    let residual = max_residual(&gradients, &energies, p, 0.07, 1e-3, &step);
    assert!(residual < 1e-10, "residual={residual}");
    Ok(())
}

#[test]
fn sample_space_sr_solves_the_same_regularized_equations() -> Result<(), Box<dyn Error>> {
    let p = 7;
    let ns = 4;
    let g: Vec<f64> = (0..p * ns).map(|i| ((i * 13 + 3) as f64).sin()).collect();
    let e = vec![-1.2, -0.6, -0.9, -1.7];
    let x = sr_step(&g, &e, p, 0.13, &mut CpuEngine::serial(), 1e-3)?;
    let residual = max_residual(&g, &e, p, 0.13, 1e-3, &x);
    assert!(residual < 1e-10, "residual={residual}");
    Ok(())
}

#[test]
fn sr_rejects_mismatched_gradient_length() {
    let error = sr_step(
        &[0.0; 5],
        &[1.0, 2.0],
        3,
        0.1,
        &mut CpuEngine::serial(),
        1e-3,
    )
    .expect_err("length mismatch must fail");
    assert!(error.to_string().contains("5 gradient entries"));
}

#[test]
fn sr_step_bound_also_controls_under_sampled_parameter_directions() {
    let gradients = vec![1.0, 0.0, -1.0, 0.0];
    let delta = vec![0.0, 10.0];
    let rate = step_rate(&gradients, &delta, 0.002);
    assert!(rate * rms(&delta) <= 0.002 + 1e-14);
    assert!(delta.iter().all(|d| rate * d.abs() <= 0.002 + 1e-14));
    let rms_rate = step_rate(&gradients, &delta, 0.0);
    assert!((rms_rate * rms(&delta) - 0.002).abs() < 1e-12);
}

#[test]
fn damping_decays_to_its_floor() {
    assert_eq!(damping(0), 100.0);
    assert!((damping(1) - 90.0).abs() < 1e-12);
    assert_eq!(damping(1000), MIN_DAMPING);
    assert_eq!(damping(usize::MAX), MIN_DAMPING);
}
