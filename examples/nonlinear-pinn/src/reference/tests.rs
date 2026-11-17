use super::*;

/// Burgers' exact solution through the Cole-Hopf transform:
/// `u(x, t) = ∫ u0(x - η) φ0(x - η) G(η) dη / ∫ φ0(x - η) G(η) dη` with
/// `φ0(y) = exp(-cos(pi y) / (2 pi nu))` and `G(η) = exp(-η² / (4 nu t))`,
/// by the trapezoidal rule in log-scaled weights.
fn cole_hopf(x: f64, t: f64) -> f64 {
    let (pi, nu) = (std::f64::consts::PI, crate::problem::NU);
    let half_width = 12.0 * (4.0 * nu * t).sqrt();
    let points = 40_001;
    let log_weight =
        |eta: f64| -(pi * (x - eta)).cos() / (2.0 * pi * nu) - eta * eta / (4.0 * nu * t);
    let etas: Vec<f64> = (0..points)
        .map(|i| -half_width + 2.0 * half_width * i as f64 / (points - 1) as f64)
        .collect();
    let top = etas
        .iter()
        .map(|&e| log_weight(e))
        .fold(f64::NEG_INFINITY, f64::max);
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for &eta in &etas {
        let w = (log_weight(eta) - top).exp();
        numerator += -(pi * (x - eta)).sin() * w;
        denominator += w;
    }
    numerator / denominator
}

#[test]
fn burgers_reference_matches_cole_hopf() {
    let reference = solve(Problem::Burgers, 1024, 2e-4, 0.05, 0.5);
    // t = 0.25 (smooth) and t = 0.5 (steep front at x = 0)
    for (s, tolerance) in [(5, 1e-6), (10, 1e-4)] {
        let t = reference.t[s];
        for j in (0..1024).step_by(37) {
            let x = reference.x[j];
            let error = (reference.u[s][j] - cole_hopf(x, t)).abs();
            assert!(error < tolerance, "t={t} x={x}: |u - u_exact| = {error:e}");
        }
    }
}

#[test]
fn allen_cahn_reference_converges_in_resolution() {
    let coarse = solve(Problem::AllenCahn, 1024, 2e-4, 0.25, 1.0);
    let fine = solve(Problem::AllenCahn, 2048, 1e-4, 0.25, 1.0);
    for s in 1..coarse.t.len() {
        let max_difference = (0..1024)
            .map(|j| (coarse.u[s][j] - fine.u[s][2 * j]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_difference < 1e-3,
            "t={}: {max_difference:e}",
            coarse.t[s]
        );
    }
}

#[test]
fn reference_starts_from_the_initial_condition() {
    let reference = solve(Problem::AllenCahn, 256, 1e-3, 0.5, 0.5);
    assert_eq!(reference.t, vec![0.0, 0.5]);
    // the first record is the rfft/irfft round trip of the initial data
    for (j, &x) in reference.x.iter().enumerate() {
        assert!((reference.u[0][j] - Problem::AllenCahn.initial(x)).abs() < 1e-12);
    }
}
