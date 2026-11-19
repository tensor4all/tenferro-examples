use super::*;

/// `-e^{-ετ}/(1 + e^{-βε})`, the imaginary-time partner of `1/(iω_n - ε)`.
fn level_tau(beta: f64, epsilon: f64, tau: f64) -> f64 {
    -(-epsilon * tau).exp() / (1.0 + (-beta * epsilon).exp())
}

fn values(t: &Tensor) -> Vec<Complex64> {
    t.as_slice::<Complex64>().unwrap().to_vec()
}

const MAX_ITERATIONS: usize = 3000;

#[test]
fn iw_to_tau_transforms_a_single_level() {
    let (beta, epsilon, n_iw) = (10.0, 0.3, 512);
    let mut grid = Grid::new(beta, n_iw).unwrap();
    let mut engine = CpuEngine::serial();
    let f_iw = grid
        .host_function(|w| 1.0 / Complex64::new(-epsilon, w))
        .unwrap();
    let f_tau = values(&grid.iw_to_tau(&f_iw, &mut engine).unwrap());
    assert_eq!(f_tau.len(), 2 * n_iw);
    for (k, value) in f_tau.iter().enumerate() {
        let tau = k as f64 * beta / (2 * n_iw) as f64;
        assert!(value.im.abs() < 1e-12, "τ = {tau}: {value}");
        // The first neglected tail term, ε/(iω)², bounds the error.
        assert!(
            (value.re - level_tau(beta, epsilon, tau)).abs() < 2e-3,
            "τ = {tau}"
        );
    }
}

#[test]
fn tau_to_iw_transforms_a_particle_hole_symmetric_pair() {
    let (beta, epsilon, n_iw) = (10.0, 0.3, 512);
    let mut grid = Grid::new(beta, n_iw).unwrap();
    let mut engine = CpuEngine::serial();
    let n = 2 * n_iw;
    // Levels at ±ε: G(τ) = (g_ε(τ) + g_{-ε}(τ))/2, G(iω) = iω/((iω)² - ε²).
    let f_tau: Vec<Complex64> = (0..n)
        .map(|k| {
            let tau = k as f64 * beta / n as f64;
            let value = 0.5 * (level_tau(beta, epsilon, tau) + level_tau(beta, -epsilon, tau));
            Complex64::new(value, 0.0)
        })
        .collect();
    let f_tau = Tensor::from_vec_col_major([n], f_tau).unwrap();
    let f_iw = values(&grid.tau_to_iw(&f_tau, &mut engine).unwrap());
    // The error grows with ω_n: 3.8e-7 at n = 49, 5.5e-4 at n = 511.
    for (k, value) in f_iw.iter().take(50).enumerate() {
        let w = matsubara(beta, k);
        let exact = -w / (w * w + epsilon * epsilon);
        assert_eq!(value.re, 0.0);
        assert!(
            (value.im - exact).abs() < 1e-6,
            "n = {k}: {} vs {exact}",
            value.im
        );
    }
}

#[test]
fn zero_interaction_converges_to_the_semicircle() {
    let mut grid = Grid::new(20.0, 256).unwrap();
    let mut engine = CpuEngine::serial();
    let start = grid.atomic(1.0).unwrap();
    let solution = grid.solve(0.0, start, MAX_ITERATIONS, &mut engine).unwrap();
    assert!(solution.converged);
    let exact = values(&grid.semicircle().unwrap());
    for (got, want) in values(&solution.g).iter().zip(&exact) {
        assert!((got - want).norm() < 1e-8, "{got} vs {want}");
    }
    assert_eq!(grid.quasiparticle_weight(&solution.sigma).unwrap(), 1.0);
}

/// At `T = D/40` the sweeps of the example coexist for `2.60 D ≤ U ≤ 2.85 D`;
/// the sparse-ir IPT tutorial puts `U_{c2}` of the same model between 2.85 D and 2.9 D.
#[test]
fn metallic_and_insulating_solutions_coexist_at_low_temperature() {
    let mut grid = Grid::new(40.0, 1024).unwrap();
    let mut engine = CpuEngine::serial();
    let metallic = grid.semicircle().unwrap();
    let mut z = |u: f64, insulating_start: bool| {
        let start = if insulating_start {
            grid.atomic(u).unwrap()
        } else {
            metallic.duplicate().unwrap()
        };
        let solution = grid.solve(u, start, MAX_ITERATIONS, &mut engine).unwrap();
        assert!(solution.converged, "U = {u}");
        grid.quasiparticle_weight(&solution.sigma).unwrap()
    };
    let metal = z(2.8, false);
    let insulator = z(2.8, true);
    assert!(metal > 0.05, "metal Z = {metal}");
    assert!(insulator < 0.01, "insulator Z = {insulator}");

    // Outside the window the start does not matter.
    let low = z(2.4, true);
    let high = z(3.2, false);
    assert!(low > 0.05, "U = 2.4 from the insulator: Z = {low}");
    assert!(high < 0.01, "U = 3.2 from the metal: Z = {high}");
}
