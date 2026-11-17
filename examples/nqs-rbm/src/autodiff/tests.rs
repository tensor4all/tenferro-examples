use super::*;
use crate::model::Hamiltonian;

const A: [f64; 3] = [0.07, -0.03, 0.11];
const B: [f64; 2] = [-0.05, 0.09];
// W has shape [2, 3], with each column contiguous.
const W: [f64; 6] = [0.3, -0.4, 0.2, 0.1, -0.15, 0.25];

/// `log sum_h exp(a.s + b.h + h^T W s)` over all hidden spins `h = +-1`.
fn explicit_log_psi(spins: &[f64], a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    let visible: f64 = a.iter().zip(spins).map(|(a, spin)| a * spin).sum();
    (0..1 << b.len())
        .map(|mask| {
            let mut exponent = visible;
            for (j, bias) in b.iter().enumerate() {
                let hidden = if mask & (1 << j) == 0 { -1.0 } else { 1.0 };
                exponent += hidden * bias;
                for (i, spin) in spins.iter().enumerate() {
                    exponent += hidden * w[j + b.len() * i] * spin;
                }
            }
            exponent.exp()
        })
        .sum::<f64>()
        .ln()
}

#[test]
fn amplitudes_match_explicit_hidden_spin_sum() -> Result<(), Box<dyn Error>> {
    let runtime = EagerRuntime::new()?;
    let mut engine = CpuEngine::serial();
    let configurations: Vec<f64> = (0..8)
        .flat_map(|mask| (0..3).map(move |i| if mask & (1 << i) == 0 { -1.0 } else { 1.0 }))
        .collect();
    let spins = runtime.constant_from(Tensor::from_vec_col_major(
        vec![3, 8],
        configurations.clone(),
    )?)?;
    let a = runtime.variable_from(Tensor::from_vec_col_major(vec![3, 1], A.to_vec())?)?;
    let b = runtime.variable_from(Tensor::from_vec_col_major(vec![2, 1], B.to_vec())?)?;
    let w = runtime.variable_from(Tensor::from_vec_col_major(vec![2, 3], W.to_vec())?)?;
    let result = runtime.with_eager_session(&mut engine, |session| {
        rbm_log_psi(session, &spins, &a, &b, &w)
    })?;
    assert_eq!(result.shape(), &[8]);
    for (value, configuration) in result
        .value()?
        .as_slice::<f64>()?
        .iter()
        .zip(configurations.chunks(3))
    {
        let expected = explicit_log_psi(configuration, &A, &B, &W);
        assert!(
            (value - expected).abs() < 1e-12,
            "log amplitude: {value}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn parameter_gradients_match_independent_finite_differences() -> Result<(), Box<dyn Error>> {
    let runtime = EagerRuntime::new()?;
    let mut engine = CpuEngine::serial();
    let configuration = [1.0, -1.0, 1.0];
    let spins = runtime.constant_from(Tensor::from_vec_col_major(
        vec![3, 1],
        configuration.to_vec(),
    )?)?;
    let a = runtime.variable_from(Tensor::from_vec_col_major(vec![3, 1], A.to_vec())?)?;
    let b = runtime.variable_from(Tensor::from_vec_col_major(vec![2, 1], B.to_vec())?)?;
    let w = runtime.variable_from(Tensor::from_vec_col_major(vec![2, 3], W.to_vec())?)?;
    runtime.with_eager_session(&mut engine, |session| {
        let output = rbm_log_psi(session, &spins, &a, &b, &w)?;
        let loss = session.reduce_sum(&output, None)?;
        session.backward(&loss)
    })?;
    let epsilon = 1e-6;
    let parameters = [A.to_vec(), B.to_vec(), W.to_vec()];
    for (group, parameter) in [&a, &b, &w].iter().enumerate() {
        let gradient = parameter.grad()?.expect("parameter gradient");
        for (index, actual) in gradient.as_slice::<f64>()?.iter().enumerate() {
            let mut plus = parameters.clone();
            let mut minus = parameters.clone();
            plus[group][index] += epsilon;
            minus[group][index] -= epsilon;
            let expected = (explicit_log_psi(&configuration, &plus[0], &plus[1], &plus[2])
                - explicit_log_psi(&configuration, &minus[0], &minus[1], &minus[2]))
                / (2.0 * epsilon);
            assert!(
                (actual - expected).abs() < 1e-8,
                "gradient {group}:{index}: {actual}, expected {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn tied_gradient_matches_analytic_derivatives() -> Result<(), Box<dyn Error>> {
    for (lx, ly, hamiltonian) in [(5, 1, Hamiltonian::Ising), (4, 4, Hamiltonian::Heisenberg)] {
        let mut model = Model::new(lx, ly, 3, hamiltonian, 79);
        // Weights of order one make tanh nonlinear, unlike the initial ones.
        for (k, w) in model.weights.iter_mut().enumerate() {
            *w = (k as f64 * 0.7).sin();
        }
        let spins: Vec<f64> = (0..model.n())
            .map(|i| if (i * 7) % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let expected = model.derivatives(&spins, &model.fields(&spins));
        let actual = tied_gradient(&model, &spins)?;
        assert_eq!(actual.len(), expected.len());
        for (k, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "{lx}x{ly} weight {k}: {a} != {e}");
        }
    }
    Ok(())
}
