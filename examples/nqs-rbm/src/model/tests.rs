use super::*;

fn log_ratio(model: &Model, spins: &[f64], fields: &[f64], flips: &[usize]) -> f64 {
    model.proposed_log(spins, fields, flips) - log_fields(fields)
}

#[test]
fn tied_derivatives_and_cached_ratios_match_full_evaluation() {
    let mut model = Model::new(4, 1, 2, Hamiltonian::Ising, 3);
    model.weights = vec![0.13, -0.27, 0.11, 0.34, -0.21, 0.09, 0.18, -0.07];
    let spins = vec![1.0, -1.0, 1.0, 1.0];
    let fields = model.fields(&spins);
    let explicit: f64 = (0..2)
        .flat_map(|f| (0..4).map(move |t| (f, t)))
        .map(|(f, t)| {
            (0..4)
                .map(|i| model.weights[f * 4 + i] * spins[(i + t) % 4])
                .sum::<f64>()
        })
        .map(|z| (2.0 * z.cosh()).ln())
        .sum();
    assert!((model.log_psi(&spins) - explicit).abs() < 1e-12);
    let gradient = model.derivatives(&spins, &fields);
    for (k, &actual) in gradient.iter().enumerate() {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.weights[k] += 1e-6;
        minus.weights[k] -= 1e-6;
        let fd = (plus.log_psi(&spins) - minus.log_psi(&spins)) / 2e-6;
        assert!(
            (actual - fd).abs() < 1e-8,
            "parameter {k}: {actual} != {fd}"
        );
    }
    for flips in [vec![0], vec![1, 3]] {
        let mut updated = spins.clone();
        for &i in &flips {
            updated[i] *= -1.0;
        }
        assert!(
            (log_ratio(&model, &spins, &fields, &flips)
                - (model.log_psi(&updated) - model.log_psi(&spins)))
            .abs()
                < 1e-12
        );
        let mut cached_spins = spins.clone();
        let mut cached_fields = fields.clone();
        model.apply_flips(&mut cached_spins, &mut cached_fields, &flips);
        assert_eq!(cached_spins, updated);
        for (a, b) in cached_fields.iter().zip(model.fields(&updated)) {
            assert!((a - b).abs() < 1e-12);
        }
    }
    let shifted = vec![spins[1], spins[2], spins[3], spins[0]];
    assert!((model.log_psi(&shifted) - explicit).abs() < 1e-12);
}

#[test]
fn local_energies_match_independent_dense_hamiltonian_action() {
    for hamiltonian in [Hamiltonian::Ising, Hamiltonian::Heisenberg] {
        let heisenberg = hamiltonian == Hamiltonian::Heisenberg;
        let model = Model::new(4, 1, 2, hamiltonian, 8);
        let mut action = [0.0; 16];
        let psi: Vec<f64> = (0..16u32)
            .map(|mask| {
                if heisenberg && mask.count_ones() != 2 {
                    return 0.0;
                }
                let s: Vec<f64> = (0..4)
                    .map(|i| if mask & (1 << i) == 0 { -1.0 } else { 1.0 })
                    .collect();
                model.log_psi(&s).exp()
            })
            .collect();
        for mask in 0..16usize {
            for i in 0..4 {
                let j = (i + 1) % 4;
                let aligned = (mask & (1 << i) == 0) == (mask & (1 << j) == 0);
                let zz = if aligned { 1.0 } else { -1.0 };
                action[mask] += if heisenberg {
                    zz * psi[mask]
                } else {
                    -zz * psi[mask]
                };
                if heisenberg {
                    if !aligned {
                        action[mask] -= 2.0 * psi[mask ^ (1 << i) ^ (1 << j)];
                    }
                } else {
                    action[mask] -= psi[mask ^ (1 << i)];
                }
            }
        }
        for mask in 0..16usize {
            if psi[mask] == 0.0 {
                continue;
            }
            let spins: Vec<f64> = (0..4)
                .map(|i| if mask & (1 << i) == 0 { -1.0 } else { 1.0 })
                .collect();
            let actual = model.local_energy(&spins, &model.fields(&spins));
            assert!(
                (actual - action[mask] / psi[mask]).abs() < 1e-11,
                "{hamiltonian:?}, state={mask}"
            );
        }
    }
}

#[test]
fn square_lattice_cache_and_translations_are_consistent() {
    let model = Model::new(4, 4, 2, Hamiltonian::Heisenberg, 27);
    assert_eq!(model.bonds.len(), 32);
    let spins: Vec<f64> = (0..16)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let mut shifted = vec![0.0; 16];
    for x in 0..4 {
        for y in 0..4 {
            shifted[x * 4 + y] = spins[((x + 1) % 4) * 4 + (y + 2) % 4];
        }
    }
    assert!((model.log_psi(&spins) - model.log_psi(&shifted)).abs() < 1e-12);
    let fields = model.fields(&spins);
    let ratio = log_ratio(&model, &spins, &fields, &[3, 10]);
    let mut updated = spins.clone();
    let mut cached = fields.clone();
    model.apply_flips(&mut updated, &mut cached, &[3, 10]);
    assert!((ratio - (model.log_psi(&updated) - model.log_psi(&spins))).abs() < 1e-12);
    for (a, b) in cached.iter().zip(model.fields(&updated)) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn length_two_directions_list_each_bond_once() {
    assert_eq!(
        Model::new(2, 1, 1, Hamiltonian::Heisenberg, 0).bonds,
        [(0, 1)]
    );
    assert_eq!(
        Model::new(2, 2, 1, Hamiltonian::Heisenberg, 0).bonds.len(),
        4
    );
}

#[test]
fn log_two_cosh_is_stable_for_large_arguments() {
    assert!((log_two_cosh(0.3) - (2.0 * 0.3_f64.cosh()).ln()).abs() < 1e-15);
    assert!((log_two_cosh(-800.0) - 800.0).abs() < 1e-12);
}
