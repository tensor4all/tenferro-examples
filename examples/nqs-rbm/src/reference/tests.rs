use super::*;

fn zero_weight_model(lx: usize, ly: usize, hamiltonian: Hamiltonian) -> Model {
    let mut model = Model::new(lx, ly, 1, hamiltonian, 0);
    model.weights.fill(0.0);
    model
}

#[test]
fn closed_forms_match_dense_diagonalization() -> Result<(), Box<dyn Error>> {
    for n in [4, 6, 8] {
        let ising = exact_check(&zero_weight_model(n, 1, Hamiltonian::Ising))?;
        assert!(
            (ising.ground_energy - ising_chain(n)).abs() < 1e-10,
            "Ising N={n}: {} vs {}",
            ising.ground_energy,
            ising_chain(n)
        );
        let heisenberg = exact_check(&zero_weight_model(n, 1, Hamiltonian::Heisenberg))?;
        let bethe = heisenberg_chain(n)?;
        assert!(
            (heisenberg.ground_energy - bethe).abs() < 1e-10,
            "Heisenberg N={n}: {} vs {bethe}",
            heisenberg.ground_energy
        );
    }
    Ok(())
}

#[test]
fn references_match_known_values() -> Result<(), Box<dyn Error>> {
    // Four-site ring: E = -2 per site in Pauli units.
    assert!((heisenberg_chain(4)? + 2.0).abs() < 1e-12);
    // N = 80 lies just below the infinite-chain value 1 - 4 ln 2 = -1.77259.
    let e80 = heisenberg_chain(80)?;
    assert!((e80 - -1.773_103_81).abs() < 1e-8, "E(80)={e80}");
    assert!((ising_chain(80) - -1.273_321_36).abs() < 1e-8);
    Ok(())
}

#[test]
fn heisenberg_reference_rejects_odd_or_tiny_chains() {
    assert!(heisenberg_chain(2).is_err());
    assert!(heisenberg_chain(7)
        .unwrap_err()
        .to_string()
        .contains("even chain"));
}

#[test]
fn two_site_singlet_is_exact_for_zero_weights() -> Result<(), Box<dyn Error>> {
    // A length-2 periodic direction has one bond, so the ground state is the
    // singlet with E = -3 / 2 sites; zero weights give it exactly in the
    // Marshall-rotated basis.
    let result = exact_check(&zero_weight_model(2, 1, Hamiltonian::Heisenberg))?;
    assert_eq!(result.dimension, 2);
    assert!((result.ground_energy + 1.5).abs() < 1e-12);
    assert!((result.variational_energy + 1.5).abs() < 1e-12);
    assert!(result.variance.abs() < 1e-12);
    assert!((result.fidelity - 1.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn exact_check_rejects_large_sectors_before_allocating() {
    let model = zero_weight_model(4, 4, Hamiltonian::Heisenberg);
    let error = exact_check(&model).unwrap_err().to_string();
    assert!(error.contains("at most 4096 states"), "{error}");
}
