use super::*;

#[test]
fn sampler_visits_all_uniform_states_and_preserves_heisenberg_sector() {
    let mut model = Model::new(4, 1, 1, Hamiltonian::Ising, 12);
    model.weights.fill(0.0);
    let mut chain = Chain::new(&model, 55);
    let mut fields = model.fields(&chain.spins);
    let mut log = log_fields(&fields);
    let mut counts = [0; 16];
    for _ in 0..1000 {
        chain.sweep(&model, &mut fields, &mut log);
        let mask = chain
            .spins
            .iter()
            .enumerate()
            .fold(0, |mask, (i, s)| mask | if *s > 0.0 { 1 << i } else { 0 });
        counts[mask] += 1;
    }
    assert!(
        counts.iter().all(|c| *c > 0),
        "uniform chain missed a parity sector: {counts:?}"
    );
    model.hamiltonian = Hamiltonian::Heisenberg;
    let mut chain = Chain::new(&model, 87);
    for _ in 0..100 {
        let mut fields = model.fields(&chain.spins);
        let mut log = log_fields(&fields);
        chain.sweep(&model, &mut fields, &mut log);
        assert_eq!(chain.spins.iter().sum::<f64>(), 0.0);
    }
}

#[test]
fn samples_are_concatenated_chain_by_chain() {
    let model = Model::new(6, 1, 1, Hamiltonian::Heisenberg, 4);
    let mut chains: Vec<Chain> = (0..3).map(|i| Chain::new(&model, 10 + i)).collect();
    let mut reference: Vec<Chain> = (0..3).map(|i| Chain::new(&model, 10 + i)).collect();
    let batch = sample(&mut chains, &model, 5, 2, true);
    assert_eq!(batch.energies.len(), 15);
    assert_eq!(batch.gradients.len(), 15 * model.weights.len());
    for (c, chain) in reference.iter_mut().enumerate() {
        let single = chain.sample(&model, 5, 2, true);
        assert_eq!(single.energies, batch.energies[c * 5..(c + 1) * 5]);
    }
}

#[test]
fn blocking_does_not_mix_distinct_chains() {
    let energies: Vec<f64> = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        .iter()
        .flat_map(|&e| std::iter::repeat_n(e, 512))
        .collect();
    let error = block_standard_error(&energies, 512);
    assert!((error - 1.0 / 7.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "do not form at least two blocks")]
fn blocking_rejects_a_single_block() {
    block_standard_error(&[1.0; 64], 64);
}
