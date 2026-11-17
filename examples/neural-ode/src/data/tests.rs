use super::*;

#[test]
fn velocity_is_the_matrix_times_the_elementwise_cube() {
    // Column 1 and column 2 of A; the cube of [1, 0] and [0, 1] is itself.
    assert!((velocity([1.0, 0.0])[0] + 0.1).abs() < 1e-15);
    assert!((velocity([1.0, 0.0])[1] + 2.0).abs() < 1e-15);
    assert!((velocity([0.0, 1.0])[0] - 2.0).abs() < 1e-15);
    assert!((velocity([0.0, 1.0])[1] + 0.1).abs() < 1e-15);

    // z = [2, -1] gives z³ = [8, -1] and A z³ = [-2.8, -15.9].
    let v = velocity([2.0, -1.0]);
    assert!((v[0] + 2.8).abs() < 1e-12, "got {v:?}");
    assert!((v[1] + 15.9).abs() < 1e-12, "got {v:?}");
}

#[test]
fn reference_trajectory_is_converged() {
    // RK4 converges at fourth order; the difference between 25 and 50 fine
    // steps per sample interval is already far below the training error, and
    // the shipped trajectory uses 100.
    let coarse = Dataset::generate_with_substeps(25);
    let fine = Dataset::generate_with_substeps(50);
    for (index, (&a, &b)) in coarse.states().iter().zip(fine.states()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8,
            "sample buffer entry {index}: {a} vs {b}"
        );
    }
}

#[test]
fn minibatch_places_the_window_at_the_documented_indices() {
    let data = Dataset::generate();
    let starts = [0usize, 5];
    let (initial, targets) = data.minibatch(&starts).unwrap();

    assert_eq!(initial.shape(), &[2, 2]);
    assert_eq!(targets.len(), WINDOW_STEPS);
    let initial = initial.as_slice::<f64>().unwrap();

    let batch = starts.len();
    for (column, &start) in starts.iter().enumerate() {
        for component in 0..2 {
            assert_eq!(
                initial[column + batch * component],
                data.state(start)[component],
                "initial state {column}/{component}"
            );
            for (step, target) in targets.iter().enumerate() {
                assert_eq!(target.shape(), &[2, 2]);
                let target = target.as_slice::<f64>().unwrap();
                assert_eq!(
                    target[column + batch * component],
                    data.state(start + step + 1)[component],
                    "target {column}/{component}/{step}"
                );
            }
        }
    }
}

#[test]
fn the_last_window_ends_at_the_last_sample() {
    let data = Dataset::generate();
    let last_start = N_SAMPLES - WINDOW_STEPS - 1;
    let (_, targets) = data.minibatch(&[last_start]).unwrap();
    let target = targets.last().unwrap().as_slice::<f64>().unwrap();
    for (component, &written) in target.iter().enumerate() {
        let expected = data.state(N_SAMPLES - 1)[component];
        assert_eq!(written, expected);
    }
}

#[test]
#[should_panic(expected = "leaves the reference trajectory")]
fn a_window_past_the_end_is_rejected() {
    let data = Dataset::generate();
    let _ = data.minibatch(&[N_SAMPLES - WINDOW_STEPS]);
}
