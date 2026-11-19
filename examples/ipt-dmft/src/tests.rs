use super::*;

#[test]
fn coexistence_spans_the_u_where_the_sweeps_differ() {
    let us = [2.0, 2.5, 3.0, 3.5];
    // The insulating Z can exceed any fixed cut at high temperature.
    let up = [0.30, 0.15, 0.10, 0.02];
    let down = [0.30, 0.02, 0.0205, 0.02];
    assert_eq!(coexistence(&us, &up, &down), Some((2.5, 3.0)));
    assert_eq!(coexistence(&us, &up, &up), None);
    assert_eq!(coexistence(&us[..1], &[0.3], &[0.1]), Some((2.0, 2.0)));
}
