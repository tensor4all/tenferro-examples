use super::*;

#[test]
fn step_decay_lr_has_three_stages() {
    let base = 1.0e-3;
    let total = 3000;
    // First half: full base rate.
    assert!((step_decay_lr(0, total, base) - base).abs() < 1e-18);
    assert!((step_decay_lr(total / 2 - 1, total, base) - base).abs() < 1e-18);
    // Second quarter: halved.
    assert!((step_decay_lr(total / 2, total, base) - base * 0.5).abs() < 1e-18);
    assert!((step_decay_lr(3 * total / 4 - 1, total, base) - base * 0.5).abs() < 1e-18);
    // Final quarter: quartered.
    assert!((step_decay_lr(3 * total / 4, total, base) - base * 0.25).abs() < 1e-18);
    assert!((step_decay_lr(total - 1, total, base) - base * 0.25).abs() < 1e-18);
}
