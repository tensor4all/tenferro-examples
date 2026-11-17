//! Tests for the figure helpers.

use super::*;
use plotters::style::BLUE;

#[test]
fn linear_bounds_pad_and_fall_back() {
    let (low, high) = linear_bounds([1.0, 3.0].into_iter());
    assert!(low < 1.0 && high > 3.0);
    assert_eq!(linear_bounds(std::iter::empty()), (0.0, 1.0));
    assert_eq!(
        linear_bounds([f64::NAN, f64::INFINITY].into_iter()),
        (0.0, 1.0)
    );
    assert_eq!(linear_bounds([2.0, 2.0].into_iter()), (1.0, 3.0));
}

#[test]
fn log_bounds_ignore_non_positive_values() {
    let (low, high) = log_bounds([1e-3, f64::NAN, -1.0, 0.0, 1e-1].into_iter());
    assert!(low < 1e-3 && high > 1e-1);
    assert_eq!(log_bounds([-1.0, 0.0].into_iter()), (1e-8, 1.0));
}

#[test]
fn loss_png_is_written() {
    let path = std::env::temp_dir().join("pinn-gauss-newton-loss-test.png");
    let path = path.to_str().expect("a UTF-8 temporary path");
    let samples = [(0.0, 1.0), (0.5, 1e-2), (1.0, 1e-4)];
    write_loss_png(path, &[(BLUE, &samples)]).expect("write the figure");
    let size = std::fs::metadata(path).expect("the figure exists").len();
    assert!(size > 0, "the figure is empty");
    let _ = std::fs::remove_file(path);
}
