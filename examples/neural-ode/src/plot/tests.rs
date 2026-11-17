use super::*;

#[test]
fn linear_bounds_pad_finite_data() {
    let (low, high) = linear_bounds([1.0, 0.0, 2.0].into_iter());
    assert!(low < 0.0 && high > 2.0);
}

#[test]
fn linear_bounds_default_when_empty_and_pad_a_constant() {
    assert_eq!(linear_bounds(std::iter::empty::<f64>()), (-1.0, 1.0));
    assert_eq!(linear_bounds([3.0, 3.0].into_iter()), (2.0, 4.0));
}

#[test]
fn log_bounds_ignore_non_positive_values() {
    assert_eq!(log_bounds(&[]), (1e-8, 1.0));
    let (low, high) = log_bounds(&[1.0, 0.1, 0.0, -2.0, f64::INFINITY]);
    assert!((low - 0.1 / 1.5).abs() < 1e-15);
    assert!((high - 1.5).abs() < 1e-15);
}

#[test]
fn plot_renders_without_a_font_backend() {
    let dir = std::env::temp_dir().join(format!("neural-ode-plot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("neural_ode.png");

    let times: Vec<f64> = (0..5).map(|i| i as f64 * 0.1).collect();
    let truth: Vec<f64> = (0..10).map(|i| 0.1 * i as f64).collect();
    let pred: Vec<f64> = truth.iter().map(|value| value * 0.9).collect();
    let loss = vec![1.0, 0.5, 0.2];

    write_figure_png(path.to_str().unwrap(), &times, &truth, &pred, &loss).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}
