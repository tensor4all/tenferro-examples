use super::*;

#[test]
fn axis_bounds_pads_finite_data() {
    let (lo, hi) = axis_bounds([1.0, 0.0, 2.0].into_iter());
    assert!(lo < 0.0 && hi > 2.0);
}

#[test]
fn axis_bounds_defaults_when_empty_or_constant() {
    assert_eq!(axis_bounds(std::iter::empty::<f64>()), (-1.0, 1.0));
    assert_eq!(axis_bounds([3.0, 3.0].into_iter()), (2.0, 4.0));
}

#[test]
fn plot_renders_without_a_font_backend() {
    let dir = std::env::temp_dir().join(format!("deeponet-plot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("deeponet.png");

    let sensors: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect();
    let y: Vec<f64> = (0..7).map(|j| j as f64 / 6.0).collect();
    let n_test = 2;
    let u = vec![0.1; n_test * sensors.len()];
    let truth: Vec<f64> = (0..n_test * y.len()).map(|i| i as f64 * 0.01).collect();
    let pred = truth.clone();

    write_test_png(
        path.to_str().unwrap(),
        &sensors,
        &y,
        &u,
        &truth,
        &pred,
        n_test,
    )
    .unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}
