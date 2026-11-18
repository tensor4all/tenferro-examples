use super::*;

#[test]
fn writes_a_png() {
    let dir = std::env::temp_dir().join("nonlinear_pinn_plot_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("snapshots.png");
    let x: Vec<f64> = (0..64).map(|j| -1.0 + 2.0 * j as f64 / 64.0).collect();
    let curve: Vec<f64> = x
        .iter()
        .map(|x| -(std::f64::consts::PI * x).sin())
        .collect();
    let snapshots = vec![(curve.clone(), curve); 4];
    write_snapshots_png(path.to_str().unwrap(), &x, &snapshots).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
}
