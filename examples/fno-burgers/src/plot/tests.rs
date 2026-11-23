use super::*;

#[test]
fn writes_a_prediction_png_and_rejects_mismatched_grids() {
    let path = std::env::temp_dir().join(format!("fno-burgers-{}.png", std::process::id()));
    let n = 16;
    let initial: Vec<f64> = (0..2 * n)
        .map(|j| (std::f64::consts::TAU * j as f64 / n as f64).sin())
        .collect();
    let reference: Vec<f64> = initial.iter().map(|v| 0.01 * v).collect();
    write_prediction_png(path.to_str().unwrap(), n, &initial, &reference, &reference).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    std::fs::remove_file(path).unwrap();
    assert!(write_prediction_png("unused.png", n, &initial, &[], &reference).is_err());
    assert!(write_prediction_png("unused.png", 0, &[], &[], &[]).is_err());
}
