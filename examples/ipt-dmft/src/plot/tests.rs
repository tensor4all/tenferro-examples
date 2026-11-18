use super::*;

#[test]
fn plot_renders_without_a_font_backend() {
    let dir = std::env::temp_dir().join(format!("ipt-dmft-plot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("z.png");
    let us = [2.0, 2.5, 3.0];
    write_z_png(
        path.to_str().unwrap(),
        &us,
        &[0.5, 0.3, 0.0],
        &[0.5, 0.0, 0.0],
    )
    .unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}
