use super::*;
use crate::model::Hamiltonian;

#[test]
fn centering_moves_the_peak_to_the_middle_and_makes_it_positive() {
    let mut weights = vec![0.0; 16];
    weights[1] = -1.0; // (x, y) = (0, 1)
    weights[2] = -0.5; // (x, y) = (0, 2)
    let image = center_filter(&weights, 4, 4);
    assert_eq!(image[2 * 4 + 2], 1.0);
    assert_eq!(image[2 * 4 + 3], 0.5);
    assert_eq!(image.iter().filter(|&&v| v != 0.0).count(), 2);

    let chain = center_filter(&[0.1, 0.0, 0.0, 0.0, 0.3], 5, 1);
    assert_eq!(chain, [0.0, 0.0, 0.3, 0.1, 0.0]);
}

#[test]
fn diverging_colors_saturate_at_the_limit() {
    assert_eq!(diverging_color(0.0, 0.15), RGBColor(255, 255, 255));
    assert_eq!(diverging_color(0.15, 0.15), RGBColor(178, 24, 43));
    assert_eq!(diverging_color(9.0, 0.15), RGBColor(178, 24, 43));
    assert_eq!(diverging_color(-0.15, 0.15), RGBColor(33, 102, 172));
    assert_eq!(diverging_color(-9.0, 0.15), RGBColor(33, 102, 172));
}

#[test]
fn energy_axis_bounds_cover_the_reference_and_skip_nonfinite_values() {
    assert_eq!(energy_axis_bounds(&[], None), (-1.0, 1.0));
    let (lo, hi) = energy_axis_bounds(&[f64::NAN, -1.0, -1.2], Some(-1.3));
    assert!(lo < -1.3 && hi > -1.0 && hi < -0.9);
}

/// All plots render without a font backend; drawing any text would panic.
#[test]
fn plots_render_without_a_font_backend() {
    let dir = std::env::temp_dir().join(format!("nqs-rbm-plot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = |name: &str| dir.join(name).to_str().unwrap().to_owned();
    let chain = Model::new(8, 1, 3, Hamiltonian::Ising, 1);
    let square = Model::new(4, 4, 5, Hamiltonian::Heisenberg, 2);
    write_filters_png(&path("chain.png"), &chain).unwrap();
    write_filters_png(&path("square.png"), &square).unwrap();
    let energies = [-1.0, -1.2, -1.27, -1.273];
    write_energy_png(&path("reference.png"), &energies, Some(-1.2733)).unwrap();
    write_energy_png(&path("energy.png"), &energies, None).unwrap();
    for name in ["chain.png", "square.png", "reference.png", "energy.png"] {
        assert!(std::fs::metadata(dir.join(name)).unwrap().len() > 0);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
