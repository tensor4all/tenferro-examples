use super::*;

#[test]
fn disconnected_parameter_returns_an_indexed_internal_error() {
    let trace = TraceContext::new();
    let connected = trace.from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let error = parameter_gradients(vec![Some(connected), None]).unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Internal);
    assert!(error.to_string().contains("parameter 1"), "{error}");
}

#[test]
fn relative_l2_reports_a_known_residual_and_rejects_invalid_samples() {
    assert_eq!(relative_l2(&[1.5, 2.0], &[3.0, 4.0]).unwrap(), 0.5);
    assert_eq!(relative_l2(&[3.0, 4.0], &[3.0, 4.0]).unwrap(), 0.0);
    for (prediction, reference) in [
        (vec![1.0], vec![1.0, 2.0]),
        (vec![], vec![]),
        (vec![0.0], vec![0.0]),
        (vec![f64::NAN], vec![1.0]),
    ] {
        assert!(relative_l2(&prediction, &reference).is_err());
    }
}
