use super::*;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{GraphCompiler, TraceContext};

#[test]
fn rk4_matches_the_exponential_for_z_prime_equals_minus_z(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // After ten steps of h = 0.1, RK4 on z' = -z is within ~3e-7 of e^{-1}.
    let trace = TraceContext::new();
    let z0 = trace.from_vec_col_major(vec![], vec![1.0_f64])?;
    let rhs = |z: &TracedTensor| z.scale_real(-1.0);
    let states = rollout(&rhs, &z0, 0.1, 10)?;
    assert_eq!(states.len(), 10, "one state per step");

    let program = GraphCompiler::new().compile_many(&[&states[9]])?;
    let mut engine = CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let out = crate::run_single(&runtime, &mut engine, &program, &[])?;
    let value = out.as_slice::<f64>()?[0];
    assert!((value - (-1.0_f64).exp()).abs() < 1e-6, "got {value}");
    Ok(())
}
