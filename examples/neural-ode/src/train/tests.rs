use super::*;
use crate::data::{Dataset, WINDOW_STEPS};
use crate::model::MLP_SIZES;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use tenferro_runtime::{Prepared, Runtime, TraceContext, TracedTensor};

/// Batch used by the checks: small enough to keep the test fast, large enough
/// for two independent windows.
const TEST_BATCH: usize = 2;

struct Rig {
    runtime: Runtime,
    engine: tenferro_cpu::CpuEngine,
    prepared: Prepared,
    params: Vec<Tensor>,
    z0: Tensor,
    targets: Vec<Vec<f64>>,
}

fn rig() -> Result<Rig, Box<dyn std::error::Error>> {
    let data = Dataset::generate();
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &MLP_SIZES, TracedTensor::tanh)?;
    let params = net.initial_values(&mut StdRng::seed_from_u64(7))?;
    let program = build(&trace, &net, TEST_BATCH)?;

    let starts = [0, 3];
    let (z0, target_tensors) = data.minibatch(&starts)?;
    let targets: Vec<Vec<f64>> = target_tensors
        .iter()
        .map(|target| Ok(target.as_slice::<f64>()?.to_vec()))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    let mut engine = tenferro_cpu::CpuEngine::serial();
    let runtime = crate::cpu_runtime(&engine)?;
    let prepared = prepare(&runtime, &mut engine, &program, &params, &z0)?;
    Ok(Rig {
        runtime,
        engine,
        prepared,
        params,
        z0,
        targets,
    })
}

#[test]
fn adjoint_matches_a_finite_difference_of_the_window_loss() -> Result<(), Box<dyn std::error::Error>>
{
    let mut rig = rig()?;
    let base = window_loss_and_grad(
        &rig.runtime,
        &mut rig.engine,
        &rig.prepared,
        &rig.params,
        &rig.z0,
        &rig.targets,
    )?;

    // A deterministic direction in parameter space, one entry per parameter.
    let direction: Vec<Vec<f64>> = rig
        .params
        .iter()
        .enumerate()
        .map(|(layer, param)| {
            (0..param.shape().iter().product())
                .map(|index| ((layer * 31 + index * 17) as f64 * 0.37).sin())
                .collect()
        })
        .collect();
    let perturbed = |sign: f64| -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
        rig.params
            .iter()
            .zip(&direction)
            .map(|(param, step)| {
                let values: Vec<f64> = param
                    .as_slice::<f64>()?
                    .iter()
                    .zip(step)
                    .map(|(&value, &delta)| value + sign * DELTA * delta)
                    .collect();
                Ok(Tensor::from_vec_col_major(param.shape().to_vec(), values)?)
            })
            .collect()
    };

    let plus = window_loss_and_grad(
        &rig.runtime,
        &mut rig.engine,
        &rig.prepared,
        &perturbed(1.0)?,
        &rig.z0,
        &rig.targets,
    )?;
    let minus = window_loss_and_grad(
        &rig.runtime,
        &mut rig.engine,
        &rig.prepared,
        &perturbed(-1.0)?,
        &rig.z0,
        &rig.targets,
    )?;

    let directional: f64 = base
        .grads
        .iter()
        .zip(&direction)
        .flat_map(|(grad, step)| grad.iter().zip(step).map(|(&g, &d)| g * d))
        .sum();
    // The window loss is nonlinear in the parameters (tanh and RK4), so this
    // is a numerical directional-derivative check with a central difference
    // accurate to O(DELTA^2), not an exact identity.
    let finite_difference = (plus.loss - minus.loss) / (2.0 * DELTA);
    assert!(
        (finite_difference - directional).abs() <= 1e-9 * (1.0 + directional.abs()),
        "finite difference {finite_difference:e} vs adjoint {directional:e}"
    );
    assert_eq!(base.states.len(), WINDOW_STEPS);
    Ok(())
}

const DELTA: f64 = 1e-5;

#[test]
fn zero_residual_has_a_zero_cotangent_and_gradient() -> Result<(), Box<dyn std::error::Error>> {
    let mut rig = rig()?;
    let forward = window_loss_and_grad(
        &rig.runtime,
        &mut rig.engine,
        &rig.prepared,
        &rig.params,
        &rig.z0,
        &rig.targets,
    )?;

    // Feeding the predicted states back as targets makes every residual zero,
    // so every direct cotangent is `sign(0) = 0` and the whole gradient dies.
    let exact = window_loss_and_grad(
        &rig.runtime,
        &mut rig.engine,
        &rig.prepared,
        &rig.params,
        &rig.z0,
        &forward.states,
    )?;
    assert_eq!(exact.loss, 0.0);
    assert!(
        exact.grads.iter().flatten().all(|value| *value == 0.0),
        "a zero residual must give an exactly zero gradient"
    );
    Ok(())
}
