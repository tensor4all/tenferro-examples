//! Checkpointed discrete adjoint for one 10-step window.
//!
//! The reverse pass does not unroll the solver into one graph (tenferro 0.11.0
//! cannot stage a deep AD graph; see `IMPLEMENTATION-PLAN.md` §7). Instead one
//! RK4-step graph is traced once with the state and the cotangent as inputs:
//!
//! ```text
//! z_next = rk4_step(rhs, z_prev)
//! spot   = sum(lambda * z_next)
//! ```
//!
//! `value_and_grad(spot)` over the parameters and `z_prev` gives both the
//! parameter VJP of this step and `lambda_prev`, so a window is 10 forward runs
//! and 10 reverse runs of the same prepared program. The mean-absolute-error
//! cotangent is a host-side `sign`, which keeps the arithmetic in the graph.

#[cfg(test)]
mod tests;

use crate::data::{H, WINDOW_STEPS};
use crate::model;
use crate::run_prepared;
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use std::error::Error;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{
    CompiledGraph, DType, ExecutionEngine, GraphCompiler, Prepared, Result as TrResult, Runtime,
    TensorRead, TraceContext, TracedTensor,
};
use tenferro_tensor::Tensor;

/// Trace and compile the single-step training program.
///
/// Inputs are the four network parameters (in `Mlp` order), `z_prev` and
/// `lambda`; outputs are `z_next` followed by the parameter gradients and the
/// `z_prev` gradient.
pub(crate) fn build(trace: &TraceContext, net: &Mlp, batch: usize) -> TrResult<CompiledGraph> {
    let velocity = |z: &TracedTensor| model::rhs(net, z);
    let z_prev = trace.input_concrete_shape(DType::F64, &[batch, 2])?;
    let lambda = trace.input_concrete_shape(DType::F64, &[batch, 2])?;
    let z_next = crate::ode::rk4_step(&velocity, &z_prev, H)?;
    let spot = lambda.mul(&z_next)?.reduce_sum(None)?;

    let mut leaves = net.placeholders();
    leaves.push(&z_prev);
    let (_, grads) = spot.value_and_grad(&leaves)?;
    let grads: Vec<TracedTensor> = grads
        .into_iter()
        .map(|grad| grad.expect("every leaf reaches the step loss"))
        .collect();

    let mut outputs: Vec<&TracedTensor> = Vec::with_capacity(1 + grads.len());
    outputs.push(&z_next);
    outputs.extend(grads.iter());
    GraphCompiler::new().compile_many(&outputs)
}

/// Build the input list shared by the training runs.
pub(crate) fn step_reads<'a>(
    params: &'a [Tensor],
    z_prev: &'a Tensor,
    lambda: &'a Tensor,
) -> Vec<TensorRead<'a>> {
    let mut inputs: Vec<&Tensor> = params.iter().collect();
    inputs.push(z_prev);
    inputs.push(lambda);
    inputs.into_iter().map(TensorRead::from_tensor).collect()
}

/// Result of one window: the host loss, the accumulated parameter gradients in
/// `Mlp` order, and the checkpointed states `z_1..z_10`.
pub(crate) struct WindowResult {
    pub(crate) loss: f64,
    pub(crate) grads: Vec<Vec<f64>>,
    /// Checkpoints `z_1..z_10`, read by the adjoint tests.
    #[allow(dead_code)]
    pub(crate) states: Vec<Vec<f64>>,
}

/// Run one window forward to collect `z_1..z_10`, then reverse through the same
/// prepared program.
///
/// `targets[k]` is the target for `z_{k+1}`, flattened column-major as
/// `[B, 2]`. The caller owns the prepared program and the runtime.
pub(crate) fn window_loss_and_grad(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    params: &[Tensor],
    z0: &Tensor,
    targets: &[Vec<f64>],
) -> Result<WindowResult, Box<dyn Error>> {
    let batch = z0.shape()[0];
    let width = batch * 2;
    let scale = 1.0 / (width * WINDOW_STEPS) as f64;
    let zeros = Tensor::from_vec_col_major(vec![batch, 2], vec![0.0; width])?;

    // Forward: the step program also emits gradients, which are unused here.
    let mut z = Tensor::from_vec_col_major(vec![batch, 2], z0.as_slice::<f64>()?.to_vec())?;
    let mut states = Vec::with_capacity(WINDOW_STEPS);
    for _ in 0..WINDOW_STEPS {
        let reads = step_reads(params, &z, &zeros);
        let mut outputs = run_prepared(runtime, engine, prepared, &reads)?;
        let next = outputs.swap_remove(0);
        states.push(next.as_slice::<f64>()?.to_vec());
        z = next;
    }

    // Host loss, and the direct cotangent of every checkpointed state.
    let mut loss = 0.0;
    for (state, target) in states.iter().zip(targets) {
        for (&predicted, &reference) in state.iter().zip(target) {
            loss += (predicted - reference).abs();
        }
    }
    loss *= scale;

    // Reverse: one VJP per step, with the carried cotangent as an input.
    let mut grad_acc: Vec<Vec<f64>> = params
        .iter()
        .map(|p| vec![0.0; p.shape().iter().product()])
        .collect();
    let mut lambda = vec![0.0; width];
    for k in (0..WINDOW_STEPS).rev() {
        for ((carried, &predicted), &reference) in
            lambda.iter_mut().zip(&states[k]).zip(&targets[k])
        {
            // `f64::signum` returns 1.0 at +0.0; the absolute value has its
            // subgradient zero there, which is what keeps a fitted state inert.
            let residual = predicted - reference;
            *carried += if residual > 0.0 {
                scale
            } else if residual < 0.0 {
                -scale
            } else {
                0.0
            };
        }
        let lambda_tensor = Tensor::from_vec_col_major(vec![batch, 2], lambda)?;
        let z_prev = if k > 0 {
            // z_k: the state the step under review started from.
            Tensor::from_vec_col_major(vec![batch, 2], states[k - 1].clone())?
        } else {
            Tensor::from_vec_col_major(vec![batch, 2], z0.as_slice::<f64>()?.to_vec())?
        };
        let reads = step_reads(params, &z_prev, &lambda_tensor);
        let mut outputs = run_prepared(runtime, engine, prepared, &reads)?;
        let state_grad = outputs.pop().expect("the step program emits z_prev grad");
        assert_eq!(outputs.len(), 1 + grad_acc.len(), "step program outputs");
        for (accumulated, grad) in grad_acc.iter_mut().zip(&outputs[1..]) {
            for (slot, &value) in accumulated.iter_mut().zip(grad.as_slice::<f64>()?) {
                *slot += value;
            }
        }
        lambda = state_grad.as_slice::<f64>()?.to_vec();
    }

    Ok(WindowResult {
        loss,
        grads: grad_acc,
        states,
    })
}

/// Prepare the training program once with the first window's inputs.
pub(crate) fn prepare(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &CompiledGraph,
    params: &[Tensor],
    z0: &Tensor,
) -> Result<Prepared, Box<dyn Error>> {
    let batch = z0.shape()[0];
    let zeros = Tensor::from_vec_col_major(vec![batch, 2], vec![0.0; batch * 2])?;
    let reads = step_reads(params, z0, &zeros);
    Ok(runtime.prepare(program, &reads, &[engine.target()])?)
}
