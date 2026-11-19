//! Learn u(x,0) -> u(x,1) for periodic viscous Burgers with a Fourier neural
//! operator (Li et al., ICLR 2021, https://arxiv.org/abs/2010.08895).

mod config;
mod data;
mod model;
mod plot;

use rand::{rngs::StdRng, Rng, SeedableRng};
use sciml_kit::{
    func::Params,
    optim::{step_decay_lr, Adam},
};
use std::error::Error;
use std::time::Instant;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor,
};
use tenferro_tensor::{Tensor, TensorRead};

const TRAIN_N: usize = 128;
const MODES: usize = 16;

fn relative_l2(prediction: &[f64], reference: &[f64]) -> Result<f64, Box<dyn Error>> {
    if prediction.len() != reference.len() || prediction.is_empty() {
        return Err("relative L2 needs matching nonempty prediction and reference arrays".into());
    }
    let error: f64 = prediction
        .iter()
        .zip(reference)
        .map(|(p, r)| (p - r).powi(2))
        .sum();
    let norm: f64 = reference.iter().map(|r| r * r).sum();
    let relative = (error / norm).sqrt();
    if !relative.is_finite() || norm <= 0.0 {
        return Err("relative L2 needs finite samples and a nonzero reference norm".into());
    }
    Ok(relative)
}

fn parameter_gradients(
    gradients: Vec<Option<TracedTensor>>,
) -> tenferro_tensor::Result<Vec<TracedTensor>> {
    gradients
        .into_iter()
        .enumerate()
        .map(|(index, gradient)| {
            gradient.ok_or_else(|| {
                tenferro_tensor::Error::Internal(format!(
                    "FNO model parameter {index} does not reach the loss"
                ))
            })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(config) = config::Config::parse(std::env::args().skip(1))? else {
        println!(
            "fno_burgers [--epochs 1000] [--seed 42] [--width 16] \
            [--train-samples 256] [--test-samples 64] [--batch 16] [--plot PATH]\n\
            Train at 128 points; evaluate held-out functions at 128, 256 and 512.\n\
            Each epoch is one independently sampled minibatch Adam step."
        );
        return Ok(());
    };
    let startup = Instant::now();
    println!(
        "seed {} (train {}, test {}, initialization {}, minibatches {})",
        config.seed,
        config.seed,
        config.seed.wrapping_add(1),
        config.seed.wrapping_add(2),
        config.seed.wrapping_add(3)
    );
    println!("reference grid {}, train grid {TRAIN_N}, modes {MODES}, width {}, four layers, train {}, test {}, batch {}, Adam steps {}",
        data::REFERENCE_N, config.width,config.train_samples,config.test_samples,config.batch,config.epochs);
    let train = data::Dataset::generate(config.train_samples, config.seed)?;
    let test = data::Dataset::generate(config.test_samples, config.seed.wrapping_add(1))?;
    let (input_scale, target_scale) = train.rms();
    if !input_scale.is_finite()
        || !target_scale.is_finite()
        || input_scale <= 0.0
        || target_scale <= 0.0
    {
        return Err("training data must have finite positive input and target RMS".into());
    }
    println!(
        "data generation: {:.3} s; train RMS input={input_scale:.8e}, target={target_scale:.8e}",
        startup.elapsed().as_secs_f64()
    );

    let trace = TraceContext::new();
    let net = model::Fno::new(&trace, config.width, MODES)?;
    let mut params = net.initial_values(&mut StdRng::seed_from_u64(config.seed.wrapping_add(2)))?;
    let real_count = net.real_param_count();
    // Placeholders stay parameters-first in the same order as Params.
    let x = trace.input_concrete_shape(DType::F64, &[TRAIN_N, config.batch, 1])?;
    let target = trace.input_concrete_shape(DType::F64, &[TRAIN_N, config.batch, 1])?;
    let prediction = net.forward(&x.scale_real(1.0 / input_scale)?, TRAIN_N, config.batch)?;
    let residual = prediction.sub(&target.scale_real(1.0 / target_scale)?)?;
    let loss = residual
        .mul(&residual)?
        .reduce_sum(None)?
        .scale_real(1.0 / (TRAIN_N * config.batch) as f64)?;
    // The default value_and_grad has no FFT extension rules in 0.12. A seed-1
    // vjp_many on the configured context is the same single reverse pass.
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_fft::semantic_ad_rules()?)?
        .build()?;
    let seed = trace.from_vec_col_major(vec![], vec![1.0_f64])?;
    let gradients = parameter_gradients(ad.vjp_many(&loss, &net.placeholders(), &seed)?)?;
    let mut outputs = vec![&loss];
    outputs.extend(gradients.iter());
    let compiled = Instant::now();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&outputs)?;
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    let runtime = builder.build()?;
    let prepared = {
        let (x, target) = train.batch(&vec![0; config.batch], TRAIN_N)?;
        let mut inputs: Vec<&Tensor> = params.iter().collect();
        inputs.extend([&x, &target]);
        let reads: Vec<_> = inputs.into_iter().map(TensorRead::from_tensor).collect();
        runtime.prepare(&program, &reads, &[engine.target()])?
    };
    println!(
        "training compile/prepare: {:.3} s; {} real and {} complex parameter tensors",
        compiled.elapsed().as_secs_f64(),
        real_count,
        params.len() - real_count
    );
    let mut real_adam = Adam::new(1e-3);
    let mut complex_adam = Adam::new(1e-3);
    let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(3));
    let trained = Instant::now();
    for epoch in 0..config.epochs {
        let indices: Vec<usize> = (0..config.batch)
            .map(|_| rng.gen_range(0..train.len()))
            .collect();
        let (x, target) = train.batch(&indices, TRAIN_N)?;
        let mut inputs: Vec<&Tensor> = params.iter().collect();
        inputs.extend([&x, &target]);
        let reads: Vec<_> = inputs.into_iter().map(TensorRead::from_tensor).collect();
        let values = runtime.run(&mut [&mut engine], &prepared, &reads)?;
        let values: Vec<Tensor> = values
            .into_iter()
            .map(|v| v.into_tensor(&mut engine))
            .collect::<Result<_, _>>()?;
        let mse = values[0].as_slice::<f64>()?[0];
        if !mse.is_finite() {
            return Err(format!("non-finite training loss at step {}", epoch + 1).into());
        }
        let lr = step_decay_lr(epoch, config.epochs, 1e-3);
        real_adam.set_lr(lr);
        complex_adam.set_lr(lr);
        real_adam.step(&mut params[..real_count], &values[1..1 + real_count]);
        complex_adam.step(&mut params[real_count..], &values[1 + real_count..]);
        if epoch == 0 || (epoch + 1) % 100 == 0 || epoch + 1 == config.epochs {
            println!("step {}: normalized minibatch MSE={mse:.8e}", epoch + 1);
        }
    }
    println!("training: {:.3} s", trained.elapsed().as_secs_f64());
    // Each evaluation shape gets one forward-only program and one Prepared.
    // The exact same final weights and held-out trajectories serve every grid.
    for (label, dataset, n) in [
        ("train", &train, 128),
        ("test", &test, 128),
        ("test", &test, 256),
        ("test", &test, 512),
    ] {
        let batch = dataset.len();
        let (input, reference) = dataset.batch(&(0..batch).collect::<Vec<_>>(), n)?;
        let x = trace.input_concrete_shape(DType::F64, &[n, batch, 1])?;
        let prediction = net
            .forward(&x.scale_real(1.0 / input_scale)?, n, batch)?
            .scale_real(target_scale)?;
        let program = compiler.compile_many(&[&prediction])?;
        let mut inputs: Vec<&Tensor> = params.iter().collect();
        inputs.push(&input);
        let reads: Vec<_> = inputs.into_iter().map(TensorRead::from_tensor).collect();
        let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;
        let evaluated = Instant::now();
        let mut values = runtime.run(&mut [&mut engine], &prepared, &reads)?;
        let prediction = values.remove(0).into_tensor(&mut engine)?;
        let prediction = prediction.as_slice::<f64>()?;
        let relative = relative_l2(prediction, reference.as_slice::<f64>()?)?;
        println!(
            "{label} n={n}, samples={batch}: relative L2={relative:.8e}; forward {:.3} s",
            evaluated.elapsed().as_secs_f64()
        );
        if label == "test" && n == 512 {
            if let Some(path) = &config.plot {
                plot::write_prediction_png(
                    path,
                    n,
                    input.as_slice::<f64>()?,
                    reference.as_slice::<f64>()?,
                    prediction,
                )?;
                println!("saved held-out prediction/reference figure to {path}");
            }
        }
    }
    println!("total elapsed: {:.3} s", startup.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests;
