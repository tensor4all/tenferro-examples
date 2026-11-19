//! Four-layer Fourier neural operator with complex low-mode channel maps.
//!
//! The real lifting, skip, and projection parameters precede all complex
//! spectral weights, both in the trace and in `Params`, so each dtype can use
//! its own Adam state. Spatial samples are the leading, contiguous axis.

use rand::Rng;
use sciml_kit::func::Params;
use sciml_kit::nn::Mlp;
use tenferro_fft::{FftOptions, TracedFftOps};
use tenferro_runtime::{
    DType, DotGeneralConfig, Error, ErrorPhase, Result, TraceContext, TracedTensor,
};
use tenferro_tensor::{Complex, PadConfig, Tensor, ValidationError};

pub(crate) struct Fno {
    lift: Mlp,
    skips: Vec<Mlp>,
    projection: Mlp,
    weights: Vec<TracedTensor>,
    width: usize,
    modes: usize,
}

impl Fno {
    /// Create real placeholders first, then four `[modes, in, out]` C64 weights.
    pub(crate) fn new(trace: &TraceContext, width: usize, modes: usize) -> Result<Self> {
        if width == 0 || modes == 0 {
            return Err(Error::validation(
                "Fno::new",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "width/modes",
                    message: "width and mode count must be positive".into(),
                },
            ));
        }
        let projection_width = width.checked_mul(2).ok_or_else(|| {
            Error::validation(
                "Fno::new",
                ErrorPhase::GraphBuild,
                ValidationError::IntegerOverflow,
            )
        })?;
        let lift = Mlp::new(trace, &[1, width], TracedTensor::gelu)?;
        let skips = (0..4)
            .map(|_| Mlp::new(trace, &[width, width], TracedTensor::gelu))
            .collect::<Result<Vec<_>>>()?;
        let projection = Mlp::new(trace, &[width, projection_width, 1], TracedTensor::gelu)?;
        let weights = (0..4)
            .map(|_| trace.input_concrete_shape(DType::C64, &[modes, width, width]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lift,
            skips,
            projection,
            weights,
            width,
            modes,
        })
    }

    pub(crate) fn real_param_count(&self) -> usize {
        2 + 2 * self.skips.len() + 4
    }

    /// Map real periodic fields `[n, batch, 1]` to the same spatial resolution.
    pub(crate) fn forward(&self, x: &TracedTensor, n: usize, batch: usize) -> Result<TracedTensor> {
        let mut v = self
            .lift
            .forward(&x.reshape([n * batch, 1])?)?
            .reshape([n, batch, self.width])?;
        for (skip, weight) in self.skips.iter().zip(&self.weights) {
            let local = skip
                .forward(&v.reshape([n * batch, self.width])?)?
                .reshape([n, batch, self.width])?;
            v = spectral(&v, weight, n, self.modes)?.add(&local)?.gelu()?;
        }
        self.projection
            .forward(&v.reshape([n * batch, self.width])?)?
            .reshape([n, batch, 1])
    }
}

impl Params for Fno {
    fn placeholders(&self) -> Vec<&TracedTensor> {
        let mut parameters = self.lift.placeholders();
        for skip in &self.skips {
            parameters.extend(skip.placeholders());
        }
        parameters.extend(self.projection.placeholders());
        parameters.extend(&self.weights);
        parameters
    }

    fn initial_values(&self, rng: &mut impl Rng) -> Result<Vec<Tensor>> {
        let mut values = self.lift.initial_values(rng)?;
        for skip in &self.skips {
            values.extend(skip.initial_values(rng)?);
        }
        values.extend(self.projection.initial_values(rng)?);
        // Complex Xavier: independent real/imaginary uniforms give E|w|² = 1/width.
        let scale = (1.5 / self.width as f64).sqrt();
        for _ in &self.weights {
            values.push(Tensor::from_vec_col_major(
                vec![self.modes, self.width, self.width],
                (0..self.modes * self.width * self.width)
                    .map(|_| {
                        Complex::new(rng.gen_range(-scale..scale), rng.gen_range(-scale..scale))
                    })
                    .collect(),
            )?);
        }
        Ok(values)
    }
}

fn spectral(
    x: &TracedTensor,
    weight: &TracedTensor,
    n: usize,
    modes: usize,
) -> Result<TracedTensor> {
    let half = n / 2 + 1;
    if n == 0 || modes == 0 || modes > half {
        return Err(Error::validation(
            "spectral",
            ErrorPhase::GraphBuild,
            ValidationError::InvalidArgument {
                argument: "n/modes",
                message: "a positive mode count must fit the real FFT's one-sided spectrum".into(),
            },
        ));
    }
    let low = x
        .rfft_with(&FftOptions::default().axis(0))?
        .slice_axis(0, 0..modes)?;
    let mixed = low.dot_general(
        weight,
        DotGeneralConfig {
            lhs_contracting_dims: [2].as_slice().into(),
            rhs_contracting_dims: [1].as_slice().into(),
            lhs_batch_dims: [0].as_slice().into(),
            rhs_batch_dims: [0].as_slice().into(),
        },
    )?;
    // dot_general places free axes before batch axes: [batch, out, mode].
    let padded = mixed.transpose(&[2, 0, 1])?.pad(PadConfig {
        edge_padding_low: vec![0; 3],
        edge_padding_high: vec![(half - modes) as i64, 0, 0],
        interior_padding: vec![0; 3],
    })?;
    // The explicit half-spectrum padding and n preserve even and odd grid sizes.
    padded.irfft_with(&FftOptions::default().axis(0).n(Some(n)))
}

#[cfg(test)]
mod tests;
