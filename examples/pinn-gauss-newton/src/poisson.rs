//! The Poisson problem: points, forcing term, residual vector and its parameter
//! Jacobian.
//!
//! The residual is one vector over all points, collocation points first and
//! boundary points second:
//!
//! ```text
//! r = [ (-Laplace u - f) / sqrt(n_col) ;  u / sqrt(n_bc) ]
//! ```
//!
//! The `1/sqrt(n)` factors give each group one mean square, as the KdV example
//! does, so the two boundary rows are not swamped by the 2048 collocation rows.

use crate::network::Model;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{Result, TraceContext, TracedTensor};
use tenferro_tensor::{SliceConfig, Tensor};

#[cfg(test)]
mod tests;

/// The point set of one run: one column-major `[dim, n_col + n_bc]` tensor.
pub(crate) struct Points {
    /// Collocation points followed by boundary points.
    pub(crate) tensor: Tensor,
    /// Number of interior (collocation) columns.
    pub(crate) n_col: usize,
    /// Number of boundary columns.
    pub(crate) n_bc: usize,
}

impl Points {
    pub(crate) fn len(&self) -> usize {
        self.n_col + self.n_bc
    }
}

/// Sample the fixed point set of a run.
///
/// Collocation points are uniform on `(0, 1)^dim`. The boundary is `x = 0` and
/// `x = 1` in one dimension and the four edges in two, with `n_bc` midpoints per
/// edge, so no corner is counted twice. `n_bc` is ignored in one dimension.
pub(crate) fn points(dim: usize, n_col: usize, n_bc: usize, seed: u64) -> Result<Points> {
    assert!(dim == 1 || dim == 2, "dim must be 1 or 2");
    assert!(n_col > 0, "at least one collocation point");
    let n_bc = if dim == 1 { 2 } else { 4 * n_bc };
    assert!(n_bc > 0, "at least one boundary point");
    let n = n_col + n_bc;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0_f64; dim * n];
    // Column j holds the point (x_j, y_j); column-major means index d + dim * j.
    for j in 0..n_col {
        for d in 0..dim {
            data[d + dim * j] = rng.gen::<f64>();
        }
    }
    let edge = |k: usize, per_edge: usize| (k as f64 + 0.5) / per_edge as f64;
    for j in n_col..n {
        let k = j - n_col;
        if dim == 1 {
            data[j] = if k == 0 { 0.0 } else { 1.0 };
        } else {
            let per_edge = n_bc / 4;
            let (x, y) = match k / per_edge {
                0 => (edge(k % per_edge, per_edge), 0.0),
                1 => (edge(k % per_edge, per_edge), 1.0),
                2 => (0.0, edge(k % per_edge, per_edge)),
                _ => (1.0, edge(k % per_edge, per_edge)),
            };
            data[2 * j] = x;
            data[2 * j + 1] = y;
        }
    }
    Ok(Points {
        tensor: Tensor::from_vec_col_major(vec![dim, n], data)?,
        n_col,
        n_bc,
    })
}

/// The evaluation grid and the exact solution on it.
///
/// One dimension uses `n` points including both ends; two dimensions use an
/// `n × n` grid. The exact solution is `sin(pi x)` and `sin(pi x) sin(pi y)`.
/// The grid comes back as `[dim, count]`, the layout the programs take.
pub(crate) fn evaluation(dim: usize, n: usize) -> Result<(Tensor, Vec<f64>)> {
    assert!(dim == 1 || dim == 2, "dim must be 1 or 2");
    assert!(n > 1, "an evaluation grid needs at least two points");
    let count = if dim == 1 { n } else { n * n };
    let mut data = vec![0.0_f64; dim * count];
    let step = 1.0 / (n as f64 - 1.0);
    for j in 0..count {
        let (x, y) = if dim == 1 {
            (j as f64 * step, 0.0)
        } else {
            ((j % n) as f64 * step, (j / n) as f64 * step)
        };
        data[dim * j] = x;
        if dim == 2 {
            data[dim * j + 1] = y;
        }
    }
    let exact: Vec<f64> = (0..count)
        .map(|j| {
            let x = data[dim * j];
            let sine = (std::f64::consts::PI * x).sin();
            if dim == 1 {
                sine
            } else {
                sine * (std::f64::consts::PI * data[dim * j + 1]).sin()
            }
        })
        .collect();
    Ok((Tensor::from_vec_col_major(vec![dim, count], data)?, exact))
}

/// The `index`-th coordinate of every point, as `[1, n]`.
fn coordinate(x: &TracedTensor, index: usize, n: usize) -> Result<TracedTensor> {
    x.slice(SliceConfig {
        starts: vec![index, 0],
        limits: vec![index + 1, n],
        strides: vec![1, 1],
    })
}

/// Columns `[start, end)` of a `[1, n]` tensor, as `[1, end - start]`.
fn columns(tensor: &TracedTensor, start: usize, end: usize) -> Result<TracedTensor> {
    tensor.slice(SliceConfig {
        starts: vec![0, start],
        limits: vec![1, end],
        strides: vec![1, 1],
    })
}

/// The unit tangents the Laplace operator differentiates along, one per
/// coordinate.
///
/// A `jvp` seed must carry data (`tenferro-ad` rejects an operation result), and
/// the transform exposes that seed as a program input: a residual program
/// therefore takes `2 * dim` inputs beyond its placeholders and the point
/// tensor, one per nested `jvp` call ([`seed_inputs`]), and the host binds them
/// like any other input. Both the data and the traced constants live here so one
/// seed serves every program of a run.
pub(crate) struct Seeds {
    values: Vec<Tensor>,
    traced: Vec<TracedTensor>,
}

impl Seeds {
    /// One seed per coordinate of a `[dim, n]` point tensor.
    pub(crate) fn new(trace: &TraceContext, dim: usize, n: usize) -> Result<Self> {
        let mut seeds = Self {
            values: Vec::with_capacity(dim),
            traced: Vec::with_capacity(dim),
        };
        for axis in 0..dim {
            let mut data = vec![0.0_f64; dim * n];
            for j in 0..n {
                data[axis + dim * j] = 1.0;
            }
            seeds
                .values
                .push(Tensor::from_vec_col_major(vec![dim, n], data.clone())?);
            seeds
                .traced
                .push(trace.from_vec_col_major(vec![dim, n], data)?);
        }
        Ok(seeds)
    }

    /// The seed data, in coordinate order.
    pub(crate) fn values(&self) -> &[Tensor] {
        &self.values
    }

    /// The seed the Laplace operator uses along coordinate `axis`.
    pub(crate) fn handle(&self, axis: usize) -> &TracedTensor {
        &self.traced[axis]
    }
}

/// The Laplace seeds in the order the programs take them as AD inputs.
///
/// `tenferro-ad` appends one program input per `jvp` call it imports, in the
/// order the calls enter the trace, and [`laplacian`] makes both calls of one
/// coordinate before moving to the next: the seeds arrive grouped by coordinate,
/// not interleaved. Since every seed has the same shape, binding the wrong order
/// still runs — it just differentiates along the wrong coordinates.
pub(crate) fn seed_inputs(seeds: &[Tensor]) -> Vec<&Tensor> {
    let mut inputs = Vec::with_capacity(2 * seeds.len());
    for seed in seeds {
        inputs.push(seed);
        inputs.push(seed);
    }
    inputs
}

/// The forcing term `f` with `-Laplace u = f`, as `[1, n]`.
pub(crate) fn forcing(x: &TracedTensor, dim: usize, n: usize) -> Result<TracedTensor> {
    let pi = std::f64::consts::PI;
    let sine = |t: &TracedTensor| t.scale_real(pi)?.sin();
    let first = sine(&coordinate(x, 0, n)?)?.scale_real(pi * pi)?;
    if dim == 1 {
        return Ok(first);
    }
    // 2 pi^2 sin(pi x) sin(pi y)
    first.mul(&sine(&coordinate(x, 1, n)?)?)?.scale_real(2.0)
}

/// The Laplace operator of the network output, `[1, n]`, with nested `jvp`s in
/// the traced input `x`.
fn laplacian(
    u: &TracedTensor,
    x: &TracedTensor,
    seeds: &Seeds,
    dim: usize,
) -> Result<TracedTensor> {
    let along = |axis: usize| -> Result<TracedTensor> {
        let seed = seeds.handle(axis);
        u.jvp(x, seed)?.jvp(x, seed)
    };
    let result = along(0)?;
    if dim == 1 {
        return Ok(result);
    }
    result.add(&along(1)?)
}

/// The residual vector `[n_col + n_bc, 1]` of the network with `params`.
pub(crate) fn residual(
    model: &Model,
    params: &[&TracedTensor],
    x: &TracedTensor,
    seeds: &Seeds,
    dim: usize,
    n_col: usize,
    n_bc: usize,
) -> Result<TracedTensor> {
    let n = n_col + n_bc;
    let u = model.forward(params, x)?;
    let pde = laplacian(&u, x, seeds, dim)?
        .add(&forcing(x, dim, n)?)?
        .neg()?
        .scale_real(1.0 / (n_col as f64).sqrt())?;
    let boundary = columns(&u, n_col, n)?.scale_real(1.0 / (n_bc as f64).sqrt())?;
    let pde = columns(&pde, 0, n_col)?;
    TracedTensor::concatenate(&[&pde, &boundary], 1)?.reshape([n, 1])
}

/// The transposed Jacobian `J^T` of `r` with respect to per-point `params`,
/// shape `[P, N]`.
///
/// One reverse pass gives every row block: `grad(sum(r), theta_b)` has the shape
/// of `theta_b`, `[out, in, N]`, whose trailing axis is the point, so flattening
/// the leading axes and concatenating in parameter order gives exactly the
/// `[P, N]` matrix the step solves with.
pub(crate) fn jacobian_transposed(
    r: &TracedTensor,
    params: &[&TracedTensor],
) -> Result<TracedTensor> {
    let n = r
        .try_concrete_shape()
        .expect("the residual shape is concrete")[0];
    let grads = r.reduce_sum(None)?.grad_many(params)?;
    let mut blocks = Vec::with_capacity(params.len());
    for (grad, param) in grads.into_iter().zip(params) {
        let grad = grad.expect("every parameter reaches the residual");
        let shape = param
            .try_concrete_shape()
            .expect("a parameter shape is concrete");
        assert_eq!(
            shape.last(),
            Some(&n),
            "the Jacobian needs per-point parameters with a trailing point axis"
        );
        let rows: usize = shape[..shape.len() - 1].iter().product();
        blocks.push(grad.reshape([rows, n])?);
    }
    let blocks: Vec<&TracedTensor> = blocks.iter().collect();
    TracedTensor::concatenate(&blocks, 0)
}
