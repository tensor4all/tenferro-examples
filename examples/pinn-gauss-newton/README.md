# pinn-gauss-newton: Levenberg-Marquardt against Adam on a Poisson PINN

A physics-informed network for

```text
1-D:  -u'' = pi^2 sin(pi x)                on (0, 1),  u = sin(pi x)
2-D:  -Laplace u = 2 pi^2 sin(pi x)sin(pi y) on (0, 1)^2, u = sin(pi x) sin(pi y)
```

with zero Dirichlet data, trained twice from the same seed and the same
initial parameters: once with Levenberg-Marquardt and once with Adam. The
network is a tanh MLP `[dim, 32, 32, 1]` with 1153 (1-D) and 1185 (2-D)
parameters. Both problems have an exact solution, so every run reports the
relative L2 error against it.

The point of the example is the *Jacobian*, not the model: Levenberg-Marquardt
needs the `N x P` derivative of the residual vector, and this is how to get it
in one reverse pass.

## The batched-Jacobian contract

The network parameters are fed to the residual twice:

- `theta`, the plain parameters `[out, in]` / `[out]`, which the host owns and
  Adam updates; and
- `theta_b`, the **per-point copy** of the same numbers, `[out, in, N]` /
  `[out, N]`, supplied by the host as an explicit broadcast copy on every step.

`grad(sum(r), theta_b)` then returns a tensor of the shape of `theta_b`, whose
trailing point axis holds one partial derivative per point. Flattening the
leading axes and concatenating in parameter order gives `J^T`, and
`(J^T J + lambda I) delta = -J^T r` is solved inside the graph. Cost: each copy
is `points * P * 8` bytes — 18.9 MB (1-D) and 20.6 MB (2-D) with the default
sizes, printed by every run — and one LM iteration makes two of them, one for
the step and one for the candidate it evaluates.

The damped solve and the two triangular solves of the factorized
`J^T J + lambda I` are `tenferro-linalg` operations. They are deliberately
**not** differentiated, and `lambda` is a program input, not a constant: a
factorization that fails (the matrix is not positive definite) surfaces as a
numerical error and the host raises `lambda` and retries the same iteration.

## What the graph needs from the host

Three details are visible in `src/main.rs` and worth knowing before writing a
similar example:

- **Each program gets its own `TraceContext`.** An AD transform walks the whole
  trace it is applied in, and a `jvp` cannot cross a linalg extension
  operation (`UnsupportedAdRule`), so the differentiated residual and the
  damped solve never share one. The candidate residual is a second program for
  the same reason.
- **The AD inputs are program inputs.** `tenferro-ad` turns every `jvp` seed it
  imports and every reverse-pass cotangent into an input of the compiled
  program, so a run binds them after the declared inputs: the Laplace seed
  twice per coordinate (the nested `jvp`s of one second derivative), then the
  scalar cotangent of the reverse pass. `Program::reads` checks the count.
- **Inputs are matched by key, not by position.** The compiler folds some
  declared placeholders into constants, so the slot order is only trustworthy
  through `CompiledGraph::input_keys()` / `TracedTensor::input_key()`.

The layer itself is written as a broadcast and a reduction over the input
features instead of a batched `dot_general`: tenferro 0.11.0 mis-indexes the
left-hand batch slice of a batched contraction (with `lhs [in, N]` and
`rhs [out, in, N]` it returns the slice of point `out` for every point). The
elementwise form costs the `out * in * N` product tensor, about 17 MB in the
widest layer.

## Running

```bash
cargo run --locked --release -p pinn_gauss_newton -- --dim 1 --optimizer lm
cargo run --locked --release -p pinn_gauss_newton -- --dim 1 --optimizer adam
cargo run --locked --release -p pinn_gauss_newton -- --dim 2 --optimizer lm
cargo run --locked --release -p pinn_gauss_newton -- --dim 2 --optimizer adam
cargo run --locked --release -p pinn_gauss_newton -- --optimizer both --plot loss.png
```

Flags: `--dim 1|2` (default 1), `--optimizer lm|adam|both` (default `lm`),
`--epochs N` (steps for whichever optimizer runs, both under `both`; defaults
150 LM iterations and 2000 Adam epochs), `--seed K` (default 42), `--plot
<path>` (loss curves, x = seconds, y = loss on a log axis).

## Measured results

Single-threaded `CpuEngine::serial()`, default sizes and seed, one thread, on
the machine this example was developed on (`--locked`, release profile):

| run | steps | wall time | final loss | relL2 | bar | |
| --- | --- | --- | --- | --- | --- | --- |
| 1-D LM | 150 | 24.1 s | 5.74e-08 | 1.92e-05 | ≤ 1e-4, ≤ 60 s | met |
| 1-D Adam | 2000 | 113.0 s | 5.38e-04 | 1.53e-03 | ≤ 5e-2 | met |
| 1-D Adam, equal wall time | 430 | 25.2 s | 4.57e-01 | 5.42e-01 | LM ≥ 10x smaller | met (2.8e4x) |
| 2-D LM | 150 | 35.4 s | 1.08e-06 | 5.77e-04 | ≤ 1e-3 | met |
| 2-D Adam | 2000 | 218.2 s | 1.82e-02 | 1.37e-01 | ≤ 1e-1 | **missed** |
| 2-D Adam, equal wall time | 320 | 35.7 s | 3.28e+00 | 1.81e+00 | LM ≥ 10x smaller | met (3.1e3x) |

The equal-wall-time rows repeat Adam with the largest epoch count whose wall
time stays within ±20 % of the LM run of the same dimension (24.1 s against
25.2 s, +4.7 %; 35.4 s against 35.7 s, +1.0 %).

The batched Jacobian is what makes the comparison look like this: LM reaches
`5.8e-04` in two dimensions after 150 iterations — 35 s — while Adam needs 218 s
(2000 epochs) for `1.4e-01` and 449 s (4000 epochs) for `1.02e-01`, still short
of the `1e-1` bar. Adam's
own bar is thus missed, and the trend is in the `--epochs` runs: a step-decayed
Adam at `1e-3` is simply far slower here than a second-order step with an exact
Jacobian, which is the point of the example. No parameter was tuned to close the
gap.

### The seed order the host must bind

One detail is easy to get wrong and expensive to notice: the Laplace seeds are
per-coordinate one-hot tensors of the same shape, so binding them in the wrong
order still runs, with no shape error, and quietly differentiates along the
wrong coordinates. The order `tenferro-ad` appends them in is the order the
`jvp` calls enter the trace — both calls of coordinate 0, then both of
coordinate 1 — so it is grouped per coordinate, not interleaved
(`poisson::seed_inputs`). Interleaving it dropped the 2-D run from `relL2`
`5.8e-04` to `1.43`, and `residual_matches_finite_differences_2d` fails with a
permuted order, which is the check that pins it.

## Tests

```bash
cargo test --release -p pinn_gauss_newton
```

Beyond the point set, the forcing term and the LM accept/reject rule, three
tests check the parts that are easy to get subtly wrong:

- `batched_jacobian_matches_per_parameter_jvps` — the batched reverse-pass
  Jacobian equals per-parameter forward-mode JVPs, entry by entry. This is the
  contract above, measured.
- `residual_matches_finite_differences_2d` — the residual of the program the
  example trains on (with its per-point parameters and its bound Laplace seeds)
  equals finite differences of the same network, including the boundary rows,
  and the loss the step program reports equals the sum of squares of that
  residual.
- `loss_gradient_matches_finite_differences` — the reverse-mode gradient of the
  loss agrees with a directional finite difference.
