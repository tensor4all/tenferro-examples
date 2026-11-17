# KdV PINN

This tutorial is about a concrete question: can we train a neural network
$u_\theta(x,t)$ to behave like the solution of a partial differential
equation, without giving the network a dense table of solution values?

This example answers that question for the
Korteweg-de Vries (KdV) equation. It builds a neural approximation to a known
single-soliton solution, constructs the PDE residual with tenferro traced
autodiff, differentiates one scalar objective with respect to the network
parameters, and updates those parameters with Adam.

Use this sample after the [Traced autodiff tutorial](https://tensor4all.org/tenferro-rs/tutorials/traced-autodiff-jax-style.html)
when you want to see a larger traced-graph workflow with reusable compiled
programs, repeated `jvp` calls, scalar loss gradients, and a custom optimizer
loop.

## Problem Being Solved

The unknown is a scalar field $u(x,t)$, where $x$ is a spatial coordinate
and $t$ is time. The Korteweg-de Vries equation used by the sample is

$$
u_t(x,t) + 6u(x,t)u_x(x,t) + u_{xxx}(x,t) = 0.
$$

Here $u_t$ is the time derivative, $u_x$ is the first spatial derivative,
and $u_{xxx}$ is the third spatial derivative. The nonlinear term
$6u u_x$ makes the wave steepen as it moves, while the dispersive term
$u_{xxx}$ spreads different wavelengths at different speeds. Their balance
is what allows KdV soliton waves to keep their shape while traveling.

This sample solves the equation on the finite space-time domain

$$
(x,t) \in \Omega = [-5,5] \times [0,1].
$$

For training targets and evaluation, the sample uses the exact single-soliton
solution

$$
u_\star(x,t) = 2\,\operatorname{sech}^2(x - 4t),
\qquad
\operatorname{sech}(z)=\frac{1}{\cosh(z)}.
$$

The point is not to fit a dense labelled trajectory. The network sees the exact
solution only at the initial line and the two spatial boundaries. Inside the
domain, it is trained by asking the predicted field to satisfy the differential
equation.

## Model

The model is a fully connected neural network

$$
u_\theta(x,t) = f_\theta([x,t]) \in \mathbb{R}.
$$

The input is one coordinate pair $[x,t]$, and the output is one scalar value
that approximates $u(x,t)$. In the code, the architecture is `[2, 64, 64, 1]`:
two inputs, two hidden layers with 64 units each, and one output. The hidden
layers use `tanh` activations.

The optimized variables are the network parameters

$$
\theta = \{W_\ell,b_\ell\}_{\ell=1}^{3},
$$

not the coordinates $x$ and $t$. In tenferro, each $W_\ell$ and $b_\ell$
is represented by a `TracedTensor` placeholder while the graph is built. The
training loop binds those placeholders to concrete `Tensor` buffers and updates
those buffers after each gradient evaluation.

## Training Points

Each epoch samples three sets of points:

$$
\mathcal{C} = \{(x_i^c,t_i^c)\}_{i=1}^{N_\mathrm{col}},
\qquad N_\mathrm{col}=1024,
$$

for interior collocation points, sampled uniformly in $\Omega$;

$$
\mathcal{I} = \{x_i^0\}_{i=1}^{N_\mathrm{ic}},
\qquad N_\mathrm{ic}=128,
$$

for initial-condition points at $t=0$, with target
$u_\star(x_i^0,0)=2\,\operatorname{sech}^2(x_i^0)$; and

$$
\mathcal{B} = \{(x_i^b,t_i^b)\}_{i=1}^{N_\mathrm{bc}},
\qquad N_\mathrm{bc}=128,
$$

for boundary points on $x=-5$ and $x=5$, with target
$u_\star(x_i^b,t_i^b)$. Half of the boundary samples are placed at the left
boundary and half at the right boundary.

The collocation points are the PINN-specific part. They do not have labelled
$u_\star$ values in the loss. Instead, they are where the network is penalized
for violating the KdV equation.

## Objective Function

The PDE residual of the neural network is

$$
r_\theta(x,t)
  = \partial_t u_\theta(x,t)
  + 6u_\theta(x,t)\partial_x u_\theta(x,t)
  + \partial_x^3 u_\theta(x,t).
$$

If $u_\theta$ exactly solved the KdV equation, this residual would be zero at
every point in the domain. The sample minimizes the composite loss

$$
\mathcal{L}(\theta)
  = \lambda_\mathrm{pde}\mathcal{L}_\mathrm{pde}
  + \lambda_\mathrm{ic}\mathcal{L}_\mathrm{ic}
  + \lambda_\mathrm{bc}\mathcal{L}_\mathrm{bc},
$$

where the default weights are
$\lambda_\mathrm{pde}=\lambda_\mathrm{ic}=\lambda_\mathrm{bc}=1$, and

$$
\mathcal{L}_\mathrm{pde}
  = \frac{1}{N_\mathrm{col}}\sum_{i=1}^{N_\mathrm{col}}
    r_\theta(x_i^c,t_i^c)^2,
$$

$$
\mathcal{L}_\mathrm{ic}
  = \frac{1}{N_\mathrm{ic}}\sum_{i=1}^{N_\mathrm{ic}}
    \left(u_\theta(x_i^0,0)-u_\star(x_i^0,0)\right)^2,
$$

and

$$
\mathcal{L}_\mathrm{bc}
  = \frac{1}{N_\mathrm{bc}}\sum_{i=1}^{N_\mathrm{bc}}
    \left(u_\theta(x_i^b,t_i^b)-u_\star(x_i^b,t_i^b)\right)^2.
$$

Training minimizes $\mathcal{L}(\theta)$ with respect to the network
parameters $\theta$. The $x$ and $t$ values are sampled inputs used to
evaluate the loss; they are not learned.

## Run The Sample

From the repository root of `tenferro-examples`:

```bash
cargo run --release -p kdv_pinn
```

The run prints training loss every 50 epochs, then reports the final loss and
the relative L2 error at `t = 0.5`. The default configuration trains for 3000
epochs on CPU. Pass `--epochs N` for a shorter run:

```bash
cargo run --release -p kdv_pinn -- --epochs 20
```

`--seed K` sets the random seed used for the network initialization and for
every sampled batch; the default is 42 and the same seed reproduces a run
exactly. The run prints the seed, and the time spent building and compiling the
training program, as its first two lines.

Optional plot outputs are available:

```bash
cargo run --release -p kdv_pinn -- --gif kdv_pinn.gif --loss-png loss.png
```

`--loss-png` writes a log-scale training-loss plot. `--gif` writes an animated
comparison between the analytic solution and the trained model prediction.

The plots contain no text, so the sample needs no font library: in the loss
plot the red curve is the loss per epoch on a logarithmic y-axis, and in each
GIF frame the blue curve is the analytic soliton and the red curve the
prediction, over `x` in `[-5, 5]` and `u` in `[-0.5, 2.5]`.

CI runs the unit tests and a 20-epoch training run that also writes both plots,
which are uploaded as workflow artifacts. It does not run the full 3000-epoch
training.

## Results And Cost

The accuracy metric is the relative $L^2$ error of the predicted
$u_\theta(\cdot, 0.5)$ against the exact soliton on 100 equally spaced points of
`[-5, 5]`, after the default 3000 epochs.

Measured on an AMD Ryzen AI MAX+ 395 with the serial CPU engine (one worker
thread), `--release`, no plot output. "build + compile" is the time from process
start to the first epoch, i.e. building the traced graph and compiling its
programs; the per-epoch figure is derived from a 3000-epoch run.

| training program | relative $L^2$ error at $t = 0.5$ | final loss | build + compile | per epoch |
| --- | --- | --- | --- | --- |
| one per parameter gradient | 1.6e-2 … 2.4e-2 | 4.5e-4 … 1.2e-3 | 0.02 s (7 training programs plus the evaluation one) | 104 ms |
| one for the loss and all gradients | 1.3e-2 … 3.3e-2 | 3.9e-4 … 7.8e-4 | 0.005 s (the training program; the evaluation program is compiled after training) | 27 ms |

Both rows are ranges over three runs: three seeds after the migration, three
unseeded runs before it. The forward pass now happens once per epoch instead of
once per parameter gradient, so the run is 3.9x faster at the same accuracy, and
peak memory drops from 158 MB to 134 MB.

Reversing a graph that already contains the three nested `jvp` calls (for
$u_x$, $u_{xx}$ and $u_{xxx}$) does not make compilation expensive: the whole
training program, loss and six gradients, compiles in 5 ms.

## How It Is Structured

The sample keeps each PINN concern in a small module, and the model and
optimizer code it shares with the other SciML examples in `crates/sciml-kit`:

| File | Role |
| --- | --- |
| `crates/sciml-kit/src/nn.rs` | Builds the MLP from `TracedTensor` parameter placeholders (Xavier initialization, activation as a parameter). |
| `crates/sciml-kit/src/func.rs` | The ordered parameter set: placeholders and initial values in one fixed order. |
| `crates/sciml-kit/src/optim.rs` | `Sgd` and `Adam` updates on the concrete parameter tensors, plus the step-decayed learning-rate schedule. |
| `src/pde.rs` | Builds the KdV residual with chained `jvp` calls for `u_t`, `u_x`, `u_xx`, and `u_xxx`. |
| `src/loss.rs` | Combines PDE, initial-condition, and boundary losses with scalar weights. |
| `src/sampler.rs` | Samples collocation, initial-condition, and boundary points. |
| `src/plot.rs` | Writes optional loss-curve PNG and solution-comparison GIF outputs. |

The graph-building phase creates placeholders for network parameters first and
training batches after them, then compiles **one** program whose outputs are the
loss and every parameter gradient, and prepares it **once**. Each epoch samples
fresh tensors, binds them, and runs that one prepared program, so the forward
pass happens once per epoch instead of once per parameter gradient.

Constants are written inline rather than as full-size host buffers, with one
exception. The `t = 0` column is a rank-0 zero expanded with
`broadcast_in_dim`. The ones used as `jvp` tangents are not: the tangent of a
`jvp` (and the cotangent of a `vjp`) must carry concrete data, and tenferro 0.11
rejects an operation result such as a broadcast with `"jvp tangent must have
concrete tensor data"`, so `ones_like` stays a full-shape constant leaf. Both
are built once when the graph is built, so neither costs anything per epoch.

## Traced Autodiff Pattern

The PDE residual is the key traced-AD pattern:

This abbreviated autodiff fragment intentionally omits its surrounding network definition.
```rust
let ones_x = ones_like(x)?;
let ones_t = ones_like(t)?;
let u_t = u.jvp(t, &ones_t)?;
let u_x = u.jvp(x, &ones_x)?;
let u_xx = u_x.jvp(x, &ones_x)?;
let u_xxx = u_xx.jvp(x, &ones_x)?;
let nonlinear = u.mul(&u_x)?.scale_real(6.0)?;
let residual = u_t.add(&nonlinear)?.add(&u_xxx)?;
```

Each `jvp` differentiates with respect to a coordinate placeholder while using a
tensor of ones as the tangent direction. The first `jvp` with respect to `t`
gives $u_t$. Repeated `jvp` calls with respect to `x` give $u_x$, $u_{xx}$,
and $u_{xxx}$. This builds $r_\theta(x,t)$ as another traced tensor in the
same graph.

That residual is squared and averaged before the PDE loss weight is applied,
matching the composite objective:

```text
loss = lambda_pde * mean(residual^2)
     + lambda_ic  * mean((u_ic - u_ic_true)^2)
     + lambda_bc  * mean((u_bc - u_bc_true)^2)
```

This is different from scaling the residual before squaring it, which would
apply the PDE weight twice.

After the scalar `total_loss` tensor is built, one reverse pass gives the loss
and every parameter gradient together:

This abbreviated autodiff fragment intentionally omits its surrounding network definition.
```rust
let (loss_value, grads) = total_loss.value_and_grad(&net.placeholders())?;
let grads: Vec<TracedTensor> = grads.into_iter().flatten().collect();
let mut outputs: Vec<&TracedTensor> = vec![&loss_value];
outputs.extend(grads.iter());
let program = compiler.compile_many(&outputs)?;
```

Those gradients are $\partial\mathcal{L}/\partial W_\ell$ and
$\partial\mathcal{L}/\partial b_\ell$. The single compiled program is prepared
once and executed each epoch; Adam applies the resulting concrete tensors to the
current parameter buffers.

## When To Use This Pattern

Use this structure when the model parameters, PDE inputs, and training data
change independently across many executions. `TracedTensor` lets the sample
compile the computational structure once, then bind concrete tensors for each
training batch and parameter update.

For shorter introductions, start with [Traced autodiff, JAX style](https://tensor4all.org/tenferro-rs/tutorials/traced-autodiff-jax-style.html)
and [Eager autodiff, PyTorch style](https://tensor4all.org/tenferro-rs/tutorials/eager-autodiff-pytorch-style.html).
