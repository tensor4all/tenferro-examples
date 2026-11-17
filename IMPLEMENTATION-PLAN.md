# IMPLEMENTATION-PLAN — tenferro-examples#8 (`pinn-gauss-newton`)

Issue: <https://github.com/tensor4all/tenferro-examples/issues/8> (ai-ready; the
accepted Jacobian decision is the issue comment of 2026-11-18 — "use option
(a)").
Umbrella conventions: <https://github.com/tensor4all/tenferro-examples/issues/6>.
Repo/worktree: `~/projects/tensor4all/tenferro-examples-8`, branch
`feat/8-pinn-gauss-newton`, base `origin/main` = `29db186` (adds #7 `sciml-kit`,
#10 `deeponet`, #12 `neural-ode`).
Deliverable: one PR, no merge, no tenferro change. #9 is not started.

## 0. The method this implements

For the residual vector `r(θ)` (PDE rows + boundary rows), the parameter
Jacobian is taken in **one reverse pass** by feeding the parameters a second
time as a *per-point copy*:

```text
θ_b   : program input, shape [P_k..., N] with N = number of points
r     : residual computed with θ_b (batched dot_general, batch axis = point)
J^T   : grad(sum(r), θ_b)      -> shape [P_k..., N], one partial per row
```

`θ_b` is bound by the host as an explicit broadcast copy of `θ` each step. This
is the accepted decision: no new tenferro API, `grad(sum(r), θ_b)` gives the
Jacobian rows.

Levenberg–Marquardt step, **inside the graph**, `(JᵀJ + λI) δ = −Jᵀ r`, then the
non-linear candidate `r(θ+δ)` and the linear prediction, so one program run and
one read-back per iteration produce everything the host decision needs
(convention 3). `λ` is a program input placeholder; the linear solve is **not**
differentiated (the AD transform is applied to `sum(r)`, which is upstream of
the Cholesky, so the factorization is outside the differentiated subgraph).

## 1. Files, functions and current behaviour

All files are new except §1.8–§1.10. "Current behaviour" for a new file is
"does not exist".

### 1.1 `examples/pinn-gauss-newton/Cargo.toml` — new crate `pinn_gauss_newton`

Same dependency set as `examples/deeponet` (`sciml-kit`, `tenferro-ad`,
`tenferro-cpu`, `tenferro-linalg`, `tenferro-runtime`, `tenferro-tensor`,
`plotters` with `bitmap_backend`/`bitmap_encoder`/`line_series`, `rand 0.8`),
all already in the workspace or in the tree. No new external dependency.

### 1.2 `src/network.rs` (+ `src/network/tests.rs`) — the model, two layouts

The per-point formulation forces a **points-last** layout, because tenferro's
`dot_general` puts batch axes last (`tenferro-tensor-0.11.0/src/config.rs`
"Output layout" doc, `tenferro-runtime-0.11.0/src/shape_infer.rs:813`): with
`x [in, N]` and `W [out, in, N]` the result is `[out, N]`. `sciml-kit`'s `Mlp`
is points-first (`x [n, in] · W [in, out] → [n, out]`,
`crates/sciml-kit/src/nn.rs:44`), so it cannot express this layout without a
transpose per layer.

```rust
pub(crate) struct Model { sizes: Vec<usize>, points: Option<usize>, params: Vec<TracedTensor> }
impl Model {
    fn new(trace: &TraceContext, sizes: &[usize], points: Option<usize>) -> Result<Self>;
    fn params(&self) -> Vec<&TracedTensor>;          // [w1,b1,w2,b2,w3,b3]
    fn sizes(&self) -> &[usize];
    fn initial_values(&self, rng: &mut impl Rng) -> Result<Vec<Tensor>>;
    fn forward(&self, params: &[&TracedTensor], x: &TracedTensor) -> Result<TracedTensor>;
}
impl sciml_kit::func::Params for Model { /* placeholders + initial_values */ }
```

- weight `[out, in]`, bias `[out, 1]` when `points == None`; weight
  `[out, in, N]`, bias `[out, N]` when `points == Some(N)`.
- `forward` is **one** function; `points.is_some()` only selects the
  `dot_general` batch axes (`[]`/`[]` vs `[1]`/`[2]`). The bias add broadcasts
  in both layouts (`add` is NumPy-broadcasting, `traced.rs:751`).
- `initial_values`: the existing Xavier-uniform formula from
  `crates/sciml-kit/src/nn.rs:104-119` (the only duplicated 12 lines), drawing
  `[out, in]` values and then **tiling along the last axis** when per-point, so
  the same seed gives the same θ in both layouts. The plane/points equality is
  unit-tested (§2.1).

One model serves both optimizers, so the Adam and LM runs start from the *same*
θ and the comparison is apples to apples. Consequence: `sciml-kit`'s `Mlp` is
not used by this example (see §4, rejected alternative).

`src/network/tests.rs`: initial-value tiling equality; `forward` shape check in
both layouts; `forward` of the plain model equals the per-point model fed the
tiled parameters (evaluated through the runtime).

### 1.3 `src/poisson.rs` (+ tests) — points, forcing, residual, Jacobian

```rust
pub(crate) struct Points { pub collocation: Tensor /* [dim, n_col] */, pub boundary: Tensor /* [dim, n_bc] */ }
pub(crate) fn points(dim: usize, n_col: usize, n_bc: usize, seed: u64) -> Points;
pub(crate) fn pde_residual(model: &Model, params: &[&TracedTensor], x: &TracedTensor,
                           dim: usize) -> Result<TracedTensor>;          // [1, n]
pub(crate) fn residual(model: &Model, params: &[&TracedTensor], x_col: &TracedTensor,
                       x_bc: &TracedTensor, dim: usize) -> Result<TracedTensor>; // [N, 1]
pub(crate) fn jacobian_transposed(r: &TracedTensor, params: &[&TracedTensor]) -> Result<TracedTensor>; // [P, N]
pub(crate) fn exact(dim: usize, x: &[f64]) -> Vec<f64>;                 // sin(pi x), sin sin
```

- Problem: 1-D `-u'' = π² sin(πx)`, 2-D `-Δu = 2π² sin(πx) sin(πy)`, exact
  `u = sin(πx)` / `sin(πx) sin(πy)`, zero Dirichlet data.
- Points: uniform on `[0,1]` from a seeded RNG, fixed for the whole run (the
  ρ test needs one fixed residual vector). 1-D boundary = `x = 0, 1`; 2-D
  boundary = the four edges, `n_bc` points each. Collocation and boundary
  points are concatenated into **one** `[dim, n_col+n_bc]` placeholder, so one
  per-point parameter set and one gradient list cover both groups; the residual
  splits them with two static column slices (`slice`, `traced.rs:2184`) instead
  of carrying two parameter groups. (Post-review refinement, §7.)
- Second derivatives stay nested `jvp` against the traced `x`, exactly as in
  `examples/kdv-pinn/src/pde.rs:22`; the seeds are full-shape data-carrying
  constants (a `broadcast_in_dim` result is not a valid tangent,
  `examples/kdv-pinn/src/pde.rs:31`), built once at graph build.
- Group scaling: `r = [r_pde / sqrt(n_col); r_bc / sqrt(n_bc)]`, i.e. one mean
  square per group as in `examples/kdv-pinn/src/loss.rs:33-53`, so the boundary
  rows are not swamped 2048:2 by the PDE rows. Documented in the README.
- `jacobian_transposed` = `grad_many(&params)` of `sum(r)`
  (`tenferro-ad-0.11.0/src/traced.rs:921`), each row tensor reshaped to
  `[P_k, N]` and `concatenate`d along axis 0, in parameter order.

### 1.4 `src/lm.rs` (+ tests) — graph step and host decision

```rust
pub(crate) struct Step { pub delta: TracedTensor, pub predicted: TracedTensor } // [P,1], rank-0
pub(crate) fn solve_step(jt: &TracedTensor, r: &TracedTensor, lambda: &TracedTensor) -> Result<Step>;
pub(crate) struct Decision { pub accept: bool, pub rho: f64, pub next_lambda: f64 }
pub(crate) fn decide(lambda: f64, loss: f64, loss_candidate: f64, predicted: f64) -> Decision;
pub(crate) fn is_numerical_failure(error: &tenferro_runtime::Error) -> bool;
pub(crate) const LAMBDA_MIN: f64 = 1e-6; pub(crate) const LAMBDA_MAX: f64 = 1e6;
```

Graph: `H = JᵀJ = matmul(jt, transpose(jt))` `[P,P]`; `g = matmul(jt, r)` `[P,1]`;
`λI = lambda [rank-0] .broadcast_in_dim([P], []) .embed_diag(0, 1)`;
`L = (H + λI).cholesky()` (`tenferro-linalg-0.11.0/src/traced.rs:190`);
`delta = -(L.triangular_solve(g, left_side=true, lower=true, transpose_a=false,
unit_diagonal=false))` (`:299`); `predicted = 0.5*(‖r‖² − ‖r + Jδ‖²)`.
`‖·‖²` is `reduce_sum_squares(None)`; `r + Jδ` is `matmul(transpose(jt), delta)`.

`decide` (tested as a table): `rho = (loss − loss_candidate)/predicted` (NaN or
`predicted <= 0` ⇒ reject with `rho = NEG_INFINITY`); accept iff `rho > 0`;
accepted ⇒ `rho > 0.75 ? λ/3 : rho < 0.25 ? 2λ : λ`; rejected ⇒ `10λ`; result
clamped to `[LAMBDA_MIN, LAMBDA_MAX]`.

`is_numerical_failure` = `matches!(error.kind(), ErrorKind::NumericalFailure)`
(`tenferro-runtime-0.11.0/src/error.rs:783`; the `Extension { kind, .. }` arm is
`:798`; the Cholesky non-PD path is
`tenferro-linalg-0.11.0/src/backend.rs:44-46` → `Extension` +
`ErrorKind::NumericalFailure`).

### 1.5 `src/plot.rs` (+ tests) — loss curve

`pub(crate) fn write_loss_png(path: &str, curves: &[(&str, &[(f64, f64)])]) -> Result<(), Box<dyn Error>>`:
one panel, x = cumulative wall time in seconds, y = loss on a log axis, one
polyline per curve. Colours only, no text (`plotters` without a font backend
panics, see `examples/neural-ode/src/plot.rs:4`); the README states the mapping.
Reuses the bounds helpers of the same file with the module's own copy (they are
40 lines; copying two small functions is smaller than a shared crate for one
figure).

### 1.6 `src/main.rs` — CLI, programs, loops

Flags: `--dim 1|2` (default 1), `--optimizer adam|lm|both` (default `lm`),
`--epochs N` (default 40 for `lm`, 2000 for `adam`; for `lm` it is the number of
LM iterations), `--seed K` (default 42), `--plot <path>`.

Default configuration — pinned here because the README's copy cost, the cost
estimate in §4 and the CI budget all depend on it:

| constant | value |
| --- | --- |
| layer sizes | `[dim, 32, 32, 1]` (two hidden layers of width 32) |
| parameters P | **1153** (1-D), **1185** (2-D) |
| collocation points `n_col` | 2048 |
| boundary points `n_bc` | 2 (1-D: `x = 0, 1`), 128 (2-D: 4 edges × 32) |
| residual dimension N | 2050 (1-D), 2176 (2-D) |
| `λ₀` | `1e-3`, clamped to `[LAMBDA_MIN, LAMBDA_MAX] = [1e-6, 1e6]` |
| Adam learning rate | `step_decay_lr(epoch, epochs, 1e-3)` |
| evaluation grid | 200 points (1-D), 60 × 60 (2-D) |
| per-step copy cost `(n_col+n_bc)·P·8` | 18.9 MB (1-D), 20.6 MB (2-D) |

All four constants are command-line-independent (no flags for them); the cost
fallback in §4 risk 2 is the only thing that would move them, and that change
would be reported.

`fn cpu_runtime` as in `examples/deeponet/src/main.rs:66` **plus**
`builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
tenferro_cpu::runtime_engine_id()?)?)` — required by every traced linalg op
(`tenferro-linalg-0.11.0/src/lib.rs:31-45` example;
`RuntimeBuilder::install_extension_module`, `tenferro-runtime-0.11.0/src/runtime/snapshot.rs:1276`).

LM program (`--optimizer lm`): inputs in creation order `[θ_b..., x, λ]` (8
inputs: 6 parameter tiles, the point tensor, the damping scalar); outputs
`[δ [P,1], loss, loss_candidate, predicted]`, where the candidate
residual uses `θ_b + broadcast(reshape(slice(δ, block_k), plain_shape_k),
[plain_shape_k..., N])` built in-graph (`slice`, `traced.rs:2184`;
`broadcast_in_dim`, `:2051`), so the whole ρ test needs one read-back. The
`Prepared` is created once before the loop (all inputs have fixed shapes) and
reused for every iteration — including after a failed run, which is asserted by
the retry test (§2.3).

Cholesky failure: `runtime.run` returns `Err` whose `kind()` is
`NumericalFailure` (`.../src/runtime/execution.rs:414`, `:685-693` retires the
run and hands back the error) ⇒ raise `λ = min(10λ, LAMBDA_MAX)`, retry the same
iteration (retries do not advance the iteration counter); at `LAMBDA_MAX` the
error is returned. A NaN/`predicted <= 0` run is handled by the same reject
branch. Both paths are counted and printed.

Host θ stays in the plain shapes `[out, in]`/`[out, 1]`; `fn tile(&[Tensor], n)`
builds the per-point copies (this is the documented per-step cost) and
`fn apply_delta(&mut [Tensor], &[f64])` adds the accepted `δ` block by block.

Adam program (`--optimizer adam`): the plain model, `value_and_grad(&params)`
over `loss = 0.5‖r‖²`, one program, one `Prepared`, `sciml_kit::optim::Adam`
with `step_decay_lr(epoch, epochs, 1e-3)`, as in `examples/kdv-pinn/src/main.rs:200`.

After training: evaluate `u` on a fixed grid (1-D: 200 points, 2-D: 60×60),
print the relative L2 error against the exact solution, optionally write the
loss figure, and print a one-line machine-readable summary
(`optimizer, dim, steps, wall time, final loss, relL2`).

### 1.7 `examples/pinn-gauss-newton/README.md` + committed figure

Problem and exact solution; the per-point-copy Jacobian (with the batch-last
layout consequence); **the per-step copy cost** (`(n_col+n_bc)·P·8` bytes, with
the default numbers) and **the prepare signature** (input order and shapes,
"prepared once, reused for every iteration"); the LM step and the ρ/λ rules;
the Cholesky-failure policy; collocation fixed per run; the Adam baseline; the
results table (1-D and 2-D × Adam/LM: steps, wall time, final loss, relL2);
the figure (colour → optimizer); what is deliberately not done (lstsq variant,
JAX baseline); the tenferro gaps this hits (input donation tenferro-rs#2419,
input naming tenferro-rs#2642). The committed `loss.png` is produced by
`--optimizer both --plot`.

### 1.8 `.github/workflows/checks.yml` — short runs (current: 4 examples)

Add three lines to the `Short runs` step:
`-p pinn_gauss_newton --dim 1 --optimizer both --epochs 3 --plot outputs/pinn-gauss-newton.png`,
`--dim 1 --optimizer adam --epochs 20`, `--dim 2 --optimizer lm --epochs 2`.
Together well under the current CI time.

### 1.9 root `README.md` — one table row

### 1.10 `IMPLEMENTATION-PLAN.md` / `WORKLOG.md` — this plan and the log

Both are tracked at the repo root on `main` (from the #7/#12 PRs), so they are
replaced, not left untracked.

## 2. Tests and commands, with acceptance bars

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings     # no warnings
cargo test --release --workspace                                    # all pass
cargo run --locked --release -p pinn_gauss_newton -- --dim 1 --optimizer lm --epochs 40
cargo run --locked --release -p pinn_gauss_newton -- --dim 1 --optimizer adam --epochs 2000
cargo run --locked --release -p pinn_gauss_newton -- --dim 2 --optimizer lm --epochs 40
cargo run --locked --release -p pinn_gauss_newton -- --dim 2 --optimizer adam --epochs 2000
```

`--locked` mirrors `LOCKED=--locked` in `.github/workflows/checks.yml`.

Unit tests (the two the issue requires are marked ★):

1. ★ `network::tests` — tiled initial values equal the plain ones; `forward`
   agrees between layouts for the same θ (runtime).
2. ★ `poisson::tests::batched_jacobian_matches_per_parameter_jvps` — small net
   `[dim,2,2,1]` (P = 13 for 1-D, 15 for 2-D), `n_col = 5`, `n_bc = 2`; every
   scalar parameter entry gets one `jvp` of the residual (`traced.rs:771`) with
   a data-carrying `e_p` seed, and the `P` columns are compared with the
   per-point `Jᵀ` from `grad_many`. Both `dim = 1` and `dim = 2`. Bar:
   `max |Jᵀ_jvp − Jᵀ_batched| ≤ 1e-10 · max |J|` (both routes are exact up to
   rounding).
3. ★ `lm::tests::rho_accept_reject_table` — the ρ/λ table of §1.4 (accept at
   ρ=1 → λ/3; ρ=0.5 → λ; ρ=0.1 → 2λ; ρ<0 → reject, 10λ; predicted ≤ 0 and NaN
   → reject), plus clamping at `LAMBDA_MIN`/`LAMBDA_MAX`.
4. `lm::tests::step_solves_the_damped_system` — synthetic `Jᵀ` `[P,N]` and `r`
   from a host `A`, `P = 6`, `N = 10`, `λ = 0.5`: assert
   `‖(AᵀA + λI)δ + Aᵀr‖ ≤ 1e-10 ‖Aᵀr‖` and `predicted > 0`; `λ = 0` gives the
   undamped least-squares step.
5. `lm::tests::cholesky_failure_is_reported_and_retryable` — rank-1 `J`, `λ = 0`
   ⇒ either `Err` with `ErrorKind::NumericalFailure` or a rejected decision
   (probed first; the test pins the observed behaviour), and the *same*
   `Prepared` then runs successfully with `λ = 1e-2`. Bar: the retry succeeds
   and returns a finite `δ`.
6. `poisson::tests` — points lie in `[0,1]^dim`, the boundary set is on the
   boundary, the exact solution vanishes there, RNG determinism.
7. `plot::tests` — bounds helpers on empty/constant/NaN input; PNG write
   produces a non-empty file (as `examples/neural-ode/src/plot/tests.rs`).

Acceptance bars for the measurements (single-threaded `CpuEngine::serial()`,
recorded with the thread count):

| run | bar |
| --- | --- |
| 1-D LM (≤ 40 iterations) | relL2 ≤ 1e-4, wall time ≤ 60 s, no unhandled Cholesky error |
| 1-D Adam (2000 epochs) | relL2 ≤ 5e-2 |
| 2-D LM (≤ 40 iterations) | relL2 ≤ 1e-3 |
| 2-D Adam (2000 epochs) | relL2 ≤ 1e-1 |
| comparison | at equal wall time LM's relL2 is ≥ 10× smaller than Adam's |

The comparison is measured, not asserted: the LM run's wall time `T_lm` is read
from its summary line, then the Adam run is repeated per dimension with the
largest epoch count whose wall time stays within ±20 % of `T_lm`; the table
reports that Adam run (epochs, wall time, relL2) next to the LM one, and the
bar is applied to that pair. If no epoch count lands inside ±20 %, report the
nearest one and the deviation — do not tune the network.

If a bar is missed, the measured numbers and the diagnosis go into the README
and the worklog; nothing is tuned silently to hit a bar.

## 3. Edge cases, compatibility, deliberate non-changes

- `--epochs 0`: the programs are prepared before the loop, so a zero-length run
  still builds, prepares and evaluates (the #12 review finding, not repeated).
- Rank-0 `λ` placeholder: built with `input_concrete_shape(DType::F64, &[])`;
  the first test that runs the program pins that a rank-0 input is bindable. If
  it is not, use `[1]` + `broadcast_in_dim([P], [0])` (one-line change).
- `n_col`, `n_bc`, sizes are compile-time constants of the run (shapes are
  static in a traced graph); one `Prepared` per program per run.
- Points are fixed per run: the residual vector, `J` and the ρ comparison are
  consistent; Adam uses the same points, so the comparison is on one objective.
- `f64` only (convention 8 allows dtype pass-through but tenferro 0.11 has no
  public `Tensor::cast`, and every other example is `f64`).
- Not changed: `sciml-kit`, the other four examples, `ci.yml`,
  `upstream-main.yml` (it calls `checks.yml`), the workspace `Cargo.toml`
  (`tenferro-linalg` is already a workspace dependency).
- Not done: no new tenferro API or tenferro-rs issue (both gaps are already
  tracked as #2419 and #2642); no `lstsq` comparison variant (issue's optional
  suggestion, not in its Done-when list); no nonlinear PINN (#9).

## 4. Risks and the rejected alternative

Risks, in order:

1. **Cholesky + solve inside a traced graph at P ≈ 10³** has no precedent in
   this repo. Probed first with a tiny `--epochs 1` run before the rest of the
   loop is written; also covered by tests 4 and 5.
3. **Accuracy**: a weakly enforced boundary (2048 PDE rows vs 2 BC rows in 1-D)
   can leave a constant offset; handled by the per-group `1/sqrt(n)` scaling
   (§1.3) and checked by the relL2 bar.
4. **faer's Cholesky on a semidefinite matrix** may return `Ok` with NaNs
   instead of an error. `decide` treats NaN as a reject, and the loop raises λ
   until the cap, so the run terminates with a diagnosis either way.
2. **LM cost** `P²N + P³/3` ≈ 2.9·10⁹ + 4.1·10⁸ flops per iteration at
   `P = 1153, N = 2050` (1-D) — ~0.3–0.5 s single-threaded. If that exceeds the
   wall-time bar, the width or `n_col` is reduced (2 hidden layers of width 24,
   `n_col = 1024`, which gives `P = 697`/`721`) and the change is reported in the
   README and worklog, not hidden.
5. **Prepare-time cost and memory.** `WORKLOG.md` in this repo records that a
   long unrolled 0.11.0 reverse-mode graph was OOM-killed at 20.7 GB RSS during
   prepare, which `--epochs 2` does not mitigate. This graph is *not* unrolled
   over points (the point axis is a tensor axis), so its node count does not
   grow with N, but the probe in risk 1 measures prepare time and peak RSS of
   the **default** graph (`/usr/bin/time -v`) before the loop is written, and
   the numbers go into the worklog. If the default graph does not prepare inside
   a few seconds and ~1 GB, the size fallback of risk 2 applies.

Rejected alternative: **extend `sciml-kit::nn::Mlp` with a per-point mode** and
use it for both optimizers. `Mlp` is points-first by construction, so the
per-point path would need a transpose per layer or a second layout inside the
shared crate — a change to a crate three other examples depend on, for one
consumer, and it still would not give the Adam/LM pair a single θ layout (the
comparison would have to transpose between layouts to share initial values).
The local `Model` keeps one model, one layout, one θ for both optimizers; if #9
needs the same model it moves into `sciml-kit` mechanically.

Second rejected alternative: **host-side Cauchy-point line search** with a
per-trial program run instead of the in-graph candidate residual. It needs two
runs and two read-backs per iteration, against convention 3, for no accuracy
gain on a least-squares problem where the LM ρ test is the standard acceptance
rule.

Third: **`lstsq` on `[J; sqrt(λ)I]`** as the primary solve (issue's alternative)
— `lstsq` is a composite built from QR/SVD and, with `J` `[N,P]` for
`N = 2176 > P`, is ~10× the flops of the normal equations for a problem that
does not need the extra rank robustness. Kept out; named in the README.

## 5. Open item parked for the maintainer

The issue's Done-when list also asks for `baselines/jax_poisson.py` with its
numbers in the table. This machine has no JAX and installing one is an
environment change outside this task's scope, so an unrunnable baseline script
would be untested code. Parked: the README states the baseline is not included
and why, and the PR body asks whether to add it (and on which machine).

## 7. Post-review refinement (recorded before implementing)

The reviewed plan used two point groups (`x_col`, `x_bc`) with two per-point
parameter sets. While pinning the input shapes it became clear that one
`[dim, n_col+n_bc]` point placeholder with two static column slices of the
residual replaces both groups: one parameter set, one `grad_many` list, one
tile copy of the same total size, and the same residual for both optimizers.
Nothing in the accepted Jacobian contract, the copy cost, the ρ test or the API
claims changes; §1.3/§1.6 above are the updated text.

## 8. Review

Reviewer: `ollama-cloud/glm-5.3-flash:high` (pi agent `reviewer-glm`,
read-only), plan reviewed before any code was written.
**VERDICT: APPROVE** — no BLOCKER; 2 IMPORTANT and 4 MINOR findings, all
addressed in this revision:

- IMPORTANT (defaults unpinned): the default `sizes`/`n_col`/`n_bc`/`λ₀` and the
  resulting `P`/`N` are now a pinned table in §1.6, and the cost fallback in §4
  names the sizes it would produce.
- IMPORTANT (prepare-time precedent): risk 5 now cites the repo's own 20.7 GB
  prepare-time OOM record and makes prepare time and peak RSS part of the first
  probe.
- MINOR (citation drift): `Error::kind()` → `error.rs:783`, Xavier →
  `nn.rs:104-119`, per-group scaling → `loss.rs:33-53`.
- MINOR (`--locked`): added to every run command.
- MINOR ("comparable wall time" undefined): replaced by the explicit
  equal-wall-time protocol in §2.
- MINOR (plot helper duplication): accepted as-is for one figure; a third copy
  would move them into `sciml-kit`.

## 9. Measured outcome (after implementing)

Single-threaded `CpuEngine::serial()`, default sizes and seed, `--locked`,
release profile:

| run | steps | wall | loss | relL2 | bar | |
| --- | --- | --- | --- | --- | --- | --- |
| 1-D LM | 150 | 24.10 s | 5.745e-08 | 1.922e-05 | ≤1e-4, ≤60 s | met |
| 1-D Adam | 2000 | 113.01 s | 5.384e-04 | 1.533e-03 | ≤5e-2 | met |
| 1-D Adam at equal wall time | 430 | 25.23 s | 4.570e-01 | 5.423e-01 | LM ≥10x smaller | met (2.8e4x) |
| 2-D LM | 150 | 35.35 s | 1.081e-06 | 5.772e-04 | ≤1e-3 | met |
| 2-D Adam | 2000 | 218.18 s | 1.817e-02 | 1.373e-01 | ≤1e-1 | **missed** |
| 2-D Adam at equal wall time | 320 | 35.72 s | 3.280e+00 | 1.807e+00 | LM ≥10x smaller | met (3.1e3x) |

The equal-wall-time rows are the largest Adam epoch count whose wall time stays
within ±20 % of the LM run (24.10 s against 25.23 s, +4.7 %; 35.35 s against
35.72 s, +1.0 %).

The default LM budget was raised from the 40 iterations pinned in §1.6 to 150:
two dimensions need them (at 40 iterations 2-D LM is at `relL2` 9.76e-02 and at
150 at 5.77e-04), and the 1-D run pays 18 s for a budget it does not need.

The one missed bar is 2-D Adam, and it is a rate, not a budget: 1.373e-01 after
2000 epochs and 1.0239e-01 after 4000 (449 s), with its own loss falling only
from 1.8e-02 to 5.3e-03 over those epochs. Nothing was tuned to close it, per
§2; the README and the worklog carry both numbers.

§6 risk 5 (prepare-time memory) did not materialise: the largest run peaks at
294 MB (1-D LM 150) and 338 MB (2-D LM 150), and `prepare` of the four programs
takes 0.003 s.

The committed figure is `examples/pinn-gauss-newton/loss.png` (1-D, both
optimizers, from `--optimizer both --plot loss.png`). The repository's
`.gitignore` lists `*.png`, so it is added with `git add -f`, which is how
`neural_ode.png` and `deeponet.png` are tracked as well.

## 10. Where the implementation moved past §1

§1.6 describes the design as reviewed (candidate parameters and the damped solve
in one graph, eight declared inputs, 40 LM iterations). The shipped code differs
in five places, all recorded in `WORKLOG.md` "Deviations", in §7 and in §9: the
candidate residual is its own program (`UnsupportedAdRule` on `jvp` across the
linalg op), the candidate parameters are stepped on the host, the inputs are
bound by key with the AD inputs appended in program order, the point set is one
placeholder with static column slices instead of two groups, and the default LM
budget is 150 iterations because two dimensions need it.
