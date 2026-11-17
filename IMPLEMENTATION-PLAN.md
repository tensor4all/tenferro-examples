# IMPLEMENTATION-PLAN — tenferro-examples#12 (`neural-ode`)

Issue: <https://github.com/tensor4all/tenferro-examples/issues/12> (part of the
SciML umbrella #6, after #7/#10).
Repo: `~/projects/tensor4all/tenferro-examples`, worktree
`~/projects/tensor4all/tenferro-examples-12`, branch `feat/12-neural-ode`,
base `origin/main` = c946ff9 (adds #7 `sciml-kit` and #10 `deeponet`).
Deliverable: one PR, no merge, no tenferro change.

## 0. APIs used, verified in the pinned 0.11.0 sources

Registry root `~/.cargo/registry/src/index.crates.io-*/`; the workspace
`Cargo.lock` pins every `tenferro-*` crate to 0.11.0.

| API | Evidence |
| --- | --- |
| `TracedTensor::mul` / `sub` / `add` | `tenferro-runtime-0.11.0/src/traced.rs:815`, `:…` |
| `TracedTensor::abs` (AD rule: `sign(primal)`) | `traced.rs:1076`; `tenferro-ad-0.11.0/src/semantic_transform.rs:555` (jvp), `:734` (vjp) |
| `TracedTensor::tanh`, `scale_real`, `reduce_sum(None)` | `traced.rs:1281`, `:1118`, `:1672` |
| `TracedTensor::stack(&[&Self], isize)` (inserts a new axis) | `tenferro-runtime-0.11.0/src/shape_packing.rs:589` |
| `TraceContext::input_concrete_shape(DType, &[usize])` | `tenferro-runtime-0.11.0/src/trace_ops.rs:245`; used by `deeponet/src/main.rs:100` |
| `TracedTensorAdExt::value_and_grad(&[&TracedTensor])` | `tenferro-ad-0.11.0/src/traced.rs:946` (trait), `:1116` (impl) |
| `GraphCompiler::compile_many(&[&TracedTensor])` | `tenferro-runtime-0.11.0/src/graph/compiler.rs:192` |
| `Runtime::prepare` / `run`, `Output::into_tensor`, `TensorRead::from_tensor` | `runtime/snapshot.rs:979`, `:1020`, `runtime/engine.rs:304` |
| `Tensor::from_vec_col_major`, `as_slice` / `as_slice_mut` | used by `kdv-pinn/src/pde.rs`, `sciml-kit/src/optim.rs` |

Program input order: `compile_many` takes the trace creation order (no explicit
order for several outputs, tenferro-rs#2642), data-carrying constants created
by AD are program inputs but keep their bound value, so the explicit reads are
the placeholders in creation order. This is exactly the pattern `deeponet`
ships (`deeponet/src/main.rs`, "Parameters are created first..."), so no
`bind_order` helper is needed here: parameters first, then `z0`, then the
window targets, then the AD seeds.

## 1. Design

### 1.1 Problem and data (host, no traced ops)

`dz/dt = A z³` elementwise cube, `A = [[-0.1, 2.0], [-2.0, -0.1]]`,
`z(0) = [2, 0]`, sampled at `N_SAMPLES = 1000` equally spaced times on `[0, 25]`.

- `H = 25 / 999` is the solver step, i.e. a training window of
  `WINDOW_STEPS = 10` RK4 steps spans `≈0.25` time units.
- Reference trajectory: host RK4 with `SUBSTEPS = 100` per sample interval
  (`h_ref = H/100 ≈ 2.5e-4`), stored column-major as `[N_SAMPLES, 2]`
  (`states[i + N_SAMPLES * c]`). No new dependency: a 20-line host loop.
  The convergence check in §3/§4 shows `h_ref` and `2*h_ref` agree far below
  the training error.
- `minibatch(starts) -> (Tensor [B, 2] z0, Tensor [B, 2, WINDOW_STEPS] targets)`.
  `targets[b, c, k]` is the reference state at `start[b] + k + 1`; the layout is
  `b + B*c + B*2*k` column-major. Includes `k = 1..=10`, not the given `z0`
  (a zero term would only rescale the loss).

### 1.2 Model, solver, loss (traced)

- `model::rhs(&Mlp, z)`: `z³ = z.mul(z)?.mul(z)?`, then `net.forward(&z³)`.
  `f_θ` is `Mlp::new(&trace, &[2, 50, 2], TracedTensor::tanh)` from
  `sciml-kit`; its 4 placeholders (W1, b1, W2, b2) are the parameter order.
  `Linear::forward` is `x.dot_general(W, contract x's last axis with W's
  first)`, i.e. `out[b,j] = Σ_i x[b,i] W[i,j]` (`sciml-kit/src/nn.rs:39-46`),
  so the column-vector form is `Wᵀ x`: a weight placeholder filled with `W`
  realizes `A z³` when `W = Aᵀ`. The model test in §3 does exactly that.
- `ode::rollout(rhs, z0, h, steps) -> Vec<TracedTensor>`: fixed-step RK4,
  one state after each step. `ode::rk4_step` is the textbook four-stage update,
  generic over `Fn(&TracedTensor) -> Result<TracedTensor>` so it can move to
  `tenferro-diffeq` later (issue requirement).
- Loss: `pred = stack(states, axis 2)` → `[B, 2, 10]`;
  `loss = mean(|pred - targets|)` = `diff.abs()?.reduce_sum(None)? / (B*2*10)`.
- One reverse pass: `loss.value_and_grad(&net.placeholders())` → loss + 4 grads,
  then `compile_many(&[loss, grads...])` → **one** program, prepared **once**
  (shapes are constant), run once per epoch.

### 1.3 Training loop and CLI

- `--epochs N` default `20_000`, `--seed K` default `42` (repo convention),
  `--plot PATH` optional.
- Per epoch: `BATCH = 64` random window starts in `0..N_SAMPLES - WINDOW_STEPS`
  (`0..990`, so `start + 10 ≤ 999`), bind params + `z0` + `targets`, run,
  `Adam` step. Learning rate `1e-3` through `sciml_kit::optim::step_decay_lr`
  (base for the first half, half for the third quarter, quarter for the last
  quarter — the kdv schedule moved verbatim).
- Print: seed/config, data generation time, graph build + compile time,
  loss every 500 epochs, final loss, full-trajectory relative L2 error, training
  wall time; then plot if requested.

### 1.4 Evaluation (full 999-step trajectory)

A forward-only program is built in the same trace after training:
`z_eval = input_concrete_shape(F64, &[1, 2])`, `rollout(..., WINDOW_STEPS)`,
`compile_many` of its 10 outputs. Prepare once with `params + z_eval`, then chain
the same prepared program 100 times on the host (10 RK4 steps = 10 samples per
run), feeding the last output back as the next `z_eval`. 100 windows cover 1000
steps; the trajectory has 999 intervals, so the last output (t = 25 + H) is
dropped and the first 999 states are compared with the reference. The number of
runs is `(N_SAMPLES - 1).div_ceil(WINDOW_STEPS) = 100`, and `main` asserts that
the collected buffer holds exactly `N_SAMPLES` states before computing the
metric, so a wrong feed-back index or a dropped state fails loudly there.

No 999-step graph is compiled: the compiled unit stays the 10-step window, as
the issue asks ("unrolled inside one traced graph per window length").

Metric: `relative L2 = sqrt(Σ |z_pred - z_true|² / Σ |z_true|²)` over both
components and all 1000 samples. Printed and reported in the README.

### 1.5 Figure

`plot::write_figure_png(path, times, truth, pred, loss_history)`: one 1400×420
image, three panels — z₁(t), z₂(t) (blue = reference, red = prediction) and the
loss (red, log y). No text (`plotters` has no font backend in this workspace,
same as the other examples), stated in the README. The PNG is committed only
after the full 20k run (`neural_ode.png`), as `deeponet` does.

## 2. Files and functions changed

New example `examples/neural-ode` (package `neural_ode`):

| File | Contents |
| --- | --- |
| `Cargo.toml` | deps `sciml-kit` and `tenferro-{ad,cpu,runtime,tensor}` from `[workspace.dependencies]`, plus `rand = "0.8"` and `plotters` (bitmap backend/encoder, line series) declared per package like the other examples; all already in the lockfile |
| `src/main.rs` | flags, ground-truth generation, graph build, training loop, evaluation, figure. `#[cfg(test)] run_single` helper copied from `kdv-pinn/src/main.rs:123` (its tests need a one-shot prepare+run) |
| `src/model.rs` | `MLP_SIZES`, `rhs(net, z)` |
| `src/ode.rs` | `rk4_step`, `rollout` |
| `src/data.rs` | `N_SAMPLES`, `WINDOW_STEPS`, `H`, `SUBSTEPS`, `velocity`, `Dataset::{generate, minibatch}` |
| `src/plot.rs` | `linear_bounds`, `log_bounds`, `write_figure_png` |
| `README.md` | problem, model, data, training, results/cost table, figure, code layout, tenferro notes, stretch status |
| `.gitignore` | `*.png` with `!neural_ode.png` (same as `deeponet`) |
| `src/*/tests.rs` | see §3 |

Shared change (now two examples use it):
`crates/sciml-kit/src/optim.rs` gains `pub fn step_decay_lr(epoch, total, base)`
and its test (moved from `examples/kdv-pinn/src/schedule.rs` +
`schedule/tests.rs`; the body and test are verbatim, the stale module doc
("no other example uses this shape yet") is rewritten to say the schedule now
lives here); `examples/kdv-pinn/src/main.rs` imports it from
`sciml_kit::optim`, and `examples/kdv-pinn/src/schedule.rs` /
`src/schedule/tests.rs` are deleted. kdv's README structure table drops its
`schedule.rs` row and names the schedule on the `sciml-kit/src/optim.rs` row.
This is the kit's own rule ("holds model and optimizer code that two or more
examples use"); behaviour of kdv is unchanged.

Root/CI:
- `Cargo.lock`: new `neural_ode` package entry (only existing deps).
- `README.md`: one table row.
- `.github/workflows/checks.yml` (shared by `ci.yml` and `upstream-main.yml`):
  `cargo run $LOCKED --release -p neural_ode -- --epochs 20 --plot outputs/neural-ode.png`
  in the `Short runs` step.

## 3. Tests and measurement commands; acceptance bars

Unit tests (no training):
- `ode`: RK4 on `z' = -z`, 10 steps of `h = 0.1` vs `e^{-1}` within 1e-6
  (compiled with `run_single`); `rollout` returns one state per step.
- `model`: `rhs` of a `[2, 2]` (linear, no hidden activation) `Mlp` whose
  column-major weight buffer holds `Aᵀ` equals `A z³` for a known `z` (1e-12):
  `forward` computes `Wᵀ z³`, so this pins "cube, then network" and that the
  weight is the transpose of the column-vector matrix. The example's
  `[2, 50, 2]` net has 4 placeholders and matching initial values.
- `data`: `velocity([1,0]) = [-0.1, -2]` and `velocity([0,1]) = [2, -0.1]`;
  `generate` is deterministic and `SUBSTEPS = 50` vs `100` agree to < 1e-9
  relative; `minibatch` places `z0` and every target at the documented index
  (hand-computed for two starts, including the last valid start 989 whose last
  target is sample 999), and `#[should_panic]` on an out-of-range start.
- `plot`: bounds padding/empty/constant, and a PNG of non-zero size is written
  to a temp dir (temp dir removed).
- `sciml-kit`: the moved `step_decay_lr_has_three_stages` test.

Commands (all from the worktree, kache verified healthy: 11961/11961 entries):

```
cargo fmt --all --check
cargo clippy -j 16 --workspace --all-targets --release -- -D warnings
cargo test -j 16 --workspace --release
cargo run -j 16 --release -p neural_ode -- --epochs 20 --plot /tmp/n12/short.png
cargo run -j 16 --release -p kdv_pinn -- --epochs 20 --loss-png /tmp/n12/kdv.png
```

Measurement (serial CPU engine, `--release`, no plot unless stated):

```
time ./target/release/neural_ode --epochs 0                 # compile cost
time ./target/release/neural_ode --epochs 100
time ./target/release/neural_ode --epochs 20000             # seeds 42, 43, 44
./target/release/neural_ode --epochs 20 --seed 42           # twice: identical final loss
./target/release/neural_ode --epochs 20 --seed 43           # differs
```

Acceptance bars:

1. fmt, clippy (`-D warnings`), tests: clean; short runs exit 0 and write the
   PNGs.
2. Training uses exactly one `value_and_grad` program with one `prepare` and one
   `run` per epoch: exactly one `value_and_grad(` call in `src/main.rs` (outside
   the tests), and the `for epoch` body contains no `compile`/`prepare` —
   checked by reading the loop, with `grep -n "prepare(\|compile_many"
   src/main.rs` showing only the setup calls.
3. Compile (graph build + `compile_many` of loss + 4 gradients, 10 unrolled RK4
   steps = 40 MLP evaluations) ≤ 5 s; per-epoch ≤ 15 ms; full 20k run ≤ 300 s.
4. Full run, seed 42: trajectory relative L2 ≤ 5% (JAX prototype of the same
   algorithm: 1.7% at 20k × batch 64; 43 → 7.9%, 44 → 47%). README reports the
   three values, not only the default. If the Rust run is materially worse than
   the prototype, re-check the loss/optimizer wiring before changing defaults.
5. Same-seed runs print the identical final loss; a different seed differs.
6. kdv's short run still prints `final loss after 20 epochs:` and the same
   behaviour (only the import moved).

## 4. Edge cases, compatibility, deliberate non-changes

- Window bounds: `start` in `0..990`, `minibatch` asserts
  `start + WINDOW_STEPS < N_SAMPLES`, so the last target is `t[999]`.
- `--epochs 0`: the training program is still compiled, the loop is empty, the
  loss stays `inf`; the evaluation still runs (the reference is independent of
  training). No division by zero in the printed numbers.
- `z³` stays in `[-6.5, 8]` on the reference trajectory (measured on the host),
  so no overflow or `tanh` blow-up; `A z³` has components up to 16.
- Eval chaining requires identical shapes each run ([1, 2]); the prepared
  program is reused only with the same shape, dtype and layout.
- AD input order: parameters first, then `z0`, then `targets`; AD seeds are
  created last and are bound constants. No hand-maintained order table, no
  `bind_order` dependence on the `#[doc(hidden)]` `input_key_index`.
- Deliberately unchanged: `sciml-kit` model code (`Mlp`, Xavier init), the
  other examples except kdv's schedule import, CI structure, tenferro versions
  (no git/path dependency), the umbrella's `local_tangent` conventions.
- **Continuous adjoint stretch: out of this PR**, as the issue schedules it as
  a separate PR. It is not approximated or half-implemented here; the README
  says so and names the host-loop/`vjp`-per-step shape it will take.

## 5. Risks and rejected alternatives

- **Risk: long-horizon error is seed-sensitive.** The training objective is
  local (10 steps), while the reported metric is a 999-step rollout, so small
  systematic window biases accumulate (JAX prototype: 1.7% / 7.9% / 47% for
  seeds 42/43/44 at 20k). Mitigation: report the range and the metric in the
  README, do not tune the metric or the window length. The default run (42)
  fits; the figure uses it.
- **Risk: JAX and tenferro optimizers differ in floating point.** Bars 4/5
  compare against the prototype with tolerance, not equality; the Rust numbers
  are what the README reports.
- **Risk: compile cost of reverse mode over 40 unrolled MLP evaluations.**
  Measured first (bar 3); expected far below 5 s given kdv's 5 ms fused program.
  No fallback changes the design: if it exceeded the bar, the compile cost would
  be reported and the graph kept as is (the issue asks for the unrolled graph).
- **Risk: moving `step_decay_lr` touches kdv.** Mechanical move; kdv tests and
  its short run are re-run (bar 6).
- Rejected: **continuous adjoint in this PR** (separate PR per the issue; each
  step's cotangent would have to be re-bound as a data-carrying leaf, i.e. one
  fresh trace+compile per step, so its compile cost belongs in that PR's
  report).
- Rejected: a `NeuralOde` wrapper struct — `Mlp` plus a free `rhs` is the whole
  model, and the test needs a smaller linear `Mlp` anyway.
- Rejected: 10 per-step target placeholders — one `[B,2,10]` tensor keeps the
  binder and the program inputs minimal; its layout is tested by hand.
- Rejected: compiling the full 999-step rollout as one graph for evaluation —
  it would make the compiled unit 100x larger than the training window and
  changes nothing about the reported error. The host chaining keeps one small
  program prepared once.
- Rejected: MSE or a normalized loss — the issue specifies mean absolute error.
- Rejected: copying `step_decay_lr` into the new example — the kit exists for
  code two examples use.

## 6. Review record

Reviewer: `reviewer-glm` = `ollama-cloud/glm-5.3-flash:high` (read-only pi
subagent), reviewed this plan before any code was written.

**VERDICT: CHANGES** — addressed as follows.

- IMPORTANT 1 (`Linear::forward` computes `Wᵀ x`, so a weight filled with `A`
  gives `Aᵀ z³`; the §1.2 model claim and the `[2,2]` test as written could not
  pass): **accepted and fixed**. §1.2 now states the convention with
  `sciml-kit/src/nn.rs:39-46`, and the model test fills the column-major buffer
  with `Aᵀ`.
- MINOR 1 (`rand`/`plotters` are per-package, not workspace deps): accepted,
  §2 wording fixed.
- MINOR 2 (schedule is half / third quarter / last quarter, not quarters):
  accepted, §1.3 fixed.
- MINOR 3 (the moved doc comment would be stale): accepted, §2 now says the
  module doc is rewritten.
- MINOR 4 (eval chaining only covered by the full-run bar; the grep is
  brittle): accepted. `main` now asserts the collected state count before the
  metric, the last-valid-window mapping is a `data` test, and bar 2 is scoped to
  the loop body instead of a bare `grep -c`.

No design change followed from the review: all findings were wording or test
coverage. The reviewer's APPROVE items (all §0 API claims, input order, eval
program inputs, `Prepared` reuse, index formulas, kdv move, CI step) were each
re-checked against the 0.11.0 sources in this session.

## 7. Measurement record: the unrolled window does not compile (blocker)

The implementation below meets the code/API decisions, and all 12 unit tests
pass, but acceptance bar 3 (compile <= 5 s) and bar 4 (a full run) cannot be
reached: `Runtime::prepare` on the 10-step window blows up inside
tenferro-runtime 0.11.0. This section records what was measured, because the
design as written in §1 is not implementable with the released compiler.

### 7.1 Sweep (same binary, only `WINDOW_STEPS` changed, `--epochs 1`)

| window | graph ops | result |
|---|---|---|
| 1 | 350 | 15.3 ms/epoch *including* prepare; trajectory written |
| 2 | 542 | prepare 20.6 s (whole run 20.8 s) |
| 3 | 1091 | OOM-killed after 92 s, max RSS 20,723,608 kB (SIGKILL 9) |
| 10 | 2747 | no completion within 100 s; no PNG; exit 2 |

Commands (from the worktree root, release profile):

```
cargo build -j 16 --release -p neural_ode
./target/release/neural_ode --epochs 1          # window = N
/usr/bin/time -v timeout 240 ./target/release/neural_ode --epochs 1
```

The forward-only evaluation program for the same 10 steps prepares in 5 ms and
runs 100 windows in 0.08 s, so the cost is specific to the *AD* graph, not the
window length as such.

### 7.2 Where the cost is

`callgrind` on the window-2 run (46.9e9 Ir): 98.7% of the instructions are in
`Runtime::prepare` -> `stage_semantic_program` ->
`compile_std_to_exec_with_effects`; the inclusive cost of `DimExpr::clone` is
23.9e9 Ir and `Vec<DimExpr>::clone` alone 14.0e9 Ir, all driven by
`infer_output_shapes` and `same_or_scalar_broadcast_shape`.

A patched *copy* of tenferro-runtime 0.11.0 (in `/tmp/n12/tr`, never committed,
used only for diagnosis) showed the expression shape. The compiler keeps two
shape streams per slot: the executable one from concrete input shapes, and a
parallel "analysis" one that `build_exec_staging` seeds with
`InputDim { input_idx, axis }` for every program input
(`tenferro-runtime-0.11.0/src/compiler/semantic_staging.rs:47`). The AD graph
bakes concrete shapes into `BroadcastInDim`/`Reshape` (the `one_like` anchor of
`tanh`, the cotangent broadcast of `reduce_sum`, and `Linear`'s
`bias.reshape([1, out])`). `broadcast_dim` cannot prove a `Const` and an
`InputDim` equal, so it emits `Max` and clones both subtrees:

```
Max(Max(InputDim { input_idx: 4, axis: 0 },
        Max(... Max(InputDim { input_idx: 4, axis: 0 }, Const(64)), ...)),
    Max(...))
```

Every sequential op adds another such layer, and because the two operands are
distinct grown expressions the tree roughly doubles per RK4 step (939 nodes at
window 2). The analysis stream is marked "best-effort" in the source
(`compiler/mod.rs:296`) but is computed unconditionally, and it OOMs.

Upstream `main` (local checkout `~/projects/tensor4all/tenferro-rs`, fa541ceb)
still contains the same unconditional block, so no newer public release in
reach of this worktree behaves differently.

### 7.3 Workarounds tested and rejected

- Symbolic inputs (`TraceContext::input_symbolic_shape`) instead of concrete
  ones: fails at graph build, because `TracedTensor::scale_real` (used by the
  RK4 combination and the loss normalisation) calls `concrete_shape`; error
  `symbolic dimension in shape hint for tensor at rank 2`.
- Batch 1 (so the concrete annotation is `Const(1)` and folds): still OOM
  (killed after 70 s) — the feature axis `Const(50)` vs `InputDim` doubles the
  same way.
- `CompilerOptions { algebraic_layout_simplifier: false, dot_decomposer: false
  }`: unchanged (>120 s at window 10); the cost is in shape staging, before
  the optimizer.
- Removing `stack` (per-step scalar loss): unchanged.
- Replacing MAE with squared error: unchanged.
- Removing `tanh` (identity activation): unchanged.

### 7.4 Consequences and the two design options

Contract bars 3-6 in §3 (compile <= 5 s, per-epoch <= 15 ms, 20k-epoch run,
seed-42 accuracy) were therefore not measured; only the unit tests and the
1-window run were. No PR is opened: the example's own default run OOMs.

Options for a follow-up decision (neither implemented):

1. **Per-step checkpointed VJP** (one program, one prepare, same 10-step
   window and batch): trace *one* RK4 step with the incoming state and the
   incoming cotangent as inputs/constants and a scalar `sum(lambda * z_next)`
   as the loss, then run it 10 times forward and 10 times backward per epoch,
   accumulating parameter gradients on the host. A 1-step graph prepares in
   15 ms (§7.1), so all compile bars pass; this is the discrete adjoint by
   checkpointing, not the continuous-adjoint stretch of the issue.
2. **Upstream fix**: report the analysis-shape blow-up to tenferro and keep
   this branch unpublished until a release makes the unrolled 10-step window
   preparable; then §1 stands unchanged.

## 8. Approved design change: checkpointed discrete adjoint

The unrolled window cannot be prepared (§7), so the user approved replacing the
single unrolled graph with a per-step checkpointed reverse pass. Batch 64, the
10-step window, the MAE loss, the MLP `[2, 50, 2]`, the RK4 stepper, the
optimizer and the evaluation program stay as planned. Only the training graph
changes: one traced RK4-step graph, run 10 times forward and 10 times backward
per epoch, with the cotangent re-bound as an input at every step.

### 8.1 Mathematics

Window `z_0 -> z_1 -> ... -> z_10`, loss `L = 1/(B*2*10) * sum_k |z_k - t_k|`.
Reverse pass with a checkpointed state and an explicit cotangent:

- carry `lambda = 0` before the first reverse step;
- for `k = 10 .. 1`: `lambda += (1/(B*2*10)) * sign(z_k - t_k)` (the direct
  loss term, added exactly once), then run the step VJP with
  `(z_{k-1}, lambda)` to get `lambda_prev = dz_k/dz_{k-1}^T lambda` (the next
  carried value) and add `dz_k/dtheta^T lambda` to the parameter gradient.

Summing the 10 step VJPs is exactly the discrete adjoint of the unrolled loss
(same arithmetic as §1's unrolled graph, up to floating-point summation order),
so the §3 accuracy bars carry over.

### 8.2 Graph and program

One traced graph, `TraceContext`:

| input | shape | role |
|---|---|---|
| 4 parameters | `[2,50]`, `[50]`, `[50,2]`, `[2]` | trained |
| `z_prev` | `[64,2]` | checkpointed state |
| `lambda` | `[64,2]` | cotangent (not differentiated) |

`z_next = rk4_step(rhs, z_prev, H)` (the §1.1 stepper, generic over RHS),
`spot = reduce_sum(lambda * z_next)`, and `value_and_grad` over the 4
parameters **and** `z_prev`; the `z_prev` gradient is `lambda_prev`. Outputs of
`compile_many`: `z_next` plus the 5 gradients (4 parameters + `z_prev`).
`lambda` is a plain program input that is never passed to `value_and_grad`.

One `prepare` before the first epoch; the same `Prepared` runs every step.
Epoch flow: 10 forward runs with `lambda = 0` (collect `z_1..z_10`), then
10 backward runs. The MAE cotangents are computed on the host (control), the
arithmetic stays in the graph.

### 8.3 Files and functions

- `src/train.rs` (new): build the step graph; `window_loss_and_grad`, the
  forward/backward loop returning the host loss, the accumulated gradient
  vectors and the checkpointed states.
- `src/main.rs`: build the training and evaluation graphs; epoch body calls
  `train::window_loss_and_grad`; fix the evaluation buffer to column-major
  (the current loop appends interleaved states while `Dataset::states` and the
  plot are column-major); keep Adam/`step_decay_lr`, logging, timing prints and
  the `--epochs/--seed/--plot` flags.
- `src/train/tests.rs` (new): a finite-difference check of the real multistep
  reverse loop over all 4 parameter tensors, and a zero-residual check
  (`targets = predicted states` must give loss 0 and an all-zero gradient).
- `src/ode.rs`: unchanged (`rk4_step`/`rollout`).
- `src/data.rs`, `src/model.rs`, `src/plot.rs`, tests: unchanged.
- `README.md` (example + root row): describe the checkpointed discrete adjoint
  and report the measured compile and per-iteration times.
- `WORKLOG.md`, this plan: record the change and the measurements.

### 8.4 Acceptance bars (unchanged numbers, new run counts)

1. `cargo fmt --all --check`, workspace clippy `-D warnings`, workspace tests.
2. One traced training graph, one `value_and_grad`, one `compile_many`, one
   `prepare`; no trace/compile/prepare inside the epoch loop; 20 runs/epoch.
3. Graph build + `compile_many` + the first `prepare` together <= 5 s (the
   staging cost lives in `prepare`, so it is part of the compile budget);
   <= 15 ms/epoch; full 20k-epoch run <= 300 s.
4. Seed 42 full run relL2 <= 5 %; README reports seeds 42/43/44.
5. Same seed reproduces the final loss exactly; kdv short run unchanged.

### 8.5 Risks

- 20 program runs per epoch is 10x the launch count of the unrolled design;
  the bars in 8.4 keep it honest, and §7's window-1 measurement (15 ms
  including prepare) leaves headroom.
- The host computes `sign(z_k - t_k)` in f64 from the same states the graph
  produced, so the adjoint is exact for this loss, not an approximation. The
  finite-difference test in `src/train/tests.rs` pins it.
- The floating-point summation order differs from a single unrolled reverse
  pass; the seed-42/43/44 bars are the check.
- No tenferro change; the diagnosis and the upstream report are separate (§7).

### 8.6 Measured results (final)

Timing, AMD Ryzen AI MAX+ 395, single-threaded `CpuEngine`, seed 42:

| quantity | value |
| --- | --- |
| graph build + `compile_many` | 0.001 s |
| first `prepare` | 0.008 s |
| training, 20 000 epochs | 151.3 s (7.57 ms/epoch) |
| runs/epoch | 20 (10 forward, 10 reverse) |

Bars 1, 2, 3, 5 passed (see WORKLOG for the commands). Bar 4 **did not pass
for seed 42**: the window loss reaches its floor (`4.0e-4`) but the pointwise
relative L2 of the chained 1000-step rollout is 31 %. Measured over six seeds
with everything else fixed:

| seed | final loss | relL2 |
| --- | --- | --- |
| 42 | 4.0e-4 | 31 % |
| 43 | 2.6e-4 | 15 % |
| 44 | 3.2e-4 | 46 % |
| 45 | 5.7e-4 | 3.8 % |
| 46 | 3.3e-4 | 17 % |
| 47 | 6.6e-4 | 15 % |

Diagnosis: the per-window MAE is at its floor for every seed (a trained model
predicts a reference-started 10-step window to 6e-5..3e-2 depending on the
window; the mean over random windows is 4e-4), but the eval feeds the model's
own state back in, so the long-horizon error is a phase drift whose sign and
size depend on the field the optimizer lands in. 50 000 epochs with seed 42
give 31.6 % with the same loss floor, so this is a property of the 10-step
objective, not undertraining. The prototype's 1.7 % (§2) was one draw of the
same heavy-tailed distribution, not a reproducible target; the bar was set
from it and is not met by the default seed. The README now reports all six
seeds and the drift; `--seed 45` shows the 3.8 % trajectory. No seed is
selectively promoted to the default.

Upstream report filed as tensor4all/tenferro-rs#2643 (analysis-shape DimExpr
blow-up; the unrolled graph would return once that is fixed).

Figure note: the loss trace was plotted as block means over 50 epochs; a
per-epoch trace of a random minibatch is not readable.

Post-implementation review (`gpt-6.1-sol`, read-only, diff vs `origin/main`):
**VERDICT: CHANGES**, all findings addressed before commit.

- IMPORTANT: `prepare` was called from inside the epoch loop behind an
  `Option` guard (so `--epochs 0` never prepared it). Hoisted before the loop,
  binding a fixed `[64, 2]` batch of first windows so the window RNG is not
  advanced; the `Option` is gone.
- MINOR: the README overstated the window loss as bounding every window (it is
  the last sampled minibatch); the continuous-adjoint note claimed a
  constant-leaf re-trace that contradicts the checkpointed program (the
  cotangent is a program input), so the claim was deleted; the finite-difference
  test comment called the check exact although the window is nonlinear; the
  model doc claimed the output vanishes cubically although the biases are
  trainable; the layout table said two-panel for a three-panel figure; the plot
  doc said one loss value per epoch although the caller passes block means.
  All six corrected.

