# WORKLOG — tenferro-examples#8 (pinn-gauss-newton)

## Model / session

Worker model: `opencode-go/deepseek-v4.1-flash` (pi session). Base: `origin/main`
29db186 ("Add the neural-ode example (checkpointed discrete adjoint) (#15)"),
branch `feat/8-pinn-gauss-newton` (worktree
`~/projects/tensor4all/tenferro-examples-8`). No tenferro changes: the example
uses the released 0.11.0 crates and workspace `sciml-kit`.

## Plan gate

`IMPLEMENTATION-PLAN.md` was written before any code. Reviewer: `reviewer-glm`,
a read-only pi subagent on **`ollama-cloud/glm-5.3-flash:high`**.

**VERDICT: APPROVE** — no BLOCKER; 2 IMPORTANT and 4 MINOR findings, all
addressed in the plan before implementing (§8 there): the default sizes and the
resulting `P`/`N` are pinned as a table, the prepare-time memory risk cites the
repo's own 20.7 GB precedent and makes prepare time and peak RSS part of the
first probe, three citation references were corrected, `--locked` was added to
every run command, the vague "comparable wall time" became the explicit
equal-wall-time protocol, and the duplicated plot-bounds helper is accepted for
one figure.

## What was implemented

`examples/pinn-gauss-newton` (package `pinn_gauss_newton`):

- `src/network.rs` — tanh MLP `[dim, 32, 32, 1]` in two layouts: plain
  `[out, in]`/`[out]` (Adam's parameters) and per-point `[out, in, N]`/`[out, N]`
  (the batched Jacobian contract), with `tile` as the host copy between them.
  The layer is broadcast-plus-reduce over the input features, not a batched
  `dot_general` (0.11.0 mis-indexes the left-hand batch slice; reproducer:
  `x [in, N]`, `W [out, in, N]` returns the slice of point `out` for every
  point).
- `src/poisson.rs` — fixed point set (2048 collocation, 2/128 boundary), forcing
  term, residual `[(-Laplace u - f)/sqrt(n_col); u/sqrt(n_bc)]`, and
  `jacobian_transposed`, which is `grad(sum(r), theta_b)` reshaped to `[P, N]`.
- `src/lm.rs` — `solve_step` in the graph (`(J^T J + lambda I) delta = -J^T r`
  through `cholesky` and two triangular solves, plus the linear-model predicted
  reduction), and the host decision `accept`/`next_lambda` on rho.
- `src/plot.rs`, `src/main.rs` — loss curves and the CLI.

Nothing outside the example changed except the CI short runs and one row in the
root `README.md`.

## Deviations from the reviewed plan

1. **The two point groups became one.** Pinning the input shapes showed that one
   `[dim, n_col+n_bc]` placeholder with two static column slices replaces the
   separate collocation and boundary placeholders and their per-point parameter
   sets: same residual, same total copy size, one `grad_many` list. Recorded in
   the plan (§7) before implementing.
2. **The candidate residual is its own program.** A second differentiated
   residual in the same trace as the damped solve fails with
   `UnsupportedAdRule { transform: "jvp", op: "tenferro-linalg.linalg.v1" }`; one
   `TraceContext` per program (Adam, LM step, LM candidate, evaluation) avoids
   it.
3. **Inputs are bound by key, not position.** `tenferro-ad` appends one program
   input per `jvp` seed and per reverse-pass cotangent, so a program has more
   inputs than the caller declares. `Program::reads` matches the declared ones
   through `CompiledGraph::input_keys()` in both directions and fills the rest
   from the caller's list in program order: `poisson::seed_inputs` (both `jvp`
   calls of coordinate 0, then of coordinate 1, then the cotangent) is shared by
   `main.rs` and by the finite-difference tests so the two cannot drift. The
   compiler may leave a declared placeholder out of a program's input list (the
   last bias in a small hand-built graph), so the binding does not require every
   handle to appear; `residual_matches_finite_differences_2d` pins that the
   trained residual equals the whole network's, with the biases shifted away from
   zero so a dropped bias could not hide.
4. **`stepped()` runs on the host**, not in the graph, so the candidate
   parameters never enter a trace.
5. **The plot has no text**: plotters here is built without a font backend and
   drawing text panics, so the axes are unlabelled and the shapes carry the
   message.

## Measurements

Single-threaded `CpuEngine::serial()`, default sizes and seed, `--locked`,
release profile, AMD Ryzen AI MAX+ 395 (one thread), rustc 1.100.0:

| run | steps | wall | loss | relL2 |
| --- | --- | --- | --- | --- |
| 1-D LM | 150 | 24.10 s | 5.745e-08 | 1.922e-05 |
| 1-D Adam | 2000 | 113.01 s | 5.384e-04 | 1.533e-03 |
| 1-D Adam at equal wall time | 430 | 25.23 s | 4.570e-01 | 5.423e-01 |
| 2-D LM | 150 | 35.35 s | 1.081e-06 | 5.772e-04 |
| 2-D Adam | 2000 | 218.18 s | 1.817e-02 | 1.373e-01 |
| 2-D Adam at equal wall time | 320 | 35.72 s | 3.280e+00 | 1.807e+00 |

Bars: 1-D LM (`≤1e-4`, `≤60 s`), 1-D Adam (`≤5e-2`), 2-D LM (`≤1e-3`) and the
equal-wall-time comparison in both dimensions (`LM ≥10x smaller`: 2.8e4x and
3.1e3x) are met. **The 2-D Adam bar (`≤1e-1`) is missed**: 1.373e-01 after 2000
epochs (218 s), and 1.0239e-01 after 4000 epochs (449 s), still short. Adam's
own loss falls only from 1.8e-2 to 5.3e-3 over those extra 2000 epochs, so the
gap is the optimizer's rate at a step-decayed `1e-3` learning rate, not budget
alone; the README reports both numbers and no parameter was tuned to close it.

Peak RSS 294 MB (1-D LM 150) and 338 MB (2-D LM 150); `prepare` of the four
programs 0.003 s, so the plan's prepare-time memory risk did not materialise.

## Verification (final state)

```
cargo fmt --all --check                                        # clean
cargo clippy --locked --workspace --all-targets --release -- -D warnings
                                                               # No issues found
cargo test --locked --workspace --release                      # 111 passed
cargo run --locked --release -p pinn_gauss_newton -- --dim 1 --optimizer lm --epochs 5 \
  --plot /tmp/outputs/pinn.png                                 # 0.89 s, PNG written
cargo run --locked --release -p pinn_gauss_newton -- --dim 2 --optimizer adam --epochs 5
                                                               # 0.66 s
```

Tests in the example: 18, including the batched Jacobian against per-parameter
JVPs, the residual *of the trained program* against finite differences in 2-D
(boundary rows included, plus the step program's reported loss against the
residual's sum of squares), and the reverse-mode loss gradient against a
directional finite difference.

## Post-implementation review

`reviewer-glm` on the diff against `origin/main`: **VERDICT: CHANGES**.

- **BLOCKER — the 2-D runs trained on the wrong operator.** `main.rs` built the
  Laplace seed list interleaved (`[s0, s1, s0, s1]`) while `tenferro-ad` appends
  one input per `jvp` call in trace order, grouped per coordinate, so two of the
  four 2-D slots were bound to the wrong coordinate and the Laplace operator
  became mixed derivatives. Every 2-D number in the first version of this log
  was therefore an artifact — it explained the "boundary-offset minimum" the
  first version diagnosed at length. Fixed by one shared helper,
  `poisson::seed_inputs`, used by `main.rs` and the tests; with the correct order
  2-D LM goes from `relL2` 1.43 to 5.77e-04 and 2-D Adam from 1.69 to 1.37e-01,
  and all 2-D measurements above were re-run. `residual_matches_finite_differences_2d`
  was verified to fail with a permuted order (1.6e-3 disagreement), so the
  suite now catches this class instead of only an example run.
- IMPORTANT (no check that a declared handle reaches the graph): the binding is
  now documented as one-directional, and the full-size finite-difference test
  shifts the biases away from zero so a dropped parameter cannot hide; it passes
  with `n_col = 2048` and `[2, 32, 32, 1]`, i.e. the trained program's residual
  equals the whole network's.
- IMPORTANT (two statements of the AD input order, only one true): fixed with
  the shared helper above; the README and module docs now describe the grouped
  order that the code uses.
- MINOR: the per-step copy wording now says an LM iteration copies twice; the
  `--epochs` documentation says it applies to both optimizers; `--dim 2
  --optimizer lm` was added to the CI short runs; `IMPLEMENTATION-PLAN.md` gained
  a note where §1.6 describes the pre-implementation design; and `--epochs 0` is
  rejected instead of reporting a `NaN` loss.
- Second pass on the corrected diff: **VERDICT: APPROVE**, no BLOCKER and no
  IMPORTANT. Three MINOR fixed: the `Seeds` doc said `dim` AD inputs where the
  count is `2 * dim` (`poisson.rs`), the README now names the epoch counts behind
  the 449 s Adam number, and plan §10 lists the 40 -> 150 iteration default with
  the other four drifts.

After the fixes: fmt clean, clippy clean (workspace, all targets), 111 tests
pass, and every measurement in the table above comes from the fixed binary.

- Follow-up round, adversarial pass on `60a3efd` after the PR was opened:
  **NEEDS_CHANGES**, one finding. `lm/tests.rs::step_solves_the_damped_system`
  built its Jacobian from `sin(0.7 i)`, whose Gram matrix is definite only to
  within an ulp: the reviewer measured an eigenvalue of `-8.2e-16` and Cholesky
  failed at `lambda = 0` in their checkout, so the undamped branch was never
  verified there. Test defect, not a runtime bug (the training loop starts at
  `lambda = 1e-3`). Fixed by an explicit integer Jacobian — ten rows of six
  integers, the first six upper triangular with the diagonal 2..7, so the columns
  are independent and `J^T J` is definite by construction (Cholesky factor
  diagonal 2.6..6.7) — an integer right-hand side, so nothing in the test depends
  on the platform's `sin`/`cos`, and an assertion that the undamped step is
  finite and solves the system to `1e-10` relative. The old matrix fails a plain
  Cholesky in high-precision arithmetic as well; the new one does not.
  `cargo test --release -p pinn_gauss_newton --locked` → 18 passed, workspace
  111 passed, and the three CI short runs reproduce.
  The review's remaining items are unchanged and none is a defect: the measured
  table is not reproduced by the reviewer, the upstream-`main` lane cannot be
  read from there, and CI asserts no quality bar by design (it runs 3-5 step
  smoke runs, so neither the equal-wall-time rows nor the missed 2-D Adam bar are
  visible to it).
