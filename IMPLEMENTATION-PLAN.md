# IMPLEMENTATION-PLAN — tenferro-examples#4

Port **every** example and `sciml-kit` from the tenferro 0.11 API to the 0.12
(`tenferro-rs` `main`) API, so the daily `Upstream main` run compiles and runs
again.

- Issue: <https://github.com/tensor4all/tenferro-examples/issues/4>
- Failing run named in the task: `44145060758`; the run the issue body links:
  `43926225130`.
- Worktree: `~/projects/tensor4all/tenferro-examples-4`, branch
  `fix/4-upstream-main-0.12`, based on `origin/main` = `ae56631` (contains all
  four examples: #13 `sciml-kit`+kdv-pinn, #14 deeponet, #15 neural-ode,
  #16 pinn-gauss-newton).
- Supersedes / closes #5 (`fix/upstream-main-api-drift`, draft, DIRTY, nqs-rbm
  only).
- One PR, **not merged**.

---

## 0. Facts established by reading, not by compiling

`tenferro-rs` `origin/main` is `0.12.0` (`Cargo.toml:40`,
`release: 0.12.0`). crates.io has **only 0.11.0** for all five crates
(`tenferro-ad`, `-cpu`, `-linalg`, `-runtime`, `-tensor`); verified through the
registry API for each name. So `main` is unreleased and 0.12 cannot be resolved
from the index — the `release` CI job cannot go green from source alone. That is
a publication blocker, not a code problem, and it is stated in the PR.

The 0.12 API is defined by `docs/getting-started/api-migration.md` and
`docs/spec/generated/api-ops.md` in `tenferro-rs` main. The changes that reach
these crates:

| 0.11 (current code) | 0.12 |
| --- | --- |
| `tenferro_linalg::TracedTensorLinalgExt` | `tenferro_linalg::TracedLinalgOps` |
| `tenferro_linalg::TensorLinalgExt` | `tenferro_linalg::LinalgOps` |
| `tenferro_runtime::TypedTensorSessionOpsExt` | `tenferro_runtime::CoreOps` |
| `EagerTensor::backward(&mut engine)` | `EagerSession::backward(&loss)` inside `with_eager_session` |
| `x.triangular_solve(&b, left, lower, trans, unit)` | `x.triangular_solve_with(&b, &TriangularSolveOptions::default().side(..).uplo(..).op(..).diag(..))` |
| `let (values, vectors) = m.eigh(..)` | `let Eigh { values, vectors } = m.eigh(..)` |
| axes `usize` | axes `isize` (negative counts from the end); shapes stay `usize` |

`cholesky`, `solve`, `matmul`, `eigh`, `dot_general`, `reduce_sum`,
`broadcast_in_dim`, `reshape`, `concatenate`, `slice` keep their names;
`cholesky` and `solve` keep the plain (no-options) form, so those call sites do
not change beyond the trait import.

---

## 1. Files and functions changed, and the current behaviour

`Cargo.toml` — `[workspace.dependencies]` pins `tenferro-* = "0.11.0"`. Change
the five pins to `"0.12.0"`. Nothing else in the manifest changes: the examples
stay on version pins, never on a git or path dependency, per `AGENTS.md`.

### 1.1 `examples/nqs-rbm` (already ported in #5 — carried over, re-verified)

- `src/autodiff.rs::tied_gradient` — builds `log_psi` inside
  `with_eager_session`, then calls `log_psi.backward(&mut engine)` outside it.
  Move the backward call into the closure
  (`session.backward(&loss).map(|_| ())`); `w.grad()?` stays. `Tensor` is still
  re-exported by `tenferro-ad`, so the import list is unchanged. The eager
  two-tensor calls in `rbm_log_psi` (`session.matmul(w, spins)`,
  `session.add(&projection, b)`, `session.mul(a, spins)`) keep their spelling:
  the operands are already `&EagerTensor` parameters, exactly what 0.12 takes,
  and clippy's `needless_borrow` rejects adding a `&`.
- `src/autodiff/tests.rs` — same shape in the test.
- `src/sr.rs` — `use tenferro_runtime::TypedTensorSessionOpsExt` becomes
  `CoreOps`; `matmul`/`solve` call sites are textually unchanged
  (`tenferro_linalg::prelude::*` already provides `LinalgOps`).
- `src/reference.rs::exact_ground_state` — `eigh` returns a struct:
  `let Eigh { values, vectors } = ...`. `jacobian.solve(&rhs, &mut engine)`
  unchanged.
- `README.md` — the two sentences that name `TypedTensorSessionOpsExt` /
  `EagerTensor::backward` / `eigh` tuple, and the 0.11 pin.

### 1.2 `examples/deeponet`

- `src/data.rs` — `use tenferro_linalg::TensorLinalgExt` → `LinalgOps`
  (`cholesky_factor`, `matrix.cholesky(engine)` unchanged); module doc says
  "from tenferro-linalg 0.11.0", update to 0.12.
- `src/{main,model,plot}.rs` — traced `dot_general`, `reduce_sum`,
  `broadcast_in_dim`, `reshape`, `add`: unchanged unless the compiler reports a
  signature change. `src/main.rs:142` `reduce_sum(Some(&[0, 1]))` may need
  `isize` literals, which integer-literal inference handles.

### 1.3 `examples/kdv-pinn`

No 0.11-only trait. Traced core ops only. Expected: zero or literal-inference
axis fixes across `src/{pde,loss,main,sampler,plot}.rs` and their `tests.rs`.

### 1.4 `examples/neural-ode`

Same shape as kdv-pinn: `TracedTensorAdExt` + traced core ops (`train.rs`,
`ode.rs`, `model.rs`, `data.rs`). Expected: no or literal-only changes.

### 1.5 `examples/pinn-gauss-newton`

- `src/lm.rs::solve_step` — `use tenferro_linalg::TracedTensorLinalgExt` →
  `TracedLinalgOps`; `transpose(&[1, 0])` (isize; literal inference);
  `normal.add(&identity)?.cholesky()?` unchanged; the two
  `triangular_solve(&x, a, b, c, d)` calls become `triangular_solve_with`:
  - `(left=true, lower=true, trans=false, unit=false)` →
    `side(Side::Left).uplo(Uplo::Lower).op(MatOp::None).diag(Diag::NonUnit)`
    — which equals the plain `triangular_solve(&gradient)`, so this one can drop
    the options entirely;
  - `(left=true, lower=true, trans=true, unit=false)` →
    `...op(MatOp::Transpose)`.
  Semantics must stay bit-identical: the same LAPACK-family call, the same side,
  the same triangle, one transpose.
- `src/main.rs`, `src/poisson.rs`, `src/network.rs` — `grad_many`,
  `concatenate`, `slice`, `broadcast_in_dim`, `reduce_sum`, `reshape`; change
  only what the compiler demands.
- `src/lm/tests.rs` — `solve_step` signature is unchanged, so only the tuple
  destructuring / trait imports move.

### 1.6 `crates/sciml-kit`

`src/{nn,optim,func}.rs` — `dot_general` with `DotGeneralConfig`,
`input_concrete_shape`, `reshape`, `add`, `TracedTensor` placeholders. Expected:
no change, or literal-only axis fixes. Module doc mentions "tenferro 0.11"; if
it does, update the version.

### 1.7 Docs

One mechanical rule: change a doc line iff it (a) names a 0.11-only API that no
longer exists, or (b) states which tenferro version the current code builds
against. Sentences that describe what 0.11 *could not* do stay — they are
accurate history and carry the design reasoning.

That reaches: `README.md:10,12` (root table: `TensorLinalgExt::cholesky`,
`backward`, `matmul`, `eigh`), `examples/nqs-rbm/README.md:233` (a
`log_psi.backward(&mut engine)?` code sample), `examples/deeponet/README.md:63-64
and :179-182` (`TensorLinalgExt::cholesky`, "only public 0.11.0 APIs"),
`examples/deeponet/src/data.rs:6` (module doc, same). `examples/{kdv-pinn,
neural-ode,pinn-gauss-newton}/README.md` and `crates/sciml-kit/src/lib.rs:18`
only describe 0.11 *limitations*, so they are not touched.

---

## 2. Exact commands and acceptance bars

All from the worktree root, with the upstream-main variant applied exactly as
`.github/workflows/upstream-main.yml` applies it, so nothing is local-only:

```sh
sed -i -E 's#^(tenferro-[a-z-]+) = "[^"]+"#\1 = { git = "https://github.com/tensor4all/tenferro-rs", branch = "main" }#' Cargo.toml
grep '^tenferro-' Cargo.toml     # must show five git deps
```

Then, in this order (CI order):

| # | command | acceptance bar |
| --- | --- | --- |
| 1 | `cargo fmt --all --check` | exit 0, no diff |
| 2 | `cargo clippy --workspace --all-targets --release -- -D warnings` | exit 0, zero warnings, no `#[allow]` added to reach it |
| 3 | `cargo test --workspace --release` | exit 0; **the same number of tests, all passing** as the pre-change baseline; no test deleted, `#[ignore]`d or loosened |
| 4 | the **nine** `cargo run --release` commands of `checks.yml` "Short runs", verbatim and in order — `kdv_pinn --epochs 20 --loss-png .. --gif ..`; `neural_ode --epochs 20 --plot ..`; `deeponet --epochs 100 --plot ..`; `pinn_gauss_newton` in three lanes (`--dim 1 --optimizer lm --epochs 5 --plot ..`, `--dim 2 --optimizer lm --epochs 3`, `--dim 2 --optimizer adam --epochs 5`); `nqs_rbm` ising-chain (`--size 8 --steps 100 --samples 256 --validation-sweeps 256 --exact` + output dir/pngs), heisenberg-chain (same, no output dir), heisenberg-square (`--size 4 --alpha 2 --steps 20 --samples 128 --validation-sweeps 128 --filters-png ..`) | exit 0 for each of the nine. Every example is run by CI, so every example is run here; no example is compile-only |

Regression bar, captured before the port from `origin/main` with the 0.11 pins
(saved under `/tmp/tenferro-examples-4/baseline/`):

- each of the nine smoke commands' **final loss / energy and any
  `summary.json` scalar** is compared against the 0.11 run of the same command;
  a difference beyond f64 round-off in the reported digits (>1e-9 relative) is a
  blocking finding to explain, not to accept silently;
- the run durations are recorded, not asserted (loaded machines).

`cargo test` counts and the smoke outputs are the only two acceptance numbers
that can regress; both are recorded in the worklog.

Everything is run **without** `--locked`: the port targets a git branch, and the
committed `Cargo.lock` stays at 0.11.0 on purpose (§4).

---

## 3. Edge cases, compatibility, deliberate non-changes

1. **`Cargo.lock` is not regenerated or committed.** 0.12 is not on crates.io,
   so the only lock that could describe the new manifest is a
   `git+https://github.com/tensor4all/tenferro-rs?branch=main` lock. With the
   committed manifest demanding `"0.12.0"` from the registry, that lock is a
   source/requirement mismatch, so release `--locked` errors instead of building
   — it does not silently switch lanes, but it also cannot help. The lock
   therefore stays 0.11.0 and `release` fails at resolution, which is the
   publication blocker already recorded in #5. After 0.12.0 is published, one
   `cargo update --workspace` finishes the PR. (`--locked` exists only in the
   release lane; the `main` lane updates the working lock in place and commits
   nothing, so the two-lane design is preserved either way.)
2. **No CI edits.** `checks.yml` / `upstream-main.yml` are untouched; no gate is
   weakened, skipped or made non-blocking to hide the `release` failure.
3. **No `#[allow(...)]`, no silencing.** If clippy fires, the code changes.
4. **`0.11`-compatible source is not kept alongside.** The two APIs are mutually
   exclusive in the same tree (`0.11` has no `CoreOps`, no
   `EagerSession::backward`, no `Eigh` struct). A `cfg`/version-shim layer would
   be a second code path for one concern, against the workspace rule; the
   release lane is unblocked by publishing, not by a shim.
5. **Deliberately unchanged:** numerical algorithms, tolerances, test
   expectations, CLI flags, plot code, the `sciml-kit` public surface, and the
   examples' structure. The port is an API translation: no refactor, no
   behaviour change, no new dependency.
6. **`isize` axes.** Only where a value is not a literal (a `usize` loop
   variable used as an axis) does the type need changing; `&[0, 1]` literals
   infer `isize` and stay as written. No `as isize` casts in hot paths beyond
   what a `usize` variable already needed.
7. `#5`'s content is carried over rather than rebased: its four nqs-rbm files are
   re-proven here against the same API, and the PR says so and closes it.

---

## 4. Risks, and the alternative rejected

| risk | mitigation |
| --- | --- |
| 0.12's change is not just renames — `#2552` is a whole new call grammar; further drift (e.g. `Runtime::prepare`, `ExecutionEngine`, `DotGeneralConfig`) may be hidden behind the four files the issue names | port is compiler-driven over the whole workspace (`--all-targets`), not a hand-list of six renames; the plan's file list is a prediction, and any extra file the compiler reports is added to the worklog with the same rule: minimal translation, no behaviour change |
| a green clippy/test but a changed number in the examples | the smoke commands are run and their reported scalars diffed against the 0.11 baseline |
| tenferro-rs `main` moves during the work, so a fixed-by-morning build breaks by evening | the PR is verified against `main`'s tip at run time and the tip SHA recorded in the worklog; a moving target is reported, not chased |
| the git dependency drags a huge tree and makes local gates slow | kache (`kache doctor --verify` clean) plus one shared `CARGO_TARGET_DIR` for all gates; no concurrent cargo on the same target dir |
| `release` stays red and a reviewer reads that as a broken PR | the PR body states it as the publication blocker with the exact resolution error, and the PR stays draft/unmerged |

Alternative rejected: **keep `Cargo.toml` at 0.11.0 and only add a `[patch]`
section**, so the release lane is unaffected. Rejected because (a) 0.12 source
cannot compile against 0.11 crates at all, so the release lane breaks either way,
(b) a `[patch]` on `tenferro-rs` would redirect the *release* lane to git as
well, which is worse than failing, and (c) it contradicts `AGENTS.md`'s "no git
or path dependency on tenferro-rs" in the committed manifest.

Second alternative rejected: **a version-gated shim crate** (one path for 0.11,
one for 0.12). Two code paths for one concern, doubling the examples' surface for
a window that closes on publication; see §3.4.

---

## 5. Review

Plan-gate reviewer (mandatory before any code):

- agent: `reviewer-glm`, model `ollama-cloud/glm-5.3-flash:high`, read-only
  (own pi session, `read/grep/find/ls` only).
- Verdict and findings: recorded below and in `WORKLOG.md` §1.

REVIEW: **VERDICT: CHANGES**, 0 BLOCKER / 1 IMPORTANT / 4 MINOR. All API
replacements in §0-§1 were confirmed against the 0.12 source by the reviewer.
Findings and their resolution:

- I1 (IMPORTANT, accepted): §2 described the "Short runs" step as four commands
  and called `neural_ode`/`deeponet`/`pinn_gauss_newton` compile-only. The step
  has **nine** commands and runs all five examples, including three
  `pinn_gauss_newton` lanes. The table above now lists all nine verbatim and the
  regression baseline covers them. (Cause of my error: an earlier read of
  `checks.yml` from the stale `feat/7-sciml-kit-kdv-pinn` branch instead of
  `origin/main`.)
- M1 (accepted): §1.1 now records that eager ops take `&EagerTensor` operands.
- M2 (accepted): §1.7 now lists every doc site, with the mechanical rule for
  which "0.11" sentences change and which stay.
- M3 (accepted): §3.1 reworded — a git-source lock is a mismatch error, not a
  silent lane switch.
- M4 (accepted, no action): the reviewer read `release/v0.12.0-prep`
  (= `origin/main`@`28417df`) because this session had no shell; §4's
  tip-SHA rule already requires re-verification against `main`'s tip, which is
  what the build does.
