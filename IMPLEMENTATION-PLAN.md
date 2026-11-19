# IMPLEMENTATION-PLAN — tenferro-examples#4, phase 2: rebase + nonlinear-pinn

Make both CI lanes of PR #18 green: rebase `fix/4-upstream-main-0.12` onto
`origin/main` (keeping main's `nonlinear-pinn` and docs) and port
`examples/nonlinear-pinn` to the tenferro 0.12 API.

- Issue: <https://github.com/tensor4all/tenferro-examples/issues/4>
- PR (reused, **not merged**): <https://github.com/tensor4all/tenferro-examples/pull/18>
- Worktree: `~/projects/tensor4all/tenferro-examples-4`, branch
  `fix/4-upstream-main-0.12`.
- Branch head `d339a30` = the 0.12 port of the four examples + `sciml-kit`
  (phase 1, verified against tenferro-rs `856db9f3`).
- `origin/main` `88bb29d` = `be1e72d` (adds `examples/nonlinear-pinn`, the
  `Lbfgs` optimizer in `crates/sciml-kit`, the `tenferro-fft = "0.11.0"`
  workspace pin and two `nonlinear_pinn` short runs in `checks.yml`) plus
  `88bb29d` (README attribution table). Branch is `behind_by: 2`, conflicting.

---

## 0. Facts established by reading, not by compiling

- The failing scheduled run `44366785210` (head `88bb29d`) reports exactly the
  phase-1 error set (already fixed by `d339a30`: `TypedTensorSessionOpsExt`,
  `FftExecutor::rfft/irfft` in **nonlinear-pinn only**, `EagerTensor::backward`,
  `matmul`, `cholesky`, one E0308, `TracedTensorLinalgExt`, `TensorLinalgExt`)
  — the only *new* code to port is `examples/nonlinear-pinn/src/reference.rs`.
- crates.io now has **0.12.0** for all six crates (`tenferro-tensor`,
  `-runtime`, `-cpu`, `-fft`, `-ad`, `-linalg`), verified through the registry
  API (`max_stable_version` = 0.12.0 each). The phase-1 publication blocker is
  gone; the release lane can resolve `--locked` from the registry.
- 0.12 FFT API (identical at the 0.12.0 release commit `35bc49d5` and current
  tenferro-rs `main` `d48982d3`):
  - `FftExecutor::rfft_with(&mut self, input: &Tensor, opts: &FftOptions, exec: &mut dyn Engine) -> Result<Tensor>`; same shape for `irfft_with`.
  - `FftOptions { n: Option<usize> (default `None`), axis: isize (default `-1`), norm: FftNorm (default `Backward`) }` with same-named setters.
  - The 0.11 positional `FftExecutor::rfft/irfft(&input, n, axis, norm, exec)`
    are gone — that is the CI error at `reference.rs:73,86`.
- The rest of `nonlinear-pinn` (`main.rs`, `problem.rs`, `plot.rs`) and the new
  `sciml-kit::optim::Lbfgs` use traced core ops whose 0.12 spelling is
  unchanged (phase 1 needed no source change in `kdv-pinn`/`neural-ode`/
  `sciml-kit`); the scheduled clippy run lists no other `nonlinear-pinn` site.
- Files changed on *both* sides since the merge base `ae56631`: `Cargo.toml`,
  `README.md` — conflicts are expected exactly there.
- tenferro-rs `main` advanced `856db9f3 → d48982d3` (7 commits: docs, CI
  classification, a tensor slice-view fix, AD test pins, an FFT-AD fix);
  `rfft_with`/`irfft_with` signatures are unchanged.

## 1. Files and functions changed, and the current behaviour

### 1.1 Rebase `d339a30` onto `88bb29d`

- `Cargo.toml` — branch pins the five crates at `"0.12.0"` (no `tenferro-fft`
  key); main added `tenferro-fft = "0.11.0"`. Resolution: the five 0.12.0 pins
  **plus** `tenferro-fft = "0.12.0"`. Current behaviour otherwise unchanged:
  registry pins only, the git dependency stays owned by `upstream-main.yml`'s
  sed.
- `README.md` — main rewrote the example table (adds the `Implemented by`
  column and the `nonlinear-pinn` row, based on the *old* deeponet cell);
  the port fixed the same deeponet cell to `tenferro-linalg` `cholesky`.
  Resolution: main's table with the port's deeponet cell, and the
  `nonlinear-pinn` row's `(`rfft`/`irfft` with a reused `FftExecutor`)` cell
  renamed to `rfft_with`/`irfft_with` (it names the calls the code makes).

### 1.2 `examples/nonlinear-pinn/src/reference.rs` — the port

Current behaviour: `Solver::forward` calls
`self.fft.rfft(&u, None, 0, FftNorm::Backward, &mut self.engine)` and
`Solver::inverse` calls
`self.fft.irfft(&spectrum, Some(self.n), 0, FftNorm::Backward, &mut self.engine)`
— the removed 0.11 positional form. New code, same semantics (same `n`, same
axis, same norm — 0 is the last axis of a 1-D tensor — and still through the
executor, so FFT plans stay reused):

```rust
use tenferro_fft::{FftExecutor, FftOptions};
// forward:
self.fft.rfft_with(&u, &FftOptions::default().axis(0), &mut self.engine)
// inverse:
self.fft.irfft_with(&spectrum, &FftOptions::default().axis(0).n(Some(self.n)), &mut self.engine)
```

`FftNorm` leaves the imports (the `Backward` default it named is now the
documented `FftOptions` default).

### 1.3 `examples/nonlinear-pinn/README.md` — docs of the ported calls

Same mechanical rule as phase 1: change a line iff it (a) names a 0.11-only
API spelling or (b) states which tenferro version the current code builds
against. That reaches:

- `:71` `` `rfft`/`irfft` through one `FftExecutor` `` → `rfft_with`/`irfft_with`;
- `:169` "The example uses only public 0.11.0 APIs." → "0.12.0".

**Not touched** (accurate history, per the phase-1 rule): `:173` "rejected as
a tangent in 0.11", `src/main.rs:180`'s "A jvp tangent must carry data in
tenferro 0.11" comment (the workaround code itself stays — it compiles under
0.12), and `crates/sciml-kit/src/lib.rs:18` "tenferro 0.11 has no public
`Tensor::cast`" (0.12 has none either; verified by grep at `35bc49d5`).

### 1.4 `Cargo.lock`

Regenerate so the six `tenferro-*` entries resolve to registry `0.12.0`:
`cargo update -p` the **five existing entries** (`tenferro-ad`, `-cpu`,
`-linalg`, `-runtime`, `-tensor`; `-p tenferro-fft` matches nothing in the
current lock and would fail) — the resolution itself adds the new
`tenferro-fft` and `nonlinear_pinn` entries. No wholesale `cargo update`.
The committed lock must contain no git source; this is what makes the
release lane's `--locked` pass.

### 1.5 `IMPLEMENTATION-PLAN.md`, `WORKLOG.md`

This rewrite (repo convention: one tracked plan per issue, superseding the
phase-1 text) and the phase-2 worklog with the review verdict.

**Deliberately not changed**: `.github/workflows/*` (§0 forbids CI edits),
every test, every other example's source, `sciml-kit` code, the jvp
constant-leaf workaround in `nonlinear-pinn/src/main.rs`.

## 2. Exact commands and acceptance bars

From the worktree root, CI order, both lanes exactly as
`.github/workflows/checks.yml` runs them.

**Release lane** (`tenferro: release`, so `LOCKED=--locked`):

| # | command | acceptance bar |
| --- | --- | --- |
| 1 | `cargo fmt --all --check` | exit 0 |
| 2 | `cargo clippy --locked --workspace --all-targets --release -- -D warnings` | exit 0, no `#[allow]` added |
| 3 | `cargo test --locked --workspace --release` | exit 0; all tests pass (phase-1 count 111 plus `nonlinear-pinn`'s and `Lbfgs`'s new tests); count recorded in the worklog; no test deleted, `#[ignore]`d or loosened |
| 4 | the **eleven** "Short runs" commands of `checks.yml` verbatim, each with `--locked`: the nine phase-1 commands plus `cargo run --locked --release -p nonlinear_pinn -- --problem burgers --adam-steps 20 --lbfgs-steps 10 --plot outputs/nonlinear-pinn-burgers.png` and `cargo run --locked --release -p nonlinear_pinn -- --problem allen-cahn --optimizer adam --adam-steps 20 --plot outputs/nonlinear-pinn-allen-cahn.png` | exit 0 each; printed losses recorded in the worklog; both PNGs produced |

**Upstream-main lane**: apply the workflow's sed exactly,

```sh
sed -i -E 's#^(tenferro-[a-z-]+) = "[^"]+"#\1 = { git = "https://github.com/tensor4all/tenferro-rs", branch = "main" }#' Cargo.toml
grep '^tenferro-' Cargo.toml     # must show six git deps now, incl. tenferro-fft
```

then the same four steps **without** `--locked`. Same bars. Afterwards restore
`Cargo.toml` and `Cargo.lock` (`git checkout -- Cargo.toml Cargo.lock`) so the
committed state carries registry pins only.

Regression bar: the nine phase-1 smoke commands' final numbers must match the
phase-1 records (those were identical to the 0.11 baseline); a difference beyond
f64 round-off (>1e-9 relative) is a blocking finding to explain, not to
accept. `nonlinear_pinn` has no prior green 0.12 run: its bar is its unit
tests (the reference is checked against exact Cole-Hopf inside them) plus both
smoke runs exiting 0 with plausible losses, recorded in the worklog.

## 3. Edge cases, compatibility, deliberate non-changes

- `axis`: `FftOptions` defaults to `-1`; the calls keep explicit `.axis(0)` to
  mirror the 0.11 call sites. Identical transform for a 1-D tensor.
- `irfft` length: `n(Some(self.n))` preserved (the real output length).
- The upstream lane's sed must not reach the commit: `git checkout --` both
  files after that lane.
- If a lane fails for registry propagation (outside this repo), wait and
  retry; never hide it in code (§0).
- One finished push to the PR branch only; no merge, no auto-merge, no other
  branch (§0). The rebase force-push rewrites only the PR's own branch.
- GPU: this repo has no CUDA lane; no local GPU evidence is required.

## 4. Risks, and the alternatives rejected

- Risk: tenferro-rs `main` (`d48982d3`, 14 commits past 0.12.0) changed
  numerics visible in the examples → the full upstream lane is run locally
  once, on the finished state, before the single push.
- Risk: release-lane `--locked` rejects a stale lock → regenerate the lock from
  the registry pins first and commit it; CI then resolves without updating.
- Risk: rebase conflicts hide a semantic merge (e.g. main's table text vs the
  port's cell fix) → both conflict sites are listed in §1.1 with the resolved
  text; anything unexpected stops the rebase and goes back to the reviewer.
- Rejected: merging `origin/main` into the branch — the task says rebase, and a
  merge commit would present the port and the unported state tangled.
- Rejected: `FftOps::rfft` on the tensors (`u.rfft(&mut engine)`) — drops the
  `FftExecutor` plan reuse the README documents; `rfft_with` keeps it.
- Rejected: pinning git deps in the committed manifest — CI keeps registry
  pins and `upstream-main.yml` owns the sed (`AGENTS.md`).

## 5. Review

Plan-gate reviewer (mandatory before any code):

- agent: `reviewer-glm`, model GLM-5.3-flash (`ollama-cloud/glm-5.3-flash`),
  read-only (own pi session, `read/grep/find/ls` only), must not edit files.
- Verdict and findings recorded below and in `WORKLOG.md`.
- Post-implementation, the same reviewer model reads the finished diff before
  the push (R-ci-pr-11), verdict recorded in `WORKLOG.md`.

REVIEW: **VERDICT: APPROVE**, 0 BLOCKER / 0 IMPORTANT / 3 MINOR (reviewer:
`reviewer-glm`, model GLM-5.3-flash via `ollama-cloud/glm-5.3-flash`,
read-only, own pi session). The reviewer confirmed the branch/lock/CI-file
state from the worktree and the 0.11 positional `rfft/irfft` signature from
the tenferro-rs checkout; it could not read the unreferenced git objects
(`88bb29d`, `35bc49d5`, `d48982d3`) with its read-only toolset — the §2 build
bars falsify those claims at the first cargo step. Findings and resolution:

- M1 (accepted): §1.4 now says `cargo update -p` on the five *existing*
  entries only; the lock has no `tenferro-fft` block yet, so `-p
  tenferro-fft` would not match — the resolution adds it.
- M2 (resolved, no plan change): the release SHA `35bc49d5` was verified
  directly in the tenferro-rs checkout (`git log --grep="release: 0.12.0"`);
  recorded here and in the worklog.
- M3 (accepted): the two `nonlinear_pinn` short-run lines and the README
table are taken verbatim from `origin/main`@`88bb29d` at rebase time (§1.1/§2
quote them from the actual files).
