# WORKLOG — tenferro-examples#4

Port every example and `sciml-kit` to the tenferro 0.12 (`tenferro-rs` `main`)
API. Branch `fix/4-upstream-main-0.12`, worktree
`~/projects/tensor4all/tenferro-examples-4`, base `origin/main` = `ae56631`.
One PR, not merged. Supersedes #5.

Machine: AMD Ryzen AI MAX+ 395 (32 hw threads), rustc 1.100.0, cargo 1.100.0,
kache v2.1.1 (`kache doctor --verify` → 12322/12322 entries valid before work).

## 1. Plan + plan review (before code)

- Plan: `IMPLEMENTATION-PLAN.md` in this worktree.
- Reviewer: `reviewer-glm` = `ollama-cloud/glm-5.3-flash:high`, read-only
  subagent, own pi session.
- Verdict: **CHANGES**, 0 BLOCKER / 1 IMPORTANT / 4 MINOR. All addressed in the
  plan before coding; none required a code change.
  - I1 (accepted, my error): §2 described the CI "Short runs" step as four
    commands and `neural_ode`/`deeponet`/`pinn_gauss_newton` as compile-only. The
    step has **nine** commands and runs all five examples, with three
    `pinn_gauss_newton` lanes. The acceptance table now lists all nine verbatim
    and the numeric baseline covers them. Cause: I had read `checks.yml` from the
    stale `feat/7-sciml-kit-kdv-pinn` checkout instead of `origin/main`.
  - M1: eager ops are documented as taking `&EagerTensor`. In practice the
    parameters are already references, so no call site changed; clippy
    (`needless_borrow`) rejected the extra `&` I first added. §1.1 records this.
  - M2: doc sites — now §1.7's mechanical rule and the file list.
  - M3: §3.1 reworded (a git-source lock is a mismatch error, not a silent lane
    switch).
  - M4: reviewer read `release/v0.12.0-prep` (`origin/main`@`28417df`) because it
    had no shell; this worklog records the tip actually built (§3).

## 2. Baseline, before the change (`origin/main` @ `ae56631`, 0.11.0 pins)

Separate worktree `/tmp/te4-baseline`, `CARGO_TARGET_DIR=/tmp/te4-target-baseline`,
data in `/tmp/te4-baseline-data/`.

- `cargo test --workspace --release` → **111 passed**, 0 failed (suites: 10, 26,
  14, 30, 18, 13, 0).
- The nine CI "Short runs" commands → all exit 0. Reported scalars:

| run | result |
| --- | --- |
| kdv_pinn `--epochs 20` | final loss 5.625707e-1 |
| neural_ode `--epochs 20` | final loss 6.069677e-2 |
| deeponet `--epochs 100` | final loss 1.146277e-2 |
| pinn_gauss_newton `--dim 1 --optimizer lm --epochs 5` | loss 7.416709e0, rel_l2 9.204844e-1, 3 rejections |
| pinn_gauss_newton `--dim 2 --optimizer lm --epochs 3` | loss 4.714285e1, rel_l2 4.076132e0, 2 rejections |
| pinn_gauss_newton `--dim 2 --optimizer adam --epochs 5` | exit 0 |
| nqs_rbm ising-chain | ground E/N -1.2814577239, trained -1.2812628356, fidelity 0.999421388 |
| nqs_rbm heisenberg-chain | ground E/N -1.8255467045, trained -1.8254521369, fidelity 0.999951147 |
| nqs_rbm heisenberg-square | exit 0 |

## 3. Port

`tenferro-rs` `main` tip built against:
`856db9f310578c572b96540d4dd06b71b1038f2f` (`tenferro-*` 0.12.0, resolved
through `branch = "main"`; confirmed from `cargo metadata`).

Applied exactly as `.github/workflows/upstream-main.yml` does:

```sh
sed -i -E 's#^(tenferro-[a-z-]+) = "[^"]+"#\1 = { git = "https://github.com/tensor4all/tenferro-rs", branch = "main" }#' Cargo.toml
```

Changes (the committed manifest keeps registry pins, per `AGENTS.md`):

| file | change |
| --- | --- |
| `Cargo.toml` | five `tenferro-*` pins `0.11.0` → `0.12.0` |
| `examples/nqs-rbm/src/autodiff.rs` | `log_psi.backward(&mut engine)` → `session.backward(&loss)` inside the existing `with_eager_session` closure |
| `examples/nqs-rbm/src/autodiff/tests.rs` | same, in the finite-difference gradient test |
| `examples/nqs-rbm/src/sr.rs` | `TypedTensorSessionOpsExt` → `CoreOps` (`matmul`, `solve` call sites unchanged) |
| `examples/nqs-rbm/src/reference.rs` | `eigh` tuple → `Eigh { values, vectors }` |
| `examples/deeponet/src/data.rs` | `TensorLinalgExt` → `LinalgOps`, module doc 0.12 |
| `examples/pinn-gauss-newton/src/lm.rs` | `TracedTensorLinalgExt` → `TracedLinalgOps`; the two positional `triangular_solve` calls → the plain form (defaults are Left/Lower/None/NonUnit) and `triangular_solve_with(.., TriangularSolveOptions::default().op(MatOp::Transpose))` |
| `README.md`, `examples/{deeponet,nqs-rbm}/README.md` | the doc sites that named a removed trait or a code sample that used `EagerTensor::backward` |

`kdv-pinn`, `neural-ode`, `pinn-gauss-newton`'s other modules and `sciml-kit`
needed **no** source change: they use only traced core ops and
`TracedTensorAdExt`, whose spelling is unchanged. The axis *types* did move
(`usize` → `isize`, negative counts from the end), but every axis argument in
these crates is a literal or `None`, so it infers and no call site moved.

Doc lines deliberately left alone: statements that describe what 0.11 *could
not* do and carry the design reasoning (`kdv-pinn/README.md:245`,
`neural-ode/README.md:72`, `pinn-gauss-newton/README.md:63`,
`deeponet/README.md:120`, `crates/sciml-kit/src/lib.rs:18`). They are accurate
history, not claims about the current build.

## 4. Verification, after the change (upstream-main lane)

`CARGO_TARGET_DIR=/tmp/te4-target`, data in `/tmp/te4-port-data/`.

| check | result |
| --- | --- |
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets --release -- -D warnings` | pass, no warnings, no `#[allow]` added |
| `cargo test --workspace --release` | **111 passed**, 0 failed — same count as the baseline |
| the nine "Short runs" commands, verbatim and in order | all exit 0 |

Numeric regression check against §2, all nine runs: **every reported scalar
identical**, including the eight-decimal energies and fidelities of the nqs runs
and all three `pinn_gauss_newton` summaries (`loss`, `rel_l2`, rejections). The
only differences anywhere are wall-clock strings.

`nqs_rbm --exact --output-dir` artefacts, compared field by field outside the
elapsed column (Python, not `diff`): `history.csv` 101/101 rows,
`validation_energies.csv` 2049/2049 rows, `weights.csv` 4/4 rows identical;
`summary.json` differs only in `elapsed_seconds`.

## 5. What is not green, and why

`release` (`ci.yml`, registry pins, `cargo --locked`) still fails, at dependency
resolution, before compiling:

```
error: failed to select a version for the requirement `tenferro-ad = "^0.12.0"`
candidate versions found which didn't match: 0.11.0, 0.10.0, 0.9.0, ...
```

0.12.0 is not on crates.io (checked today for all five crates through the
registry API: `0.11.0` is the newest). This is the publication blocker recorded
in #5, and it is not fixable from this repository: the source that satisfies
`upstream-main` cannot satisfy `release` until 0.12.0 is published. No CI change
was made to hide it. After publication, `cargo update --workspace` is the only
remaining step.

`Cargo.lock` is deliberately unchanged (still the 0.11.0 lock): the only lock
that could describe the new manifest would record a `git+…?branch=main` source,
which is a requirement/source mismatch against the committed registry pins, so
`--locked` would reject it rather than build. It buys nothing and muddies the
release lane, so it stays out.

## 6. Finished-diff review

`independent-review` (post mode), reviewer `openai/gpt-6.1-sol`, own pi process,
read-only. Verdict: **APPROVE**, 0 BLOCKER / 0 IMPORTANT / 3 MINOR — all three
were wording errors in this worklog and the PR body, corrected in place:
(a) "a git lock would move the release lane onto `main`" is wrong, `--locked`
rejects the mismatch; (b) "axis types unchanged" is wrong, the signatures moved
`usize` → `isize` while the call sites stayed put through literal inference;
(c) the plan claimed the eager two-tensor calls gain `&`, which they do not.
The review additionally confirmed the triangular-solve mapping
(`api/concrete.rs:414-434` maps the options back to exactly
`(true, true, true, false)`), that `EagerTensor::grad()` still reads the slot
`session.backward` writes, and that no removed API call remains in the tree.
Recorded in the PR body as `review: gpt-6.1-sol APPROVE`.

---

# Phase 2: rebase onto main + nonlinear-pinn port

Trigger: scheduled `Upstream main` run `44366785210` (head `88bb29d`,
2026-11-18T18:10:30Z) fails in `checks.yml`; `main` added
`examples/nonlinear-pinn` (`be1e72d`, unported: 0.11 `FftExecutor::rfft/irfft`)
and docs (`88bb29d`); branch is CONFLICTING `behind_by: 2`. Publication blocker
gone: tenferro **0.12.0** is on crates.io for all six crates (registry API
`max_stable_version` = 0.12.0 each, checked 2026-11-19), so the release lane
can resolve `--locked` from the registry.

## P2.1 Plan + plan review (before code)

- Plan: `IMPLEMENTATION-PLAN.md` rewritten for phase 2 (supersedes the phase-1
  text, same file per repo convention).
- Reviewer: `reviewer-glm` = `ollama-cloud/glm-5.3-flash`, read-only subagent,
  own pi session.
- Verdict: **APPROVE**, 0 BLOCKER / 0 IMPORTANT / 3 MINOR, before any code:
  - M1 (accepted): `cargo update -p` only the five *existing* tenferro lock
    entries; `-p tenferro-fft` would match nothing in the 0.11 lock.
  - M2 (resolved): the 0.12.0 release SHA `35bc49d5` verified directly in the
    tenferro-rs checkout (`git log --grep="release: 0.12.0"`), not from a tag
    guess.
  - M3 (accepted): the two `nonlinear_pinn` short-run lines and the README
    table come verbatim from `origin/main`@`88bb29d`, which is where §1.1/§2
    of the plan quoted them from.
  - The reviewer could not read unreferenced git objects with its read-only
    toolset; the §2 build bars falsify those claims at the first cargo step.

## P2.2 Change

- Rebased `d339a30` onto `origin/main`@`88bb29d` → `4385395`; conflicts
  resolved exactly as planned (`Cargo.toml`: five 0.12.0 pins +
  `tenferro-fft = "0.12.0"`; `README.md`: main's attribution table with the
  port's deeponet cell). No other conflict.
- `examples/nonlinear-pinn/src/reference.rs`: `rfft`/`irfft` positional calls →
  `rfft_with`/`irfft_with` with `FftOptions::default().axis(0)` (and
  `.n(Some(self.n))` for `irfft`); imports `FftExecutor, FftOptions`
  (`FftNorm` dropped: `Backward` is the documented `FftOptions` default).
- Root `README.md` nonlinear-pinn row and
  `examples/nonlinear-pinn/README.md:71,170`: `rfft_with`/`irfft_with`
  spelling, "public 0.12.0 APIs". Left untouched (accurate history):
  "rejected as a tangent in 0.11", the `main.rs:180` 0.11 jvp comment,
  `sciml-kit/src/lib.rs:18`.
- `Cargo.lock`: the six `tenferro-*` entries (+ their internal `tenferro-*`
  deps) moved to registry 0.12.0 via `cargo update -p` on exactly those; no
  git source, no wholesale update.

## P2.3 Verification, release lane (`ci.yml`, registry pins, `--locked`)

All from the worktree root, CI order; artifacts in `outputs/`, raw logs in
`/tmp/te4-phase2-data/short-runs.log`:

| check | result |
| --- | --- |
| `cargo fmt --all --check` | pass |
| `cargo clippy --locked --workspace --all-targets --release -- -D warnings` | pass, no warnings, no `#[allow]` added |
| `cargo test --locked --workspace --release` | **124 passed**, 0 failed (phase-1: 111; +13 new nonlinear-pinn/Lbfgs tests) |
| the eleven "Short runs" of `checks.yml`, verbatim, with `--locked` | all exit 0 |

Numeric regression: the nine phase-1 runs report **every scalar identical** to
the phase-1 0.12 records (kdv 5.625707e-1; neural-ode 6.069677e-2; deeponet
1.146277e-2; pinn-gauss-newton 7.416709e0/9.204844e-1/3 rejections,
4.714285e1/4.076132e0/2 rejections, adam exit 0; nqs ising
-1.2814577239/-1.2812628356/fidelity 0.999421388; heisenberg-chain
-1.8255467045/-1.8254521369/0.999951147; heisenberg-square exit 0).

`nonlinear_pinn` (first green 0.12 run; its unit tests check the reference
against exact Cole-Hopf, all passing):

| run | result |
| --- | --- |
| burgers `--adam-steps 20 --lbfgs-steps 10` | adam 4.0725e-1 → 3.3553e-1, l-bfgs → final loss 1.9452e-1, relative L2 5.6796e-1, PNG saved |
| allen-cahn `--optimizer adam --adam-steps 20` | 2.8347e1 → final loss 4.6337e0, relative L2 7.3412e-1, PNG saved |

(One rerun: the kdv gif first went to a wrong filename in my invocation;
repeated verbatim from `checks.yml` — same scalars, both exit 0.)
