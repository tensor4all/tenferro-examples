# WORKLOG — tenferro-examples#12 (neural-ode)

## Model / session

Worker model: `opencode-go/deepseek-v4.1-flash` (pi session). Base: `origin/main`
c946ff9 ("Add the deeponet ... (#10) (#14)"). Branch: `feat/12-neural-ode`
(worktree `~/projects/tensor4all/tenferro-examples-12`). No tenferro changes;
deps are the released 0.11.0 crates and workspace `sciml-kit`.

## Plan gate

`IMPLEMENTATION-PLAN.md` was written before any code. Reviewer: `reviewer-glm`
= **`ollama-cloud/glm-5.3-flash:high`**, read-only pi subagent.

**VERDICT: CHANGES** — 1 IMPORTANT (`Linear::forward` computes `Wᵀ x`, so the
model test must fill the weight with `Aᵀ`) and 4 MINOR (per-package deps,
schedule wording, moved module doc, evaluation assertions). All accepted and
fixed in the plan (§6). No design change followed.

## First implementation (unrolled window) and blocker

New example `examples/neural-ode` (package `neural_ode`) with `src/data.rs`
(reference trajectory `dz/dt = A z³`, `A = [[-0.1, 2.0], [-2.0, -0.1]]`,
`z(0) = [2, 0]`, 1000 samples on `[0, 25]`, `H = 25/999`, 100 RK4 substeps),
`src/ode.rs` (`rk4_step` generic over the RHS closure, `rollout`),
`src/model.rs` (`[2, 50, 2]` tanh MLP applied to `z³`), `src/plot.rs`,
`src/main.rs`, tests and CI/README rows. Shared change: `step_decay_lr` moved
from `examples/kdv-pinn/src/schedule.rs` to `crates/sciml-kit/src/optim.rs`.

The specified design — one unrolled 10-step RK4 graph through
`value_and_grad` — could not be prepared by tenferro-runtime 0.11.0: the AD
graph's parallel analysis-shape stream builds `Max` trees that roughly double
per sequential op. Measured on the pristine registry build: window 1 15.3 ms
including prepare, window 2 20.6 s prepare, window 3 OOM-killed at 20.7 GB RSS
(92 s), window 10 (the default) no completion in 100 s. Callgrind, the
patched-copy diagnosis and the rejected workarounds are in
`IMPLEMENTATION-PLAN.md` §7. The forward-only evaluation program with the same
10 steps prepares in ~5 ms, so the blow-up is specific to reverse-mode staging.

Commit 577fb8c holds that state; it was never pushed.

## Approved design change: checkpointed discrete adjoint

The user chose the checkpointed reverse pass (plan §8): one traced RK4-step
graph with `z_prev` and the cotangent as inputs, `spot = sum(lambda * z_next)`,
`value_and_grad` over the 4 parameter tensors and `z_prev`; `compile_many`
emits `z_next` plus the 5 gradients; an epoch is 10 forward and 10 reverse runs
of the single prepared program. This is the discrete adjoint of the window,
not the continuous-adjoint stretch.

Independent review of the plan section before implementation (`gpt-6.1-sol`):
3 IMPORTANT (the last MAE cotangent was double-counted; the evaluation buffer
must stay column-major; the adjoint needs a finite-difference test and a
zero-residual test) and 1 MINOR (the MLP has 4 parameter tensors, so 5
gradients and 6 outputs). All fixed in §8 before coding.

Implementation:

- `src/train.rs` — `build` (the step graph), `step_reads`,
  `window_loss_and_grad` (forward checkpoints, host MAE loss, reverse
  accumulation with `lambda` carried through the `z_prev` gradient), `prepare`.
- `src/train/tests.rs` — `adjoint_matches_a_finite_difference_of_the_window_loss`
  (directional derivative, `DELTA = 1e-5`, relative tolerance 1e-9) and
  `zero_residual_has_a_zero_cotangent_and_gradient`.
- `src/main.rs` — epoch body calls `train::window_loss_and_grad`; one
  `prepare` before epoch 0; evaluation is a forward-only 10-step program
  chained on the host; trajectory stored column-major; per-time error prints;
  the figure plots block means of the loss (a single random minibatch is not
  readable).
- `src/data.rs`, `src/model.rs`, `src/plot.rs`, `src/ode.rs` — unchanged except
  `data::BATCH` and an `ode` doc line.

## Verification performed (final tree)

```
cargo fmt --all --check                        # clean
cargo clippy -j 16 --workspace --all-targets --release -- -D warnings
                                               # No issues found
cargo test -j 16 --workspace --release         # 91 passed
cargo test -j 16 -p neural_ode --release       # 14 passed
cargo run -j 16 --release -p kdv_pinn -- --epochs 20 --loss-png /tmp/n12/kdv.png
                                               # 1.6 s, saved PNG (unchanged)
```

Full run, seed 42 (`/tmp/n12/run/seed42.log`): build + compile 0.001 s,
prepare 0.008 s, training 151.7 s (7.58 ms/epoch), final window loss
4.0e-4, trajectory relL2 **31 %**. Seed sweep (20k epochs each, same code):
43 15 %, 44 46 %, 45 **3.8 %**, 46 17 %, 47 15 %. 50k epochs, seed 42: 31.6 %.

The accuracy bar (`seed 42 relL2 <= 5 %`) is **not met**; the window loss is at
its floor for every seed and the long-horizon error is a phase drift that the
10-step objective does not select. The README reports the six seeds and the
drift; `--seed 45` shows the 3.8 % trajectory. Bar status and diagnosis are in
`IMPLEMENTATION-PLAN.md` §8.6.

## Post-implementation review

`gpt-6.1-sol` on the staged diff vs `origin/main`: **VERDICT: CHANGES**.
1 IMPORTANT — `prepare` sat inside the epoch loop behind an `Option` guard, so
`--epochs 0` never prepared it; hoisted before the loop with a fixed `[64, 2]`
binding batch, the guard and `expect` removed. 6 MINOR — inaccurate
README/comment claims (per-window loss wording, the obsolete constant-leaf
claim about the continuous adjoint, an "exact" finite-difference comment, a
cubic-vanishing claim about the biased MLP, a two-vs-three panel count, and a
plot doc on the loss array length); all corrected. After the fixes: fmt clean,
clippy clean, 93 tests pass, and a fresh seed-42 20k run reproduces the final
loss (4.002925e-4) and relL2 (3.110504e-1) exactly.

Upstream report: tensor4all/tenferro-rs#2643.

## Environment

tenferro 0.11.0 (crates.io), AMD Ryzen AI MAX+ 395, rustc/cargo 1.100.0,
single-threaded `CpuEngine`. Raw logs and the patched runtime copy are under
`/tmp/n12/` (not part of the repository).
