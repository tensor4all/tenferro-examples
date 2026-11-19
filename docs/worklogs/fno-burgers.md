# FNO Burgers example

The example completes #11 using released tenferro 0.12, which contains the FFT
AD, slice-extent and padded-adjoint fixes. The 0.12 examples port (#18) is now
part of the base. No port or upstream implementation is included here.

The native real-FFT path follows the later maintainer guidance instead of the
early C2C fallback. Explicit half-spectrum padding preserves the model's mode
truncation boundary in the differentiated graph. Shared MLP components provide
all pointwise layers. The existing Adam implementation requires homogeneous
parameter dtypes per instance, so real and complex parameter tensors are ordered
in separate contiguous groups and updated by two Adam instances. This avoids an
unrelated optimizer API change. A configured seed-1 `vjp_many` supplies the
single reverse pass with FFT extension rules that default `value_and_grad`
does not install in 0.12.

The reference adapts the existing nonlinear-PINN integrating-factor solver to a
random periodic ensemble. Initial dealiasing is necessary before the quadratic
product; exact Cole–Hopf, conservation, alias-rejection and timestep tests cover
that boundary. Training-only RMS normalization keeps the strongly diffused
zero-mean target at a useful loss scale. One common fine held-out ensemble makes
the resolution comparison about the same functions and final weights.

Full settings, measured learning errors and reproduction instructions belong in
the [example README](../../examples/fno-burgers/README.md). The five-step CI case
is a smoke run. The CPU reference and spectral gradient checks are quantitative;
no GPU or paper-scale accuracy is claimed. Concrete reference FFTs allocate
outputs even though the executor reuses plans; the owned stage buffer is reused.

Local release verification passed: formatting, workspace Clippy with warnings denied,
141 workspace tests, and all 12 workflow short runs. The default experiment
measured held-out relative L2 errors of 2.0534%, 2.0543%, and 2.0544% at
128/256/512. The 17 focused tests include real/imaginary spectral gradients
and a seed-42 reference timestep residual of 1.73e-8. Hosted exact-head CI and
upstream-main verification remain publication-time checks; GPU was not run.
Independent review is recorded in this chat's handoff, separate from the
scientific reproducibility instructions.
