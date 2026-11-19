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

The [released-dependency CI run](https://github.com/tensor4all/tenferro-examples/actions/runs/44440237228)
passed for the original head `0d3a28d`: formatting, workspace Clippy, all 141
workspace tests and all 12 short runs. GPU and upstream-main execution were not
verified locally.

The review follow-up makes help parsing respect option-value positions, returns
an indexed typed internal error for disconnected model parameters, and asserts
nonempty RMS storage. The numerical guards now allow modest headroom over the
recorded residuals instead of orders of magnitude. All 146 workspace tests pass,
including 22 FNO tests with the new failure-path and CLI regressions; formatting and focused
Clippy pass. A fresh default run on source commit `55dde3b` reproduces all four
printed learning errors and retains its settings, timings and numerical checks
in the [committed run log](../../examples/fno-burgers/results/default-run.txt).
Hosted CI validates each revision; the different-provider review of the original
head does not cover later pushes.
