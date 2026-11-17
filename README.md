# tenferro-examples

Runnable examples for [tenferro-rs](https://github.com/tensor4all/tenferro-rs),
built against the tenferro crates released on crates.io.

| Example | What it shows | tenferro features |
| --- | --- | --- |
| [`kdv-pinn`](examples/kdv-pinn) | Train a physics-informed neural network for the Korteweg-de Vries equation | traced autodiff (`jvp`, `value_and_grad`), one compiled graph per epoch, CPU runtime |
| [`neural-ode`](examples/neural-ode) | Fit a 2-D spiral ODE with a checkpointed discrete adjoint through an RK4 solver | traced autodiff (`value_and_grad`) through a 10-step solver, one compiled graph per epoch, CPU runtime |
| [`deeponet`](examples/deeponet) | Learn the antiderivative operator with a DeepONet | traced autodiff (`value_and_grad`), one compiled graph per epoch, host `TensorLinalgExt::cholesky` for the random-field data, CPU runtime |
| [`nqs-rbm`](examples/nqs-rbm) | Find spin-model ground states with a neural-network quantum state (Carleo and Troyer) | eager autodiff (`backward`), typed-tensor `matmul`, `tenferro-linalg` `solve` and `eigh` |

## Running an example

Each example is its own crate in this Cargo workspace:

```bash
cargo run --release -p kdv_pinn
```

Model and optimizer code that two or more examples share — the traced MLP, the
ordered parameter set and Adam — lives in [`crates/sciml-kit`](crates/sciml-kit).
It holds no tenferro wrappers: compiling, preparing and running stay plain
tenferro calls in each example.

See each example's README for its options and an explanation of the code.

## tenferro versions

The tenferro version used by every example is set once, in
`[workspace.dependencies]` of the root `Cargo.toml`. Dependabot proposes the
bump when a new tenferro release is published.

## CI

- **CI** (`ci.yml`, on pull requests and pushes to `main`): `cargo fmt`,
  `cargo clippy -D warnings`, unit tests, and a short run of every example
  against the released tenferro crates.
- **Upstream main** (`upstream-main.yml`, daily and on demand): the same checks
  with the tenferro crates replaced by the tenferro-rs `main` branch. A failure
  opens (or comments on) an issue labeled `upstream-main`, so breakage is seen
  before the next tenferro release.

## Adding an example

1. Create `examples/<name>/` with its own `Cargo.toml` that takes tenferro
   crates from the workspace (`tenferro-runtime.workspace = true`).
2. Add a `README.md` that explains the problem and how the code is laid out.
3. Give it a short mode (like `--epochs`) and add that command to the
   `Short runs` step in `.github/workflows/checks.yml`, which `ci.yml` and
   `upstream-main.yml` both call.
4. Add a row to the table above.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
