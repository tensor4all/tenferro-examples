# tenferro-examples

Runnable examples for [tenferro-rs](https://github.com/tensor4all/tenferro-rs),
built against the tenferro crates released on crates.io.

| Example | What it shows | tenferro features |
| --- | --- | --- |
| [`kdv-pinn`](examples/kdv-pinn) | Train a physics-informed neural network for the Korteweg-de Vries equation | traced autodiff (`jvp`, `grad`), compiled graphs reused across epochs, CPU runtime |

## Running an example

Each example is its own crate in this Cargo workspace:

```bash
cargo run --release -p kdv_pinn
```

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
   `Short runs` step in `.github/workflows/ci.yml` and `upstream-main.yml`.
4. Add a row to the table above.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
