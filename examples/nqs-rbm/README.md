# Neural-network quantum states

Can a neural network represent the ground state of an interacting quantum
many-body system, and can it be trained without ever writing down the
exponentially large state vector?

[Carleo and Troyer](https://arxiv.org/abs/1606.02318) (Science 355, 602 (2017))
answered yes, using a restricted Boltzmann machine (RBM) as the wavefunction
and variational Monte Carlo to train it. This example trains that network for
the three systems of the paper's Figure 2. Tenferro is used for the linear
algebra of the optimizer, and its autodiff checks the hand-written derivatives
the optimizer relies on.

This is a qualitative reproduction with a simpler, real-valued ansatz; see
[Differences from the paper](#differences-from-the-paper).

## Problem Being Solved

The unknown is the ground state $|\Psi_0\rangle$ of a spin-1/2 lattice
Hamiltonian, written with Pauli matrices as in the paper. The two models are
the critical transverse-field Ising model

$$
H_\mathrm{TFI} = -\sum_{\langle ij\rangle}\sigma^z_i\sigma^z_j - h\sum_i\sigma^x_i,
\qquad h = 1,
$$

and the antiferromagnetic Heisenberg model

$$
H_\mathrm{AFH} = \sum_{\langle ij\rangle}
  \left(\sigma^x_i\sigma^x_j+\sigma^y_i\sigma^y_j+\sigma^z_i\sigma^z_j\right).
$$

$\langle ij\rangle$ runs over nearest neighbors, and all lattices are periodic.
Energies quoted for $\mathbf S_i\cdot\mathbf S_j$ must be multiplied by four
before comparison. The three systems are:

| `--system` | Lattice | $\alpha$ | Hidden units $\alpha N$ | Weights | Reference $E/N$ |
| --- | ---: | ---: | ---: | ---: | --- |
| `ising-chain` | 80 sites | 4 | 320 | 320 | exact free-fermion solution |
| `heisenberg-chain` | 80 sites | 4 | 320 | 320 | exact Bethe-ansatz solution |
| `heisenberg-square` | 10×10 | 16 | 1600 | 1600 | QMC, $-4(0.67155267)$ ([Chen and Heyl](https://pmc.ncbi.nlm.nih.gov/articles/PMC11392813/)) |

The program prints the relative error of the trained energy against the
reference. Other square-lattice sizes have no reference value.

## Model

A configuration $s = (s_1,\dots,s_N)$, $s_i = \pm1$, is a basis state in the
$\sigma^z$ basis. The wavefunction is an RBM whose hidden units have been
summed out. Its $\alpha$ filters $W^{(f)}$ each hold one weight per site and
are shared by all lattice translations $t$, as in the paper's Appendix E:

$$
\log\psi(s) = \sum_{f=1}^{\alpha}\sum_{t}
  \log\left[2\cosh\left(\theta_{f,t}(s)\right)\right],
\qquad
\theta_{f,t}(s) = \sum_i W^{(f)}_i\, s_{i+t}.
$$

The trained parameters are the $\alpha N$ filter weights. Biases are zero and
the weights are real, so every amplitude is positive. That is exact for the
Ising model. The Heisenberg ground state on a bipartite lattice has the
Marshall sign $(-1)^{\#\text{up spins on one sublattice}}$; the code
works in the rotated basis in which that sign is absorbed into $H$, and it
samples only configurations of zero total magnetization, the sector that
contains the ground state.

## Variational Monte Carlo

The energy of $\psi$ is an average over configurations drawn from
$|\psi(s)|^2$:

$$
E(W) = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
     = \left\langle E_\mathrm{loc}(s)\right\rangle_{|\psi|^2},
\qquad
E_\mathrm{loc}(s) = \sum_{s'} H_{ss'}\frac{\psi(s')}{\psi(s)}.
$$

$H$ is sparse: $s'$ differs from $s$ by one flipped spin (Ising) or one
exchanged antiparallel pair (Heisenberg), so $E_\mathrm{loc}$ costs $O(N)$
amplitude ratios. Eight Metropolis chains propose the same moves and persist
across optimization steps; the hidden-unit inputs $\theta_{f,t}$ are cached,
so each ratio costs $O(\alpha N)$.

## Stochastic Reconfiguration

With log-derivatives $O_k(s) = \partial\log\psi(s)/\partial W_k$, each step
solves the stochastic reconfiguration (SR) equations

$$
\left(S + \lambda\,\mathrm{diag}(S) + \epsilon I\right)\delta = F,
\qquad
S_{kl} = \langle O_k O_l\rangle - \langle O_k\rangle\langle O_l\rangle,
\qquad
F_k = \langle O_k E_\mathrm{loc}\rangle - \langle O_k\rangle\langle E_\mathrm{loc}\rangle,
$$

and updates $W \leftarrow W - \eta\,\delta$. The damping is
$\lambda = \max(100\cdot0.9^p, 0.01)$ at step $p$, and $\epsilon$ is
`--sr-shift`. The rate $\eta \le 0.05$ is reduced so that the step length in
the sampled metric is at most 0.12 and the root-mean-square weight change is
at most 0.002. `--max-weight-step` also bounds the largest single weight
change; the sampled $S$ does not constrain directions that the batch does not
explore.

When there are fewer samples than parameters (the square lattice), the
Woodbury identity turns the $p\times p$ system into an equivalent system of
the sample count's size, so $S$ is never formed.

## Run The Sample

From the repository root of `tenferro-examples`:

```bash
cargo run --release -p nqs_rbm
cargo run --release -p nqs_rbm -- --system heisenberg-chain
cargo run --release -p nqs_rbm -- --system heisenberg-square
```

Each run prints the energy every 20 SR steps, then an independent validation
estimate from eight new chains (400 burn-in sweeps and 2048 samples per chain,
with the standard error from 64-sweep blocks) and its relative error against
the reference.

The default number of steps is reduced from the longer runs in
[Results from longer runs](#results-from-longer-runs) so that each system
finishes in about a minute on a laptop:

| `--system` | Steps | Samples per step | Validation $E/N$ | Relative error | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ising-chain` | 300 | 1024 | −1.273206 ± 0.000023 | 0.009% | 33 s |
| `heisenberg-chain` | 300 | 1024 | −1.76469 ± 0.00018 | 0.47% | 28 s |
| `heisenberg-square` | 200 | 512 | −2.65779 ± 0.00053 | 1.1% | 88 s |

Wall times are for an 11-core Apple laptop; sampling runs the eight chains in
parallel, so fewer cores are slower.

Pass `--steps 700` for the chains, or `--steps 500` for the square lattice, to
run the longer calculation. `--help` lists every option; the defaults for each
system are the SR settings and seeds used for the longer runs.

### Exact check on small lattices

For lattices whose sampled sector has at most 4096 configurations (Ising
chains up to 12 sites, Heisenberg chains up to 14), `--exact` builds the
Hamiltonian as a dense matrix, diagonalizes it with tenferro, and evaluates the
trained wavefunction exactly:

```bash
cargo run --release -p nqs_rbm -- --size 8 --steps 1000 --samples 512 \
  --sr-shift 1e-3 --max-weight-step 0.002 --exact
```

prints

```text
exact (256 states): ground E/N = -1.2814577239, trained E/N = -1.2814571676, relative error 4.341e-7, fidelity 0.999999582
```

and the same command with `--system heisenberg-chain` gives a relative error
of $2.7\times10^{-9}$ and fidelity 0.999999998. This validates the sampler,
local energies, and optimizer on small lattices; it does not establish the
accuracy of the full-size calculations.

### Outputs

- `--output-dir DIR` writes `history.csv` (energy, variance, acceptance, and
  step size per SR step), `weights.csv` (one filter per line, saved every 20
  steps), `validation_energies.csv`, and `summary.json`. The program refuses a
  directory that already holds a `history.csv`.
- `--init-weights FILE` starts from a `weights.csv`. The damping schedule
  starts again from step 0, so this is not an exact continuation of the
  earlier run.
- `--energy-png FILE` plots the training energy per site (red) against SR
  step on a linear axis, with the reference energy as a black horizontal line.
- `--filters-png FILE` draws the trained filters, as in the paper's Figure 2.
  A chain shows one row per filter; a square lattice shows one tile per
  filter, with $x$ to the right and $y$ upward. Blue is negative, white zero,
  and red positive, saturating at $\pm0.15$ for chains and $\pm0.30$ for the
  square lattice, the paper's color ranges. Translating a filter or changing
  its sign leaves $\psi$ unchanged, so each filter is shown shifted to put its
  largest-magnitude weight at the center, with that weight positive.

The plots contain no text, so the sample needs no font library.

CI runs the unit tests and short runs on 8-site chains (with `--exact`) and a
4×4 square lattice. It does not run the full-size calculations.

## How It Is Structured

| File | Role |
| --- | --- |
| `src/model.rs` | Translation-invariant RBM: amplitudes, log-derivatives, cached flip ratios, and local energies of both Hamiltonians. |
| `src/sampler.rs` | Metropolis chains, parallel sampling with `rayon`, and blocked error estimates. |
| `src/sr.rs` | SR solve with tenferro `matmul` and `solve`, damping schedule, and step-size bounds. |
| `src/autodiff.rs` | The RBM in tenferro eager autodiff, used to check the analytic log-derivatives. |
| `src/reference.rs` | Reference energies and the exact small-lattice check with tenferro `eigh`. |
| `src/config.rs` | Command-line options and per-system defaults. |
| `src/plot.rs` | Filter and training-energy PNGs. |

## How tenferro Is Used

The training loop computes $O_k(s)$ analytically, because it needs them for
every sample. Tenferro's eager autodiff checks them. `rbm_log_psi` writes a
general RBM, $\log\psi(s) = a^Ts + \sum_j\log 2\cosh(b_j + (Ws)_j)$, for a
batch of configurations stored one per column:

```rust
let projection = session.matmul(w, spins)?;
let z = session.add(&projection, b)?;
// log(2 cosh(z)) = softplus(2z) - z does not overflow for large z.
let twice_z = session.add(&z, &z)?;
let softplus = session.softplus(&twice_z)?;
let log_cosh = session.sub(&softplus, &z)?;
let hidden = session.reduce_sum(&log_cosh, Some(&[0]))?;
let products = session.mul(a, spins)?;
let visible = session.reduce_sum(&products, Some(&[0]))?;
session.add(&visible, &hidden)
```

The translation-invariant ansatz is this RBM with zero biases, evaluated on
the $N$ translated copies of one configuration and summed. Reverse mode then
gives the tied-weight derivative:

```rust
runtime.with_eager_session(&mut engine, |session| {
    let per_translation = rbm_log_psi(session, &spins, &a, &b, &w)?;
    let loss = session.reduce_sum(&per_translation, None)?;
    session.backward(&loss)
})?;
let gradient = w.grad()?.ok_or("weights did not receive a gradient")?;
```

Every run compares the two on a starting configuration before training and
stops if they differ by more than $10^{-10}$ relative. The unit tests also
check `rbm_log_psi` against an explicit sum over hidden spins and its
gradients against finite differences.

The SR step uses tenferro typed tensors on the CPU engine: `matmul` forms
$S$ (or the sample-space Gram matrix), and `tenferro-linalg` `solve` solves
the regularized system. The same `solve` drives Newton's method for the Bethe
equations, and `eigh` diagonalizes the dense Hamiltonian for `--exact`.

## Differences From The Paper

- Weights are real and biases are zero, with the analytic Marshall sign for
  the Heisenberg models. The paper uses complex weights, which can learn a
  sign structure.
- The SR matrix has an identity shift $\epsilon$ in addition to the
  paper's diagonal damping, and the step-size bounds above. The paper reports
  a typical damping floor of $10^{-4}$; this code uses $10^{-2}$.
- Each step uses 1024 samples for the chains and 512 for the square lattice.
  A sweep attempts $N$ single flips (Ising) or $N/2$ exchanges (Heisenberg);
  5% of proposals are skipped, which stops a chain that accepts every flip
  from alternating between the two magnetization parities.
- Error bars are standard errors from 64-sweep blocks. They do not include
  optimization bias, and a fixed burn-in and block size do not prove that
  the chains have equilibrated.

## Results From Longer Runs

Running with `--steps 700` for the chains and `--steps 500` for the square
lattice, otherwise with the defaults, gives these validation estimates (±1
estimated standard error). The chains take about 1.5 minutes each and the
square lattice about 3.5 minutes on the laptop above.

| System | Trained $E/N$ | Reference $E/N$ | Relative error |
| --- | ---: | ---: | ---: |
| Ising, $N=80$ | −1.273256 ± 0.000021 | −1.27332136 | 0.0051% |
| Heisenberg, $N=80$ | −1.772983 ± 0.000057 | −1.77310381 | 0.0068% |
| Heisenberg, 10×10 | −2.66383 ± 0.00047 | −2.68621068 | 0.83% |

The 2D result is still well above its benchmark, so the trained filters and
energies are a first reproduction, not a replication of the paper's
precision. Weights can differ greatly between runs with similar energies; the
Heisenberg chain run has a largest absolute weight of about 9, which the
fixed color range saturates. Use `weights.csv` to study the weights
themselves.
