# /// script
# requires-python = ">=3.11"
# dependencies = ["jax>=0.4.30", "optax>=0.2.4", "numpy"]
# ///
"""JAX + Optax baseline for the nonlinear-pinn example (not run in CI).

Same problems, network, training points, loss weights, Adam schedule and
evaluation grid as `src/main.rs`; L-BFGS is `optax.lbfgs` with its default
zoom (strong Wolfe) line search instead of the example's backtracking Armijo
search. Run single-threaded, like the example:

    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c 0 uv run jax_baseline.py --problem burgers --optimizer adam-lbfgs
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

NU = 0.01 / np.pi
EPSILON = 1e-4
N_F, N_IC, N_BC = 8000, 256, 64
DEPTH, WIDTH = 6, 32
ADAM_LR = 1e-3
LBFGS_MEMORY = 50


def initial(problem, x):
    return -np.sin(np.pi * x) if problem == "burgers" else x**2 * np.cos(np.pi * x)


def reference(problem, n, dt, dt_out=0.01, t_end=1.0):
    """Lawson integrating-factor RK4 pseudo-spectral solve, as reference.rs."""
    c = NU if problem == "burgers" else EPSILON
    k = np.pi * np.arange(n // 2 + 1)
    e, e2 = np.exp(-c * k**2 * dt), np.exp(-c * k**2 * dt / 2)
    keep = (3 * np.arange(n // 2 + 1) < n).astype(float)
    x = -1 + 2 * np.arange(n) / n

    def nonlinear(s):
        u = np.fft.irfft(s, n)
        if problem == "burgers":
            return -0.5j * k * np.fft.rfft(u * u) * keep
        return np.fft.rfft(5 * u - 5 * u**3) * keep

    s = np.fft.rfft(initial(problem, x))
    per = round(dt_out / dt)
    us = [np.fft.irfft(s, n)]
    for _ in range(round(t_end / dt_out)):
        for _ in range(per):
            k1 = nonlinear(s)
            k2 = nonlinear(e2 * (s + dt / 2 * k1))
            k3 = nonlinear(e2 * s + dt / 2 * k2)
            k4 = nonlinear(e * s + dt * e2 * k3)
            s = e * s + dt / 6 * (e * k1 + 2 * e2 * (k2 + k3) + k4)
        us.append(np.fft.irfft(s, n))
    return x, np.linspace(0, t_end, len(us)), np.array(us)


def features(problem, x, t):
    if problem == "burgers":
        return jnp.stack([x, t])
    return jnp.stack([jnp.cos(np.pi * x), jnp.sin(np.pi * x), t])


def init_params(key, sizes):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        scale = np.sqrt(6 / (fan_in + fan_out))
        params.append((jax.random.uniform(sub, (fan_in, fan_out), minval=-scale, maxval=scale), jnp.zeros(fan_out)))
    return params


def make_loss(problem, data):
    def u(params, x, t):
        h = features(problem, x, t)
        for w, b in params[:-1]:
            h = jnp.tanh(h @ w + b)
        w, b = params[-1]
        return (h @ w + b)[0]

    u_t = jax.grad(u, 2)
    u_x = jax.grad(u, 1)
    u_xx = jax.grad(u_x, 1)

    def residual(params, x, t):
        v = u(params, x, t)
        if problem == "burgers":
            return u_t(params, x, t) + v * u_x(params, x, t) - NU * u_xx(params, x, t)
        return u_t(params, x, t) - EPSILON * u_xx(params, x, t) + 5 * v**3 - 5 * v

    batch_u = jax.vmap(u, (None, 0, 0))
    batch_r = jax.vmap(residual, (None, 0, 0))
    ic_weight = 1.0 if problem == "burgers" else 100.0

    def loss(params):
        total = jnp.mean(batch_r(params, data["x_f"], data["t_f"]) ** 2)
        ic = batch_u(params, data["x_ic"], jnp.zeros_like(data["x_ic"])) - data["u_ic"]
        total += ic_weight * jnp.mean(ic**2)
        if problem == "burgers":
            total += jnp.mean(batch_u(params, data["x_bc"], data["t_bc"]) ** 2)
        return total

    return loss, batch_u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="burgers", choices=["burgers", "allen-cahn"])
    parser.add_argument("--optimizer", default="adam-lbfgs", choices=["adam", "adam-lbfgs"])
    parser.add_argument("--adam-steps", type=int, default=10000)
    parser.add_argument("--lbfgs-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    problem = args.problem

    n_ref = 4096 if problem == "burgers" else 2048
    x_ref, t_ref, u_ref = reference(problem, n_ref, 1e-4)

    rng = np.random.default_rng(args.seed)
    data = {
        "x_f": rng.uniform(-1, 1, N_F),
        "t_f": rng.uniform(0, 1, N_F),
        "x_ic": rng.uniform(-1, 1, N_IC),
        "x_bc": np.repeat([-1.0, 1.0], N_BC),
        "t_bc": rng.uniform(0, 1, 2 * N_BC),
    }
    data["u_ic"] = initial(problem, data["x_ic"])
    data = {name: jnp.asarray(value) for name, value in data.items()}
    loss, batch_u = make_loss(problem, data)
    input_width = 2 if problem == "burgers" else 3
    params = init_params(jax.random.PRNGKey(args.seed), [input_width] + [WIDTH] * DEPTH + [1])

    schedule = optax.piecewise_constant_schedule(
        ADAM_LR, {args.adam_steps // 2: 0.5, 3 * args.adam_steps // 4: 0.5}
    )
    adam = optax.adam(schedule)
    state = adam.init(params)

    @jax.jit
    def adam_step(params, state):
        value, grad = jax.value_and_grad(loss)(params)
        updates, state = adam.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value

    start = time.perf_counter()
    for step in range(args.adam_steps):
        params, state, value = adam_step(params, state)
        if step % 1000 == 0:
            print(f"adam {step}: loss {float(value):.4e}", flush=True)
    jax.block_until_ready(params)
    print(f"adam done: {time.perf_counter() - start:.1f} s")

    if args.optimizer == "adam-lbfgs":
        lbfgs = optax.lbfgs(memory_size=LBFGS_MEMORY)
        state = lbfgs.init(params)
        value_and_grad = optax.value_and_grad_from_state(loss)

        @jax.jit
        def lbfgs_step(params, state):
            value, grad = value_and_grad(params, state=state)
            updates, state = lbfgs.update(grad, state, params, value=value, grad=grad, value_fn=loss)
            return optax.apply_updates(params, updates), state, value

        for step in range(args.lbfgs_steps):
            params, state, value = lbfgs_step(params, state)
            if step % 1000 == 0:
                print(f"l-bfgs {step}: loss {float(value):.4e}", flush=True)
    jax.block_until_ready(params)
    train_seconds = time.perf_counter() - start

    stride = n_ref // 256
    xs = x_ref[::stride]
    grid_x = np.tile(xs, len(t_ref))
    grid_t = np.repeat(t_ref, len(xs))
    prediction = np.asarray(batch_u(params, jnp.asarray(grid_x), jnp.asarray(grid_t)))
    truth = u_ref[:, ::stride].reshape(-1)
    error = np.linalg.norm(prediction - truth) / np.linalg.norm(truth)
    print(
        f"{problem} {args.optimizer}: final loss {float(loss(params)):.4e}, "
        f"relative L2 error {error:.4e}, training {train_seconds:.1f} s"
    )


if __name__ == "__main__":
    main()
