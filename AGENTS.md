# AGENTS.md

Before acting, read the latest shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`,
and load only the rule files relevant to the task.

Repository-specific notes:

- Every example builds against the released tenferro crates pinned in the root
  `Cargo.toml`. Do not switch an example to a git or path dependency on
  tenferro-rs; the `upstream-main.yml` workflow covers `main`.
- An example that needs an unreleased tenferro API waits on a branch until that
  API is released.
- Keep each example runnable in a short mode so CI can execute it, not only
  compile it.
