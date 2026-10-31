# IFT cost diagnostic protocol

This protocol describes how `ift_cost_diagnostic.csv` was produced. It is method-cost metadata for a fixed-branch JAX fixture, not a production benchmark.

## Machine and software

- Python: 3.9.6 (default, Jan  9 2026, 11:03:41)  [Clang 17.0.0 (clang-1700.6.4.2)]
- Platform: macOS-26.4.1-x86_64-i386-64bit
- Machine: x86_64
- Processor: i386
- NumPy: 2.0.2
- SciPy: 1.13.1
- JAX: 0.4.30
- JAXLIB: 0.4.30
- JAX backend: cpu

## Fixture

- Parameter: HeatLossPercent
- Output: final_T_K
- Fixture: fixed-branch JAX scan fixture with synthetic two-cell/N-cell source pattern
- Nonlinear solve: fixed-iteration _newton_loop_12; no production solver API changes

## Method definitions

- `ift_custom_vjp`: each cell solve uses the existing custom VJP helper. The backward pass applies the implicit-function theorem by solving `F_x.T @ lambda = cotangent` and returning the parameter/carry cotangents from `-lambda.T @ F_theta` via the residual pullback.
- `direct_unrolled_ad`: the same fixed-iteration Newton solve is used as an ordinary JAX computation graph. `jax.grad(final_temperature)` differentiates through the unrolled nonlinear iterations and scan carry directly, without custom VJP.

## Timing definitions

- `compile_or_cold_time_s`: wall-clock time for the first `jax.jit(jax.grad(...))(theta)` call inside a fresh subprocess for that method/N case. It includes tracing, XLA compilation, and the first execution, but not CSV writing.
- `warm_eval_time_s`: median wall-clock time across 2 subsequent calls to the same compiled gradient function in the same subprocess.
- The very small warm times are expected because they measure already-compiled scalar-gradient calls on the same fixed fixture.

## Timeout rule

- Each method/N case is executed in its own spawned Python subprocess. The parent process waits `timeout_s=90.0` seconds using `Process.join(timeout_s)`.
- If the subprocess is still alive after the timeout, it is terminated and a row with `status=timeout` and `failure_reason='case exceeded timeout_s=...'` is recorded.
- A timeout is treated as a reproducibility/cost boundary, not hidden or retried with looser settings.

## Evidence boundary

This protocol supports the methodological claim that IFT/custom VJP is more scalable than direct unrolled AD for this fixed-branch implicit-solve fixture. Timings are machine- and JAX-version-dependent. They are not production reactor benchmarks, training results, or hybrid calibration accuracy evidence.
