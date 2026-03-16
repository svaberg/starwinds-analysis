# Octree Interpolator Performance Report

Date: 2026-03-08  
Environment: `batwind`  
Dataset: `sample_data/difflevels-3d__var_1_n00000000.dat`

## Scope

This report summarizes measured timing results from today/yesterday while optimizing the octree interpolator with numba, threaded execution, and coherence-aware lookup reuse.

## Executive Summary

The biggest gains came from:

1. Moving lookup + trilinear hot loops into numba nopython kernels.
2. Running batch query loops with numba parallel `prange`.
3. Adding coherence-aware previous-cell containment reuse for ray/grid-like query orders.

Net effect:

1. `interp(xyz)` at `n=4000`: about `41x` faster vs pre-numba baseline.
2. Lookup-only path: about `8x` faster vs pre-numba baseline.
3. Additional parallel gain on large batches: up to `~4.1x` from 1 to 8 threads.
4. For coherent rays with new seeded chunked mode: about `~3x` faster kernel time vs independent kernel.

## Baseline vs Optimized

| Scenario | Query count | Time (s) | Throughput | Speedup |
|---|---:|---:|---:|---:|
| Pre-numba `interp(xyz)` | 4000 | 0.193639 | 20.7k q/s | 1.0x |
| Post-numba `interp(xyz)` | 4000 | 0.004679 | 855k q/s | 41.4x |
| Pre-numba lookup-only | 4000 | 0.128748 | 31.1k q/s | 1.0x |
| Post-numba lookup-only | 4000 | 0.015505 | 258k q/s | 8.3x |

## Parallel Scaling (Numba)

### `interp(xyz)` on 200k queries

| Threads | Time (s) | Throughput | Speedup vs 1T |
|---:|---:|---:|---:|
| 1 | 0.219384 | 0.91M q/s | 1.00x |
| 2 | 0.117134 | 1.71M q/s | 1.87x |
| 4 | 0.064243 | 3.11M q/s | 3.42x |
| 8 | 0.053349 | 3.75M q/s | 4.11x |

### `interp(rpa)` on 200k queries

| Threads | Time (s) | Throughput | Speedup vs 1T |
|---:|---:|---:|---:|
| 1 | 0.219046 | 0.91M q/s | 1.00x |
| 2 | 0.116986 | 1.71M q/s | 1.87x |
| 4 | 0.063995 | 3.13M q/s | 3.42x |
| 8 | 0.057401 | 3.48M q/s | 3.82x |

## Bottleneck Breakdown (Post-numba/parallel)

### Kernel internal split (`xyz` path)

| Query count | Lookup-only (s) | Trilinear-only (s) | Lookup share of kernel |
|---:|---:|---:|---:|
| 4000 | 0.000707 | 0.000206 | 67.1% |
| 20000 | 0.003308 | 0.001493 | 64.5% |
| 200000 | 0.036319 | 0.015667 | 62.2% |

Interpretation: lookup remains the primary bottleneck after optimization.

## Coherence-Aware Seeded Mode (Current)

Auto mode picks seeded kernel only for coherent query orderings.

| Query set | Query count | Coherent detected | Auto time (s) | Throughput |
|---|---:|---|---:|---:|
| Random sample order | 200000 | No | 0.069440 | 2.88M q/s |
| Ray ordered | 200000 | Yes | 0.006159 | 32.48M q/s |
| Regular cartesian grid order | 384000 | Yes | 0.042373 | 9.06M q/s |

Kernel-only comparison (current code):

| Query set | Independent kernel (s) | Seeded chunked kernel (s) | Kernel speedup |
|---|---:|---:|---:|
| Random | 0.070187 | 0.060786 | 1.15x |
| Ray | 0.008697 | 0.002925 | 2.97x |
| Grid | 0.033282 | 0.029460 | 1.13x |

Note: random-order seeded benefit is variable run-to-run; coherence gating is still kept to avoid unstable regressions.

## What Gave Improvement

1. Numba JIT on lookup/trilinear hot loops removed Python per-point overhead.
2. Batch kernels moved work from Python loops to contiguous array kernels.
3. `prange` parallelism scaled throughput on large batches.
4. Chunked previous-cell containment reuse reduced lookup work for coherent trajectories.
5. Coherence detection avoided applying seeded mode blindly to all orders.

## Current Regressions and Validation Status

Validation runs:

1. `pytest -q test/test_octree*.py` -> `54 passed, 1 skipped`.
2. `STARWINDS_RUN_PERF_TESTS=1 pytest -q test/test_octree_performance.py` -> `1 passed`.
3. Full `pytest -q` unchanged global status: one known failure in `test_code_rules_baseline` (not runtime behavior regression in octree logic).

