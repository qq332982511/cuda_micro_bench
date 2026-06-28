# Agent Instructions: cuda_micro_bench

## Project Purpose

CUDA / PTX micro-architecture benchmarking suite for NVIDIA GPUs. Measures device topology, memory hierarchy, compute throughput/latency, atomics, tensor cores, warp shuffle, barriers, PCIe/launch latency, cluster basics, and occupancy using inline PTX where helpful. Results are written to JSON and visualized/compared in `results/report.html`.

## Build

```bash
make              # compiles for the installed GPU via -arch=native
make ARCH=sm_80 # or target a specific architecture
```

## Run

```bash
./cuda_micro_bench              # all benchmarks
./cuda_micro_bench --device     # device info only
./cuda_micro_bench --memory --compute --latency  # multiple sections
```

## Code Style

- Use `cmb::` namespace for all project code.
- Always wrap CUDA calls with `CUDA_CHECK`.
- Use inline PTX (`asm volatile`) to prevent compiler optimizations in latency/throughput kernels.
- Use `clockVal()` / `clock64Val()` from `timer.h` for device-side timing.
- When measuring latency, guard the second clock read with a dependency on the chain result to avoid compiler reordering (see `CMB_DEPENDENCY_GUARD`).
- Use a single `extern __shared__ char smem_raw[];` per translation unit and cast inside kernels.
- Keep kernels minimal; do not add unrelated logic that could skew measurements.

## Adding a New Benchmark

1. Add `src/your_bench.cu` and declare `void benchmark_your_bench();`.
2. Call it from `src/main.cu` and add a `--your-bench` option.
3. Update `README.md`.
4. Run `make` and `./cuda_micro_bench --your-bench` to validate.

## Testing Expectations

- `make` must succeed.
- `./cuda_micro_bench` must run without CUDA errors.
- Numbers should be physically plausible for the target GPU.
