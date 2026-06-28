# CUDA / PTX Micro-architecture Benchmark Suite

A comprehensive micro-benchmarking repository for NVIDIA GPUs, written in CUDA C++ and inline PTX. It measures the key micro-architectural characteristics of NVIDIA GPUs at the instruction and SM level.

**Live report:** https://qq332982511.github.io/cuda_micro_bench/  
**Source code:** https://github.com/qq332982511/cuda_micro_bench

## Reference

Inspired by [zartbot/micro_arch](https://github.com/zartbot/micro_arch.git), this project focuses on NVIDIA GPUs and uses CUDA/PTX to measure:

- Device / SM topology and limits
- Core clock and timer behavior
- Global memory bandwidth and latency
- L1 / L2 cache size, line size and latency
- Shared memory size, banks, bandwidth and latency
- Instruction throughput and latency (FP32, FP64, INT, SFU, etc.)
- Atomic operation throughput and latency
- Tensor Core (MMA) latency and throughput via PTX
- Occupancy and register file effects
- Warp scheduler / divergence behavior

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit >= 11.0 (tested with CUDA 13.2)
- GNU make
- `nvcc` in your `PATH`

## Build

```bash
make
```

For a specific architecture, override `ARCH`:

```bash
make ARCH=sm_100
```

## Run

```bash
./cuda_micro_bench
```

Run only a specific benchmark:

```bash
./cuda_micro_bench --device
./cuda_micro_bench --memory
./cuda_micro_bench --compute
./cuda_micro_bench --latency
./cuda_micro_bench --atomics
./cuda_micro_bench --tensor
./cuda_micro_bench --occupancy
./cuda_micro_bench --shuffle
./cuda_micro_bench --barriers
./cuda_micro_bench --instructions
./cuda_micro_bench --tensor-ext
./cuda_micro_bench --pcie
./cuda_micro_bench --cluster
./cuda_micro_bench --all
./cuda_micro_bench --json   # run all and write results/results.json
```

## Viewing the HTML Report

After running the benchmark, open `results/report.html` in a browser (served over HTTP):

```bash
python3 -m http.server 8000
# then open http://localhost:8000/results/report.html
```

The report loads `results/results.json` and `results/reference.json` and compares measured values against the SM_120 reference from [zartbot/micro_arch](https://github.com/zartbot/micro_arch.git).

## Project Layout

```
├── include/         Common headers (timer, utilities)
├── src/             Benchmark kernels and driver
├── results/         Benchmark JSON output and HTML report
├── tools/           Helper scripts (reference extraction)
└── README.md
```

## Coverage vs. Reference

The reference repo [zartbot/micro_arch](https://github.com/zartbot/micro_arch.git) documents ~1500 test cases across 24 sections for SM_120. This implementation covers the most important executable categories:

- Topology / clocks / theoretical limits
- Memory hierarchy (DRAM, L1/L2, shared memory)
- Instruction latency/throughput (integer, FP32/FP64, SFU)
- Tensor Core (FP16/BF16/TF32/FP8/INT8 MMA)
- Atomics, barriers, warp shuffle
- PCIe / kernel launch latency
- Occupancy and cluster basics

Items not yet implemented include TMA engine characterization, multi-GPU P2P, fine-grained power/thermal profiling, and full instruction-encoding coverage. The JSON/HTML framework is designed to be extended as more tests are added.

## Methodology

- **Latency**: measured with long dependency chains and `clock()` / `%clock` PTX.
- **Throughput**: measured with enough independent warps/SMs to saturate the pipeline.
- **Bandwidth**: pointer-chase or streaming kernels with known data sizes.
- **PTX**: inline `asm volatile` blocks prevent compiler optimizations and expose exact instruction behavior.
- **Anti-DCE**: results are stored to global memory to prevent `ptxas` from eliminating measured chains.

## License

MIT
