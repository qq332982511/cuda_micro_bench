# CUDA / PTX Micro-architecture Benchmark Suite

A comprehensive micro-benchmarking repository for NVIDIA GPUs, written in CUDA C++ and inline PTX. It measures the key micro-architectural characteristics of NVIDIA GPUs at the instruction and SM level.

**Live report:** https://qq332982511.github.io/cuda_micro_bench/  
**Source code:** https://github.com/qq332982511/cuda_micro_bench

## Reference

Inspired by [zartbot/micro_arch](https://github.com/zartbot/micro_arch.git), this project focuses on NVIDIA GPUs and uses CUDA/PTX to measure:

- Device / SM topology and limits
- Core clock and timer behavior
- Global memory bandwidth and latency
- Reported L2 size, cache working-set curves and dependency costs
- Shared memory size, banks, bandwidth and latency
- Instruction throughput and latency (FP32, FP64, INT, SFU, etc.)
- Atomic operation throughput and latency
- Tensor Core (MMA) latency and throughput via PTX
- Occupancy and kernel launch limits
- Warp scheduler / divergence behavior

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 13.2 for the reviewed build (older toolkits have not been validated)
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

The report loads `results/results.json`. The archived SM_120 reference from [zartbot/micro_arch](https://github.com/zartbot/micro_arch.git) describes a different GB202 GPU and is not a pass/fail target for RTX 5060 Ti.

See [REVIEW.md](REVIEW.md) for the architecture sources, corrected methodology, three full retests and remaining limitations. `docs/index.html` and `docs/results.json` mirror the local report and data.

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

Items not yet implemented include multi-GPU P2P, fine-grained power/thermal profiling, and full instruction-encoding coverage. The JSON/HTML framework is designed to be extended as more tests are added.

## Methodology

- **Dependency-chain cost**: PTX chains are timed using SM clocks. A result-dependent predicate guards the final clock read. Costs include loop/address overhead and are not automatically physical pipeline depths or isolated SASS latencies.
- **Compiler verification**: `asm volatile` protects against CUDA C++ optimization but does not disable ptxas optimization. Inspect SASS to verify operation counts, register paths and timer dependencies. RCP and integer loops that fold when unrolled use `#pragma unroll 1`; scalar ADD latency uses an alternating two-register chain to prevent ADD/IADD3 merging. SFU probes use approximate FTZ operations where supported.
- **Throughput**: independent chains across enough warps/SMs; FMA counts two FLOPs and integer matrix multiply counts multiply plus add. Arithmetic IPC counts warp PTX operations per SM, not individual lane instructions. Compute throughput uses sustained same-kernel prewarming and five-sample medians. Event-based IPC uses the nominal clock attribute and can differ with GPU boost.
- **Memory**: read-only streams and read+write copies have separate metrics. DRAM streaming sets exceed L2; each cold chase sample traverses the whole randomized cycle, with its minimum touched-sector footprint exceeding L2. Random dependent loads visit all 128-byte slots in the linked cycle, with `.ca` for L1 probes and `.cg` for L2/DRAM. A stride curve does not establish cache line size or associativity.
- **Units**: cycle-derived time is `cycles * 1e6 / clock_khz` and is labeled nominal nanoseconds. GiB/s uses binary bytes; GB/s uses decimal bytes.
- **Interpretation**: ILP saturation does not measure I-buffer depth, enqueue timing is not device launch latency, managed first-touch timing includes launch/synchronization, and a converged warp-sync loop may have its sync instruction removed by ptxas. Unavailable runtime properties are JSON `null` with unit `N/A`.

Validate the full RTX 5060 Ti result and both report copies:

```bash
make
./cuda_micro_bench --json
cp results/results.json docs/results.json
python3 tools/validate_results.py
python3 tools/validate_report.py
python3 tools/validate_sass.py
node tools/validate_render.js
```

`validate_results.py` checks this review's RTX 5060 Ti coverage and sanity bounds;
it is not a portable golden-data test. Selected-section runs write only those
sections to `results/results.json`; regenerate a full run before validating the
full report. WSL GPU debugging must be enabled on the Windows host to use Compute
Sanitizer; the current host's disabled interface is documented in the review.

## License

MIT
