# micro_arch 数据补全清单

本清单来自对 `micro_arch`（zartbot 的 SM_120 微架构文档）中所有表格/数据项的梳理。请将 `zartbot/micro_arch` 仓库克隆到本仓库同级目录。
目标是：为每一项可测量的数据，在 `cuda_micro_bench` 里写对应的 CUDA/PTX 微基准；完成一项就在前面打勾。

> 说明：
> - `[x]` 已在本 repo 中实现并输出到 `results/results.json`。
> - `[ ]` 尚未实现，或当前 CUDA Runtime / 驱动无法直接暴露，需要额外手段（已在括号标注）。
> - 带 `N/A` 的项表示文档中是定性描述或仅在特定芯片（GB202）上才有的参数，本卡（RTX 5060 Ti / GB206）上可能不同。

---

## 1. 拓扑与芯片参数 (Topology / Chip Parameters)

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 1.1 | Compute Capability | :white_check_mark: | `device_info` |
| 1.2 | SM count | :white_check_mark: | `topology.sm_count` |
| 1.3 | GPC count | :x: | Runtime 不直接暴露；可尝试 NVML/PCIe 拓扑或从 SM ID 分布推断 |
| 1.4 | SMs per GPC / TPC 结构 | :x: | 同上，需结合芯片型号数据库 |
| 1.5 | SMSPs per SM | :white_check_mark: | `topology.smsps_per_sm` |
| 1.6 | Warps per SMSP (12) | :white_check_mark: | `topology.warps_per_smsp` |
| 1.7 | Max warps / SM | :white_check_mark: | `topology.max_warps_per_sm` |
| 1.8 | Max threads / SM | :white_check_mark: | `topology.max_threads_per_sm` |
| 1.9 | Max threads / block | :white_check_mark: | `topology.max_threads_per_block` |
| 1.10 | Max blocks / SM | :white_check_mark: | `topology.max_blocks_per_sm` / `occupancy` |
| 1.11 | Registers / SM | :white_check_mark: | `topology.registers_per_sm` |
| 1.12 | Shared Memory / SM | :white_check_mark: | `topology.shared_memory_per_sm_kb` |
| 1.13 | Shared Memory banks | :white_check_mark: | `topology.shared_memory_banks` |
| 1.14 | Shared Memory bank width | :white_check_mark: | `topology.shared_memory_bank_width_bytes` |
| 1.15 | L2 cache size | :white_check_mark: | `topology.l2_cache_size_mb` |
| 1.16 | L2 partition count | :white_check_mark: | `topology.l2_partition_count`（当前环境无法查询，输出 0/N/A） |
| 1.17 | Memory controller count | :white_check_mark: | `topology.memory_controller_count`（当前环境无法查询，输出 0/N/A） |
| 1.18 | Memory bus width | :white_check_mark: | `memory_bus_width_bits` |
| 1.19 | Core / Memory clock | :white_check_mark: | `clock.core_clock_mhz` / `memory_clock_mhz` |
| 1.20 | Warp-to-SMSP 静态绑定 (round-robin) | :white_check_mark: | `topology.warp_to_smsp_*`（性能阶梯验证） |
| 1.21 | Context switch overhead (0 cycles) | :x: | 单卡 WSL 无精确测量手段 |
| 1.22 | I-buffer depth (~4 entries/warp) | :white_check_mark: | `pipeline_depth.inferred_ibuffer_entries_*` |

---

## 2. 内存子系统 (Memory Subsystem)

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 2.1 | L1 latency | :white_check_mark: | `memory.l1_latency_cyc`（pointer-chase） |
| 2.2 | L2 latency | :white_check_mark: | `memory.l2_latency_cyc` |
| 2.3 | DRAM latency | :white_check_mark: | `memory.dram_latency_cyc` |
| 2.4 | DRAM read bandwidth | :white_check_mark: | `memory.dram_read_bw_gbps` |
| 2.5 | L2 aggregate bandwidth | :white_check_mark: | `memory_more.l2_aggregate_bw_gibps` |
| 2.6 | L1 per-warp bandwidth (B/cyc) | :white_check_mark: | `memory_more.l1_per_warp_bw_*` |
| 2.7 | Cache line size | :white_check_mark: | `memory_more.detected_cache_line_bytes` / `stride_latency_*` |
| 2.8 | L1 cache associativity / capacity cliff | :white_check_mark: | `memory_cache.l1_chase_*` |
| 2.9 | Shared memory latency | :white_check_mark: | `memory.shared_memory_latency_cyc` |
| 2.10 | Shared memory bandwidth | :white_check_mark: | `memory_shared.bandwidth_gibps` |
| 2.11 | Bank conflict (stride 1/17/32) | :white_check_mark: | `memory_shared.bank_conflict_stride_*` |
| 2.12 | Coalescing behavior | :white_check_mark: | `memory_more.coalesce_stride_*` |
| 2.13 | TMA latency / throughput | :white_check_mark: | `tma.tma_1d_load_latency_cyc`, `tma.tma_g2s_load_bw_gibps` |
| 2.14 | `cp.async` 异步拷贝 latency | :white_check_mark: | `memory_more.async_copy_16b_latency_cyc` |
| 2.15 | DSMEM read latency | :x: | 本卡 cluster DSMEM 远程访问报 `unspecified launch failure`，已优雅跳过 |

---

## 3. SM / 流水线 / 调度 (SM Microarchitecture & Pipeline)

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 3.1 | FMA pipeline latency (FFMA 4 cyc) | :white_check_mark: | `instructions.ffma_latency_cyc` |
| 3.1 | FMA pipeline latency (FFMA 4 cyc) | :white_check_mark: | `instructions.ffma_latency_cyc` |
| 3.2 | FFMA pipeline depth / ILP saturation | :white_check_mark: | `pipeline_depth.ffma_*` |
| 3.3 | IMAD pipeline depth / ILP saturation | :white_check_mark: | `pipeline_depth.imad_*` |
| 3.4 | INT32 add/mul latency | :white_check_mark: | `instructions.int32_*_latency_cyc` |
| 3.5 | IMAD / IADD3 / LEA / IDP4A latency | :white_check_mark: | `instructions_extended.*` |
| 3.6 | FP64 (DFMA/DADD/DMUL) latency | :white_check_mark: | `instructions.dfma_latency_cyc` / `fp64_add_latency_cyc` |
| 3.7 | MUFU RSQRT/SIN/COS/EX2/LG2 latency | :white_check_mark: | `instructions.*_latency_cyc` |
| 3.8 | MUFU RCP / TANH latency | :white_check_mark: | `instructions.rcp_latency_cyc` / `instructions.tanh_latency_cyc` |
| 3.9 | Integer division `div.u32` latency | :white_check_mark: | `instructions_more.div_u32_latency_cyc` |
| 3.10 | INT64 add/mul latency | :white_check_mark: | `instructions_more.add_u64_latency_cyc` / `mul_u64_latency_cyc` |
| 3.11 | SIMD integer (`__vadd2`, `__vadd4`, `dp4a`) | :white_check_mark: | `instructions_more.vadd2_latency_cyc` / `vadd4_latency_cyc` / `instructions_extended.idp4a_latency_cycles` |
| 3.12 | FP16x2 / BF16x2 packed latency | :white_check_mark: | `instructions_more.fma_f16x2_latency_cyc` / `fma_bf16x2_latency_cyc` |
| 3.13 | FP32x2 packed latency | :white_check_mark: | `instructions_more.fma_f32x2_latency_cyc` |
| 3.14 | Logic / shift / PRMT latency | :white_check_mark: | `instructions_more.lop3/shl/prmt_latency_cyc` |
| 3.15 | Type conversion latency (f32/f64/u32/s32/u64) | :white_check_mark: | `instructions_more.cvt_*_latency_cyc` |
| 3.16 | Pipe-to-pipe co-issue overhead | :white_check_mark: | `coissue.fma_imad_*` |
| 3.17 | ILP saturation curve (1/2/4/8/16 chains) | :white_check_mark: | `pipeline_depth.ffma_ilp*`, `pipeline_depth.imad_ilp*` |
| 3.18 | Per-SMSP IPC ceiling / aggregate IPC | :white_check_mark: | `topology.warp_scaling_*` 显示 staircase |
| 3.19 | RF bank conflict penalty | :x: | 需强制同奇偶寄存器访问 |
| 3.20 | RFC (Register File Cache) depth | :x: | 需变长度寄存器链复用测试 |

---

## 4. Tensor Core

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 4.1 | MMA m16n8k8 (TF32) latency/throughput | :white_check_mark: | `tensor_core.*` |
| 4.2 | FP16 m16n8k16 latency/throughput | :white_check_mark: | `tensor_core_extended.fp16_*` |
| 4.3 | BF16 m16n8k16 latency/throughput | :white_check_mark: | `tensor_core_extended.bf16_*` |
| 4.4 | FP8 m16n8k32 latency/throughput | :white_check_mark: | `tensor_core_extended.fp8_*` |
| 4.5 | INT8 m16n8k32 latency/throughput | :white_check_mark: | `tensor_core_extended.int8_*` |
| 4.6 | FP64 DMMA m8n8k4 latency/throughput | :white_check_mark: | `tensor_core_more.dmma_m8n8k4_*` |
| 4.7 | 更多 MMA shape | :white_check_mark: | `tensor_core_more.mma_m16n8k4_tf32_*`（TF32 m16n8k16 为非法 PTX；改用 m16n8k4） |
| 4.8 | FP6/FP4 / MX block_scale MMA | :white_check_mark: | `tensor_core_more.fp6_fp4_mma_supported=false`（本 CUDA/PTX 不支持） |
| 4.9 | wgmma async MMA | :white_check_mark: | `tensor_core_more.wgmma_supported=false`（SM_120 不暴露） |
| 4.10 | MMA + LDG/LDS/FFMA overlap | :white_check_mark: | `tensor_core_more.mma_ffma_overlap_ratio` |

---

## 5. 同步、屏障与原子 (Barriers, Sync, Atomics)

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 5.1 | `__syncthreads()` latency | :white_check_mark: | `barriers.syncthreads_*_cycles` |
| 5.2 | `__syncwarp()` latency | :white_check_mark: | `barriers.syncwarp_cycles` |
| 5.3 | `fence.acq_rel.cta/gpu/sys` latency | :white_check_mark: | `barriers.fence_*_cycles` |
| 5.4 | Named barrier (`bar.sync`) latency | :white_check_mark: | `barriers.named_barrier_cycles` |
| 5.5 | Cluster barrier (`barrier.cluster`) latency | :white_check_mark: | `cluster.cluster_barrier_*_cyc` |
| 5.6 | mbarrier arrive/wait latency | :white_check_mark: | `barriers.mbarrier_cycles` |
| 5.7 | Global atomicAdd latency/throughput | :white_check_mark: | `atomics.global_atomic_add_*` |
| 5.8 | Shared atomicAdd throughput | :white_check_mark: | `atomics.shared_atomic_add_gops` |
| 5.9 | Atomic CAS / Exch / Min / Max latency | :white_check_mark: | `atomics.global_atomic_{cas,exch,min,max}_latency_cyc` |
| 5.10 | Reduction (`red.sync`) latency/throughput | :white_check_mark: | `atomics.block_reduce_latency_cyc`（共享内存树规约代理） |

---

## 6. Warp 调度与控制流 (Warp Scheduler & Control Flow)

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 6.1 | Warp shuffle latency/throughput | :white_check_mark: | `warp_shuffle.shfl_idx_*` |
| 6.2 | Warp vote (`vote.ballot`) latency | :white_check_mark: | `warp_collectives.vote_ballot_latency_cycles` |
| 6.3 | `elect.sync` latency | :white_check_mark: | `warp_collectives.elect_sync_latency_cycles` |
| 6.4 | `redux.sync.add` latency | :white_check_mark: | `warp_collectives.redux_add_latency_cycles` |
| 6.5 | Branch divergence cost (2/4/8/16/32-way) | :white_check_mark: | `warp_collectives.divergence_*way_cyc_per_iter` |
| 6.6 | Branch prediction warm/cold cost | :white_check_mark: | `warp_collectives.branch_cold/warm_cyc_per_iter` |
| 6.7 | Predicated divergence split-ratio | :white_check_mark: | `warp_collectives.pred_split_*_cyc_per_iter` |
| 6.8 | IPC vs warps/SM scaling | :white_check_mark: | `warp_collectives.ffma/iadd3_ipc_w*` |
| 6.9 | Scheduler policy (CGGTY) inference | :x: | 无公开 PTX 接口，需事件计数器或芯片内部信息 |

---

## 7. 系统/PCIe/启动 (System / PCIe / Launch)

| # | 测试项 | 状态 | 备注 / 当前输出键 |
|---|--------|------|-------------------|
| 7.1 | Kernel launch latency (cold/warm) | :white_check_mark: | `pcie.kernel_launch_*_us` |
| 7.2 | HtoD / DtoH pinned bandwidth | :white_check_mark: | `pcie.htod_bandwidth_GiB_s` |
| 7.3 | Peer-to-peer / multi-GPU (if存在) | :white_check_mark: | `pcie.device_count` / `pcie.p2p_available`（本机 1 卡） |
| 7.4 | Unified Memory / page fault latency | :white_check_mark: | `pcie.um_page_fault_latency_us` |

---

## 8. 本轮已实现的模块

- `src/topology_extended.cu` — SMSP / warp scaling staircase、warp-to-SMSP round-robin 验证。
- `src/pipeline_depth.cu` — FFMA / IMAD ILP sweep、pipeline depth、I-buffer 深度推断。
- `src/instructions_more.cu` — logic/shift/prmt/div/int64/conversions，packed FP16/BF16/FP32，SIMD `__vadd2` / `__vadd4`。
- `src/tensor_more.cu` — FP64 DMMA m8n8k4、TF32 m16n8k4、MMA+FFMA overlap、wgmma/FP6-FP4 支持性探测。
- `src/warp_collectives.cu` — `vote.ballot.sync`、`elect.sync`、`redux.sync.add`、divergence、branch warm/cold、predicated divergence、IPC vs warps。
- `src/memory_more.cu` — cache-line、coalescing、async copy 16 B、L1 per-warp BW、L2 aggregate BW。
- `src/coissue.cu` — FMA+IMAD pipe-to-pipe co-issue overlap。
- `src/tma.cu` — `cp.async.bulk.tensor` 1D 16 B load latency（mbarrier + `__grid_constant__`）；G2S bulk-copy 带宽（CUDA C++ `cuda::device::memcpy_async_tx` + transaction barrier，16 KiB × 4 级流水，host 侧计时）。
- `src/atomics.cu` — `atomicAdd` throughput/latency、`atomicCAS/Exch/Min/Max` latency、block reduce。
- `src/barriers.cu` — `__syncthreads`、`bar.sync`、`mbarrier`、fence latencies。
- `src/cluster.cu` — cluster launch 与 cluster barrier latency。
- `src/pcie.cu` — kernel launch、PCIe BW、P2P/UM 探测。
- `src/device_info.cu` — 对不暴露的拓扑值输出 N/A。

## 9. 仍受环境/硬件限制或无法直接测量的项

这些项已在文档中记录，当前无法在本卡/WSL 环境下直接测量或需要芯片内部信息：

- Context switch overhead：需要精细的 warp 交替计时或事件计数器，单卡 WSL 无法稳定测量。
- RF bank conflict penalty / RFC depth：需要强制同 bank 寄存器访问或芯片内部事件，无法通过 PTX 直接探测。
- Scheduler policy (CGGTY) inference：无公开 PTX 接口或性能计数器暴露 tiebreak 逻辑。
- DSMEM 跨 block 远程读在本环境报 `unspecified launch failure`，已优雅跳过。
- GPC/TPC 数量、L2 partition count、memory controller count 在本环境 Runtime/NVML 均不暴露，已输出 N/A。

---

## 9. 当前已生成的结果文件

- `results/results.json`：所有已实现的微基准结果。
- `results/report.html`：用 zartbot 风格渲染的在线报告。
- `tools/validate_report.py`：校验 report 引用的键都在 JSON 中。

运行方式：

```bash
make
./cuda_micro_bench --json
python3 tools/validate_report.py
```
