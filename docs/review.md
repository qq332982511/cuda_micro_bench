# RTX 5060 Ti 测试代码与结论审查（2026-10-14）

已检查所有 benchmark 模块、公共计时/日志代码、原始结果和报告，修复可复现的测量与解释错误，并完成本机重测。最终证据是 `results/review/final-{1,2,3}.{log,json}`；`results/results.json`、`results/run.log` 与 `docs/results.json` 使用第三轮数据。275 个结果项包括查询值、派生值、支持性标志和实测值，不代表 275 项独立的硬件参数鉴定。

## 本机架构与解释范围

本机 CUDA Runtime 报告：RTX 5060 Ti、CC 12.0、36 SM、48 warps/SM、24 blocks/SM、65,536 个 32-bit registers/SM、100 KiB shared/SM、48 KiB 默认 shared/block、99 KiB opt-in shared/block、32 MiB L2、128-bit 显存总线。每线程 register 上限是 255；原代码用 `regsPerBlock / maxBlocksPerSM` 得到 2730，已纠正。

- [NVIDIA GPU CC 表](https://developer.nvidia.com/cuda/gpus) 将 RTX 5060 Ti 列为 12.0；[官方产品规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/) 将其归为 Blackwell，列出 4608 CUDA cores 和 448 GB/s 显存带宽。
- [RTX Blackwell 白皮书](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf) 说明 RTX Blackwell 的 FP32/INT32 资源统一。128 KiB 的 L1/shared 合并空间不等于本机可分配的 shared 容量；资源上限采用本机 Runtime 查询值，默认与 opt-in 的 block limit 分别输出。数据中心 B200/SM_100 的缓存、shared、cluster 等参数不作为本卡参数。
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 将 FP6/FP4 MMA 列为 SM_120a/f 的特性。当前 native SM_120 构建没有实现该 benchmark；报告区分 ISA 可用与已实现。WGMMA 的目标支持范围不能用“CC >= 9.0”概括。
- [共享内存 bank 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns) 给出 32 banks、每 bank 每周期 32 bits 的带宽。这提供了共享带宽的合理上限检查；允许 nominal/boost 频率差异。
- [DSMEM 编程规则](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#distributed-shared-memory) 要求所有远程访问结束后才能退出被访问的 block。修复该生命周期错误后，本机 DSMEM 正常执行。

环境：WSL2，CUDA Toolkit 13.2，NVIDIA 驱动 610.88（nvidia-smi 同时显示 UMD 610.57.01），构建使用 `-arch=native`，SASS 为 SM_120。显卡同时服务桌面，当前 PCIe 链路报告为 Gen5 x8。没有锁频或独占 GPU；这些数据描述本环境下的 kernel 成本与达到的吞吐率。

## 主要发现与修复

| 范围 | 原问题 | 最终处理 |
|---|---|---|
| 公共计时、所有 ns 输出 | `cycles * 1e9 / clock_khz` 放大 1000 倍；clock 与链结果的条件依赖可能被删除；CUDA event 调用未检查；CPU timer 使用可跳变的 wall clock | 使用 `1e6 / clock_khz`，标注 nominal ns；共同的 predicated end-clock helper；检查 event API；CPU 使用 steady_clock；SASS 核验最终时钟依赖 |
| Global memory | 4 MiB 的规则 `+64` 链被称为 DRAM；复制的读+写字节率被称为只读带宽 | 随机单环遍历所有 128-byte slots；DRAM 工作集 256 MiB（8×L2）、每样本完整遍历，实际触及 sector 至少 64 MiB（2×L2）且 `.cg`，padding 随机化，64-bit clocks；独立只读流与复制流；非零随机数据；5 次样本中位数 |
| Cache | 规则流读取/浮点累积不是 load-to-use 链；stride 同时改变 coalescing/MLP，却被用于确认 cache line；L2 键常缺失 | `.ca` 随机依赖链工作集曲线；`.cg` 8 MiB warmed L2 依赖读；校验最后 index；撤销 cache-line/associativity 确认结论 |
| Shared、STS、L1/L2 BW | C++ `asm volatile` 不禁止 ptxas 删除未使用的读取或重复写入；宣称向量读取却可能只保留最后一次 | Shared LD/ST 使用 PTX `ld.volatile.shared` / `st.volatile.shared`；缓存读取累积全部四个分量并保留 `.ca/.cg`；SASS 检查访存位于回跳循环内；字节数对应真实访问 |
| Shared latency / banks | 共享地址未转换真实 dynamic shared base；多 warp 干扰和索引算术混入；重复初始化存在竞争 | 显式 `__cvta_generic_to_shared`；单 warp、依赖地址的 self-link 读；无竞争初始化；不同 stride 测试有独立输出 |
| Scalar / packed instructions | SQRT 被称为 RSQRT；DADD 被称为 DFMA；常量乘法/ADD/RCP 等被简化，uniform ALU 与 vector ALU 混用 | 真正 RSQRT、DFMA；SFU 使用支持的 approximate FTZ 变体；EX2/LG2 通过位映射保持正常数输入并单独报告组合序列成本；线程相关操作数；吞吐率使用至少 200 ms 同 kernel 预热与 5 样本中位数；ADD 两寄存器交替链避免 IADD3 合并；易折叠循环限制展开；64-bit 输入两半均非退化 |
| Instruction / warp IPC | lane 指令数与 warp 指令数混用；整数 IPC 的结果未被有效使用，整个工作循环可消失；展开的 ADD 会合并 | 区分 warp PTX ops 与 warp SASS instructions；保留结果依赖；限制 ADD 展开；确认 timed loop 含实际 IADD/FFMA；不把 PTX 复合操作当成单条 SASS |
| Pipeline / topology | ILP 饱和被等同于 I-buffer entries；链 latency 被等同于 pipeline stages；1/4 warp 扩展被当成 round-robin 证明；sweep 索引超过 vector 范围 | 只输出依赖链成本、ILP 饱和和并行扩展比；sweep 限于单 block 的 32 warps；95% 饱和使用 aggregate rate；撤销结构深度与静态绑定结论 |
| Co-issue | FP/INT 输入常量导致 UFFMA/UIMAD；链 overlap 被解读为独立资源或同时 issue | 所有链随线程和 accumulator 变化；保留 sink；确认普通 FFMA/IMAD 路径；ratio 仅表示达到的混合成本/速率。统一 FP32/INT32 资源下仍可隐藏依赖延迟 |
| Atomics | shared 输出只分配 SM count，实际启动 4×SM blocks；unused-return ADD 被当成 RMW latency；常量更新的聚合被当成通用 atomic 指令率 | 分配正确容量并验证每 block 计数；验证全局计数；依赖返回值的 RMW ADD；RED.ADD 与 shared constant INC 按逻辑更新率单独命名，记录编译器/硬件聚合 |
| Tensor | TF32 被用作 FP16 peak；TF32 k8 latency 被命名为另一 shape；INT8 MAC 少计一次操作；sink 调用不保证终止时钟依赖 | 分离 TF32/FP16 键与 accumulation 类型；FP8 使用正常数 1.0 的 E4M3 编码；每 MMA 使用真实 logical shape；INT8 multiply+add 计两次；result-dependent end clock；DMMA/HMMA/IMMA 的实际 lowering 已检查 |
| Warp collectives / divergence | ELECT 的 `lane & 0` 无依赖；两个工作次数不同的循环被称为 branch cold/warm；32-way switch 有重复 default 路径 | ELECT 标为 repeated issue cost；两个循环按实际工作命名；32 条独立路径与 PTX body；不推断 branch predictor |
| Barriers | 收敛 warp 的 sync 被消除，却被称为 WARPSYNC latency；多个 lane 写同一 timer slot | 仅选定 lane 写 timer；保留 barrier/fence 原始成本；SASS 没有 WARPSYNC 时标为 converged syncwarp loop cost |
| PCIe / occupancy | CPU enqueue 被称为 device launch latency；UM first touch 包含 launch/sync 却被称为 page fault；occupancy 示例未改变 register pressure | 修正指标名称；保留 pinned memcpy 计时；occupancy 为当前 kernel 的 API 查询/运行结果，不声称 RF pressure sweep |
| TMA | bulk 带宽输入重复字节；热 16 B tile/wait 成本与 DRAM miss latency 直接求比 | bulk 使用随机输入；只与实际只读流比较 bandwidth；取消不等价 latency 的 ratio；记录 issue/wait 与 transfer size；launch/driver API 错误不再被吞掉或写成零延迟 |
| Cluster | rank 0 可在 rank 1 读完前退出；各 block 争写一个计时槽；猜测 max cluster=8；错误被吞掉并解释成硬件不支持 | 前后 cluster.sync，依赖远程 `.shared::cluster` 读；分别计时；查询 occupancy API 上限、验证 cluster size；CUDA 错误直接失败；DSMEM warmup + 5 样本中位数 |
| 构建、日志、报告 | header 改动可不重编译；无法写 JSON 时静默；N/A 被输出成数字 0；diagram 留有旧推断与静态旧值 | `.d` header dependencies；检查 JSON 写入；N/A 为 null；同步本地/发布目录的字段、解释和数据，撤销错误别名 |

## 最终三轮结果

三次命令分别为 `./cuda_micro_bench`、`./cuda_micro_bench --json` 和 `./cuda_micro_bench --all`。全部退出 0，没有 CUDA error、launch failure 或 benchmark skip。下表给出三轮报告值的中位数和范围；部分单项内部再使用 5 个样本的中位数。

| 项目 | 中位数 | 三轮范围 | 单位 |
|---|---:|---:|---|
| 只读 DRAM 带宽 | 419.326 | 417.759–420.650 | GB/s |
| L1 随机依赖读 | 44.032 | 44.032–44.032 | cycles |
| L2 随机依赖读 | 452.936 | 382.220–561.837 | cycles |
| DRAM 随机依赖读 | 915.321 | 902.483–915.608 | cycles |
| 共享内存依赖读 | 28.007 | 28.007–28.007 | cycles |
| 共享内存向量读带宽 | 12255.436 | 12245.831–12276.834 | GiB/s |
| L1 单 warp 向量读/累积 | 60.832 | 60.831–60.833 | nominal GiB/s |
| L2 向量读/累积 | 2038.989 | 2019.082–2050.166 | GiB/s |
| INT32 ADD 交替依赖链 | 5.201 | 5.201–5.201 | cycles |
| FP32 FMA | 23123.269 | 23116.189–23201.435 | GFLOP/s |
| INT32 MUL 链 | 4.234 | 4.234–4.234 | cycles |
| DFMA 链 | 63.540 | 63.540–63.540 | cycles |
| 32 warp IADD IPC | 2.046 | 2.046–2.046 | warp inst/cycle/SM |
| 32 warp FFMA IPC | 3.689 | 3.688–3.689 | warp inst/cycle/SM |
| DSMEM 依赖远程读 | 211.003 | 211.003–211.003 | cycles |
| INT8 MMA | 202.542 | 202.542–202.678 | TOP/s |

这些数值不能与旧结果逐列作性能提升比较：访问模式、工作集、cache policy、计数口径和 accumulation 类型都有修正。只读 DRAM 带宽低于 448 GB/s；L1、L2、DRAM 依赖成本有明确层级；共享带宽满足 bank/boost 上限；tensor 计数按真实 shape 与数据类型解释。L2 和 event 吞吐率仍受频率、地址/累积指令、桌面负载影响。源码/报告中的 latency 是整个依赖序列成本，包含编译器调度、循环和地址开销；例如未展开 RCP 的成本不能直接当成裸 MUFU latency。

## 验证与证据

- `results/review/before.json`：开始审查时仓库已有数据；`baseline.json` / `baseline.log`：原代码在当前机器的复跑。
- `results/review/build-full.log`：强制完整 native 构建；`make-check.log`：确认目标最新。
- `results/review/final-{1,2,3}.{log,json}`：最终三轮完整执行；`comparison.json`：选定指标的原值、最终中位数和范围。
- `results/review/sass-audit.json`：原始/最终 SASS 的 opcode 和计时/控制流摘要。完整 `.sass` 可由 `cuobjdump` 重新生成，未纳入版本文件。
- `tools/validate_results.py`：验证本卡完整覆盖、275 项的有限值/单位、关键物理上限、层级及错误别名的撤销。
- `tools/validate_sass.py`：验证实际二进制循环内仍有待测工作、co-issue 非 uniform 路径、ADD 没有合并、end-clock predicate 和 sync-elision 结论。
- `tools/validate_report.py`：表格、diagram、比较行字段存在，两个报告/数据副本一致。
- `tools/validate_render.js`：以真实 JSON 执行报告 JS，核对所有表格和 diagram bindings；这是渲染逻辑检查，未代替真实浏览器布局检查。

Compute Sanitizer **未通过，也未成功建立 GPU instrumentation**。当前 WSL 报 WDDM debugger interface 无法初始化、Device not supported，退出 99；日志为 `results/review/sanitizer-final.log`（`--memory --cluster --device`）。该日志中 benchmark 本身运行成功不能当成 memcheck 通过。[NVIDIA Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior) 要求 Windows host 开启 GPU 调试接口。没有修改宿主机注册表或调试配置。源码中的越界/竞争已修复，运行时指针链和 atomic 计数另有校验，但不能据此宣称完整 memcheck/racecheck 已完成。

## 保留的限制

未实现的硬件结构鉴定仍包括：I-buffer 深度、cache line/associativity、精确 scheduler policy、RF bank/RFC、独立 page-fault 开销、FP6/FP4 benchmark 和多 GPU P2P 性能。GPC/TPC、L2 partitions/controllers 没有可用 Runtime 查询结果。支持性标志、proxy/序列成本、查询值与实测架构参数已经区分，并同步修正 `MISSING_TESTS.md`。本轮验证范围是这台 RTX 5060 Ti 与当前 CUDA 13.2 toolchain，其他 ARCH/Toolkit 尚未完成同样核验。

## 字段兼容性

纠正了多个错误命名的 JSON 字段并移除了错误别名；本仓库的报告和校验工具已同步。读取旧键的外部脚本需要跟进 `results/results.json` 的新键与单位。报告的 `review.md` 副本与本文件相同。
