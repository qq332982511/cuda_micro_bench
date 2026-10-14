# 审查证据

最终说明：[REVIEW.md](../../REVIEW.md)。

- `before.json`：开始时仓库已有的数据。
- `baseline.log` / `baseline.json`：原始代码在当前 GPU 上的全量运行。
- `build-full.log` / `make-check.log`：最终强制完整编译及目标检查。
- `final-1/2/3.log` 与同名 JSON：最终三个完整运行，分别使用无参数、`--json`、`--all`。
- `comparison.json`：最终三轮的中位数/范围与原始数据。
- `sass-audit.json`：原始/最终反汇编摘要；`tools/validate_sass.py` 检查当前实际二进制。
- `environment.json`：CUDA、驱动、GPU 和 PCIe 链路观察。
- `validation.log` / `validation.json`：最终校验输出与源码/结果哈希。
- `sanitizer-final.log`：WDDM debugger interface 未启用导致工具失败，不能作为 memcheck 通过证据。

完整 `.sass` 在本地保留但未版本化，可重新用 `cuobjdump --dump-sass cuda_micro_bench` 生成。
