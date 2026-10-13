# CPU P1 FFT 与硬件诊断

此工具只运行 synthetic FFT 和节点硬件/线程信息采集，输出统一标记为 `P1_DIAGNOSTIC_ONLY`、`NOT_PRODUCTION_QUALIFIED`。它不加载 canonical 光学源、真实 HR-4 截面或传播配置，不执行光学传播，也不代表 A/B 输入已准入。

完整 FFT 输入为 C 连续 `complex128` 数组 `(384, 351, 301)`，解释为 `(slice, y, x)`；固定随机种子生成，仅用于可重复性能测试。逐 slice NumPy 与 `np.fft.fft2/ifft2(..., axes=(-2,-1))` 批量 NumPy、显式 `scipy.fft` workers 对比。NumPy 路径记录 `THREADING_NOT_SUPPORTED_BY_BASELINE`。每个组合先 warmup，再做三次独立 pair 计时，保存 forward、inverse、pair 的 wall/process CPU p50 和 nearest-rank p90，并核验往返误差、相对逐 slice NumPy reference 的 forward-spectrum max_abs/max_rel/L2，以及合成数组 SHA256。

完整模式只接受单节点 96 核、`v6_384` partition、`bscc-t` account、`normal` QoS、GPU 未分配且 `UPPE_USE_GPU=0` 的 Slurm 环境。核数依次为 1、96、12、24、48；每个算法/核数组合是单独子进程，并按可用 affinity 中检测到的物理核选择前 k 个。父进程轮询 `/proc` 进程树 RSS/CPU time，在申请内存 80% 或总运行时限达到时终止当前子进程组；warmup 与每次重复完成后立即写出 progress JSON，已完成样本会保留。sysfs 物理核拓扑不可用时完整作业阻断；核档被跳过时整体标记 `INCOMPLETE`。

依赖 Python 3、NumPy、SciPy、Linux `/proc`。正式诊断要求冻结版本 `threadpoolctl 3.6.0` 可导入；把已核验 wheel 解压到 vendor 目录，runner 会在版本检查前将该目录加入 sys.path，并记录线程库版本及实际线程池信息。`lscpu`、`numactl` 为可选采集项，缺失时明确记录 `UNAVAILABLE`。

```bash
FIXED_PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python
BENCH_ROOT=/absolute/path/to/Filament_1
OUTPUT_ROOT=/absolute/path/to/new/cpu-p1-fft-run
VENDOR_DIR=/absolute/path/to/cpu-p1/dependencies
"$FIXED_PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_fft.sbatch" --fixed-python "$FIXED_PYTHON"
bash -n "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_fft.sbatch"
sbatch --test-only "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_fft.sbatch" "$BENCH_ROOT" "$OUTPUT_ROOT" "$VENDOR_DIR"
```

只有所有前置审计成功并由任务负责人按 submission 预算授权时，才单独执行 `sbatch`。本实现不提交作业。输出路径必须不存在，父目录必须预先存在。`--smoke` 用小型 synthetic 数组做软件检查，不要求 Slurm，不产生完整形状性能结论。

输出包括 `performance_summary.csv`、逐条追加的 `benchmark_results.jsonl`、终态 `status.json`、环境/affinity/线程池 JSON，以及 lscpu、numactl、NumPy 配置和 `/proc` 文本快照；每个子进程另留 stdout/stderr。

## CPU P1 Hydro A0/A1

`cpu_p1_hydro.py` 只测冻结 `advance_hr4_single_screen` 的 CPU 单屏计算，输出均为 `P1_DIAGNOSTIC_ONLY` / `NOT_PRODUCTION_QUALIFIED`。正式诊断只接受父端审定的 48 屏历史 S5 POST/NEXT 输入清单；它不是 S4R 384 屏输入，因此不运行 384 屏强扩展，也不直接对比 S4R 服务吞吐。

A0 对真实 POST ordinal 24 连续运行两次，各 1000 步、`require_stable=True`。两次输出三场 canonical hash 必须逐项完全相同；否则记为 `CPU_REPEAT_DRIFT` 并停止 A1，同时保留误差与观测量回执。仅 A0 完整成功后才进入 A1。A1 依次执行 P=1、8、24、48、96，每 worker 处理四个完整屏任务；每个 worker 绑定一个 sysfs 验证的物理核、线程池限为 1。按全档 `n=4P` 展平任务序号：当 `n≤48` 时 ordinal=`floor(flat_task_index*48/n)`，否则 ordinal=`flat_task_index%48`。因此 P=1/8 分别覆盖 4/32 个不同屏；P=24/48/96 均复用 48 屏，统一标记 `REPEATED_REAL_INPUT_SCALING_ONLY`。完整模式只接受 raw SHA256 为 `f273cfeb2f112de62be9a787d58bc2a7c40ac44f5d7e9978c057dfb9d3aa552a` 的冻结 manifest，并核对其中 hydro 参数和网格间距。所有输入任务在每次读取时重验 NPZ 原始 SHA256、三场 canonical array hash、shape/dtype/C 布局/有限数和嵌入 metadata；每次完整任务单独落盘，超时保留已完成任务。父进程以文件承接子进程日志并监控进程树 RSS/CPU；到 80% 申请内存或 3300 秒总预算时熔断，A0 子项上限 900 秒。每档结束后才能开始下一档；任一档非 COMPLETE 即停止扩展。

节点 receipt 记录分配/affinity、`lscpu -e`、`/proc/self/status`、线程池和 complex64/complex128 NumPy FFT 返回 dtype 探针。NumPy 后端必须在导入 HR-4 前固定为 `UPPE_USE_GPU=0` 并经 `debug_backend()` 复核。任务计时拆分输入读/原始与 canonical 哈希、HR-4 kernel wall/CPU time、输出写入/哈希和完整 service；数值比较分别保留三个场的 max_abs/max_rel/L2、标量 observables、同 worker 重复值以及对应历史 NEXT 参考。比较只作诊断记录，不赋予 GPU 配对或科学准入资格。

合成 `--smoke` 只调用小数组 fake callable 验证输出与非覆盖行为，不导入/运行 HR-4，也不代表真实数据路径成功。完整模式要求 Linux `/proc`、经 `thread_siblings_list` 验证的 96 物理核、固定解释器、冻结 threadpoolctl 3.6.0 与有效 Slurm CPU allocation。

```bash
FIXED_PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python
BENCH_ROOT=/absolute/path/to/Filament_1
OUTPUT_ROOT=/absolute/path/to/new/cpu-p1-hydro-run
VENDOR_DIR=/absolute/path/to/cpu-p1/dependencies
INPUT_ROOT=/absolute/path/to/input_readonly
"$FIXED_PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_hydro.sbatch" --fixed-python "$FIXED_PYTHON"
bash -n "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_hydro.sbatch"
sbatch --test-only "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_hydro.sbatch" "$BENCH_ROOT" "$OUTPUT_ROOT" "$VENDOR_DIR" "$INPUT_ROOT"
```

这里仅展示提交前检查命令；此代码变更不提交 Slurm 作业。任务需另行授权后才能实际提交。输出目录必须不存在且其父目录预先存在；最大工作集为 708 个完整 1000 步屏任务，证据保留在任务输出目录。

