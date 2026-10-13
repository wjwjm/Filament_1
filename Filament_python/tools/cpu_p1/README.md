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

