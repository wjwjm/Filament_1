# Stage 1：单脉冲 40 fs / 120 fs 比较

Stage 1 比较相同峰值功率 `P0_peak=17 GW` 下的 40 fs 和 120 fs 单脉冲成丝。`run.Npulses=1` 与 `beam.energy_J=null` 由阶段定义强制验证；脉冲能量会随脉宽变化。

```bash
cd Filament_python
python submit_stage.py --spec stages/stage1_single_pulse_optimization.json
```

结果位于 `outputs/single_pulse_filament_optimization/<run_id>/`。两个 GPU case 无依赖提交；后处理使用 `afterok` 自动提交，并申请一张 GPU 节点资源以获得节点 CPU，但不执行 GPU 计算。最终查看 `reports/stage1_report.md`、`comparison/` 和各 case 的 `figures/`。

预览提交计划而不写文件或调用 Slurm：

```bash
python submit_stage.py --spec stages/stage1_single_pulse_optimization.json --run-id stage1_dry_run --dry-run
```

Stage 2 尚未定义。本阶段只产生比较证据；没有定义优化目标和排序规则时，报告不会声明 40 fs 或 120 fs 最优。
