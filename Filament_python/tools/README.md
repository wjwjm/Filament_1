# tools 目录说明

本目录存放独立工程工具，不参与主传播循环。所有命令均从仓库根目录执行。

| 工具 | 用途 |
| --- | --- |
| `build_ion_lut_cache.py` | 按配置只构建/复用电离 LUT 缓存 |
| `validate_ion_lut.py` | 验证 LUT 与 reference evaluator 的误差 |
| `validate_ion_lut_runtime.py` | 验证传播运行时使用的 LUT evaluator |
| `build_nonlinear_ablation_configs.py` | 从 FT90 基准配置和声明式覆盖项生成非线性消融配置与 manifest；该工具不含 `sbatch` 或作业提交功能 |
| `validate_nonlinear_switch_isolation.py` | 运行 CPU 小网格开关隔离/全模型回归检查，只写 JSON 报告并删除临时 NPZ |
| `validate_ionization_time_integrator.py` | 复用生产电离求解器执行 0D 时间包络扫描，输出每个 species 的 `I(t)`、`W(t)`、`rho(t)` 和稳定性基础指标 |

推荐使用仓库现有 LUT 配置：

```powershell
python Filament_python/tools/build_ion_lut_cache.py --config Filament_python/khz_config_lut.json
python Filament_python/tools/validate_ion_lut_runtime.py --config Filament_python/khz_config_lut.json --outdir Filament_python/lut_validation
```

工具会读写 `cache/rate_tables`；缓存是可再生成的运行产物，不应提交。模型、采样范围、插值方式或 reference 精度变化后，应重新检查缓存签名和验证误差。

相关说明：[电离子包](../KHz_filament/ionization/README.md) 与 [运行指南](../README.md)。

非线性消融配置示例（只生成 JSON 和 manifest，不运行仿真）：

```powershell
python Filament_python/tools/build_nonlinear_ablation_configs.py --out-dir Filament_python/results/nonlinear_ablation_configs/example_run
```

输入阶段说明为 `Filament_python/stages/nonlinear_ablation_stage1.json`。每个 manifest 条目固定记录基准配置及哈希、覆盖项、40/120 fs、最终有效开关、电离求解状态、预期 NPZ 文件名、代码 SHA 和生成时间。
