# tools 目录说明

本目录存放独立工程工具，不参与主传播循环。所有命令均从仓库根目录执行。

| 工具 | 用途 |
| --- | --- |
| `build_ion_lut_cache.py` | 按配置只构建/复用电离 LUT 缓存 |
| `validate_ion_lut.py` | 验证 LUT 与 reference evaluator 的误差 |
| `validate_ion_lut_runtime.py` | 验证传播运行时使用的 LUT evaluator |

推荐使用仓库现有 LUT 配置：

```powershell
python Filament_python/tools/build_ion_lut_cache.py --config Filament_python/khz_config_lut.json
python Filament_python/tools/validate_ion_lut_runtime.py --config Filament_python/khz_config_lut.json --outdir Filament_python/lut_validation
```

工具会读写 `cache/rate_tables`；缓存是可再生成的运行产物，不应提交。模型、采样范围、插值方式或 reference 精度变化后，应重新检查缓存签名和验证误差。

相关说明：[电离子包](../KHz_filament/ionization/README.md) 与 [运行指南](../README.md)。
