# tools 目录说明

该目录存放“可直接执行”的工程工具，不参与主循环传播。

- `build_ion_lut_cache.py`：按配置预生成并缓存电离 LUT。
- `validate_ion_lut_runtime.py`：对比 reference evaluator 与 LUT evaluator 的误差。

典型用途：
- 先离线构建 LUT，再运行主仿真，减少首轮开销。
- 进行模型替换/调参后，单独验证 LUT 精度。
