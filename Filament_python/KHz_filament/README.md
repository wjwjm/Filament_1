# KHz_filament 目录说明

该目录是仿真核心包，负责从配置加载到传播计算与诊断输出。

- `cli.py`：命令行入口。
- `runner.py`：仿真编排（读配置、建初场、调传播器）。
- `summary.py`：运行摘要输出。
- `config.py` / `config_schema.py` / `config_normalize.py` / `confio.py`：配置定义、标准化与读取。
- `propagate.py`：Strang 分裂主循环。
- `linear*.py`、`nonlinear.py`、`raman.py`、`heat.py`：物理算子。
- `diagnostics.py`：诊断计算与输出。
- `ionization/`：电离子包（模型、LUT、runtime 接口）。

如果要理解整体运行链路，建议从 `cli.py -> runner.py -> propagate.py` 追踪。

## 焦区窗口（`limit_focus_window`）坐标说明

- 当启用 `propagation.limit_focus_window=true` 且 `window_halfwidth_m>0` 时，`runner.py` 会先线性预推进到 `z_start`，再把后续传播长度改为局部窗口长度 `z_end-z_start`。
- 此时传给 `propagate_one_pulse(...)` 的 `focus_center_m` 会被转换为局部坐标：`focus_center_local_m = focus_center_m - z_start`，用于 `dz_focus` 触发判据。
- 注意：窗口模式下 `propagate_one_pulse(...)` 产出的 `z_axis` 是局部坐标（从 0 开始）；若需要绝对坐标，请使用 `z_abs = z_local + z_start`。
