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
