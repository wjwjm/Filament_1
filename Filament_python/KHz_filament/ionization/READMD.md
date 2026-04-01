# ionization 子包说明

该子包实现电离相关功能，采用“模型 / LUT / 运行时”分层。

- `models_ppt.py`：PPT/Talebpour 分支实现。
- `models_popruzhenko.py`：Popruzhenko 原子分支实现。
- `lut.py`：LUT 构建、签名、缓存与插值。
- `runtime.py`：传播主循环调用接口（如 `make_Wfunc`、密度演化）。
- `rate_registry.py`：rate 名称映射、别名与模型族管理。
- `common.py`：数值安全与通用工具。
- `__init__.py`：对外导出稳定 API。

目标：在不改变主循环接口的前提下，支持模型扩展与缓存复用。
