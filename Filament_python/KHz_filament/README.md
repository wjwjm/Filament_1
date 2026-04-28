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

## 修改边界与常见入口
本目录是仿真核心包。agent 修改时应优先按任务类型定位，不要跨模块做无关重构。

| 任务类型 | 优先文件 | 说明 |
|---|---|---|
| 命令行入口 / 参数传入 | `cli.py` | 只处理启动参数与入口调用，不应放复杂物理逻辑 |
| 仿真编排 | `runner.py` | 建网格、建初场、调用传播器、多脉冲循环 |
| 配置默认值 | `config.py` | dataclass 默认值与物理/数值默认参数 |
| 配置结构约束 | `config_schema.py` | 配置字段结构和校验相关逻辑 |
| 配置兼容与派生 | `config_normalize.py` | 旧字段兼容、互斥检查、派生量计算 |
| 配置文件读取 | `confio.py` | JSON/YAML/TOML 读取与标准化入口 |
| 主传播循环 | `propagate.py` | Strang splitting 主循环，避免无关重构 |
| 线性传播 | `linear*.py` | UPPE/paraxial/bk_nee 传播因子和频域操作 |
| 非线性物理项 | `nonlinear.py` | Kerr、电离、等离子体相位/吸收等 |
| Raman / 吸收 | `raman.py` | 延迟 Raman 响应和相关吸收模型 |
| 慢时热/密度通道 | `heat.py` | 脉冲间热扩散和 `dn_gas` 演化 |
| 诊断输出 | `diagnostics.py` | `I_max_z`、`U_z`、`rho_*`、`fwhm_*` 等 |
| 运行摘要 | `summary.py` | 输出摘要，不应承载核心物理计算 |
| 电离模型 | `ionization/` | 电离模型、LUT、runtime 接口 |

修改原则：
- 只改参数含义或默认值时，优先改配置相关文件，不要改传播主循环。
- 只改输出字段时，优先改 `diagnostics.py` 和保存逻辑，不要改物理算子。
- 只改电离速率时，优先在 `ionization/` 子包内完成，不要把模型公式写入 `nonlinear.py`。
- 只改运行摘要时，优先改 `summary.py`，不要改变 `.npz` 数据结构。

## agent 最小阅读顺序
当 agent 需要修改本目录代码时，建议按以下顺序阅读：

1. 仓库根目录 `AGENTS.md`；
2. 根目录 `README.md`；
3. `Filament_python/README.md`；
4. 本文件；
5. 与任务直接相关的源文件；
6. 相关测试或最小运行脚本。

如果任务涉及电离模型，必须额外阅读 `Filament_python/KHz_filament/ionization/README.md`。
