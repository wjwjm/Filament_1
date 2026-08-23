# KHz_filament 核心包说明

本包负责从配置加载到传播、诊断和多脉冲慢时演化。运行链路为：

```text
cli.py -> confio.py / config_normalize.py -> runner.py -> propagate.py -> diagnostics.py
```

## 模块职责

| 任务 | 优先文件 | 责任边界 |
| --- | --- | --- |
| CLI 与启动参数 | `cli.py` | 解析 `cfg`、`--out`、`--dtype`；不放物理逻辑 |
| 配置 | `config.py`、`config_schema.py`、`config_normalize.py`、`confio.py` | 默认值、字段定义、历史兼容、派生量与读取 |
| 仿真编排 | `runner.py` | 网格、初场、透镜相位、多脉冲循环 |
| 主传播 | `propagate.py` | 线性半步 -> 非线性整步 -> 线性半步 |
| 线性物理 | `linear.py`、`linear_full.py`、`air_dispersion.py` | UPPE/paraxial/BK-NEE 传播因子 |
| 非线性与慢时项 | `nonlinear.py`、`raman.py`、`heat.py` | Kerr、电离耦合、拉曼/吸收、`dn_gas` |
| 诊断与摘要 | `diagnostics.py`、`summary.py` | `.npz` 字段和终端摘要 |
| 电离 | `ionization/` | 模型、LUT、rate registry、runtime 接口 |

## 配置与输出

- 配置默认值以 `config.py` 为准，允许 JSON/YAML/TOML；字段解释见 [Config_explain.md](Config_explain.md)。
- `config_normalize.py` 是历史字段兼容、幅值反推和物种归一化的唯一入口；不要在 `cli.py` 或物理模块复制兼容逻辑。
- `diagnostics.py` 中的输出字段被 MATLAB 后处理依赖。新增或重命名字段时，必须同步检查 `summary.py`、`Filament_python/matlab/` 和说明文档。
- 当保存 `rho_onaxis_t_z` 时，`runner.py` 同时保存 `t_axis`（秒），供 Python/Matlab 的 z–t 密度图使用；保留历史字段 `t` 以兼容既有读取脚本。
- Phase 1 非线性可观测性：每个 `z_axis` 记录都会保存电子 Kerr、旋转 Raman、Raman 卷积、等离子体相位/折射率、离化/IB/Raman 吸收和能量收支的历史数组。运行会额外生成 `<out stem>.diagnostic_report.json`（例如 `khzfil_out.npz` 对应 `khzfil_out.diagnostic_report.json`）；其中包含字段物理意义、数据来源、单位、建议用途与自动一致性检查结果。旧字段不改名，旧 NPZ 读取脚本无需调整。
- 独立反馈开关：`propagation.use_electronic_kerr`、`use_raman_phase`、`use_raman_full_operator`、`use_plasma_phase`、`use_ionization_loss`、`use_raman_absorption` 与既有 `use_self_steepening` 可单独门控传播反馈。`use_raman_full_operator` 仅显式启用完整 Isaacs Eq. (27) 算子，默认关闭；旧配置省略该字段时严格沿用既有 split Raman 语义。
- Raman `operator_mode="full_isaacs_eq27"` 保持旧边界：只把 rotational `D[I_R A]` 作为复场子算子，电子 Kerr 仍为 scalar phase/shock。新模式 `operator_mode="full_isaacs_eq27_complete"` 为显式 opt-in 的 combined `D[(n2 I+n_R I_R)A]`，启用时电子与 rotational 项都不再进入 scalar Kerr/shock；该模式不改变默认值、Raman 参数、电离或等离子体路径。

## 焦区窗口坐标

启用 `propagation.limit_focus_window=true` 且 `window_halfwidth_m>0` 时：

1. `runner.py` 先线性预推进到 `z_start`；
2. 随后的 `propagate_one_pulse(...)` 使用局部传播长度 `z_end-z_start`；
3. 传入的 `focus_center_m` 会转换为 `focus_center_m-z_start`；
4. 返回的 `z_axis` 从 0 开始。恢复绝对坐标时使用 `z_abs=z_local+z_start`。

## 修改约束

- 物理核心改动必须保持 Strang 分裂思想、能量诊断和 CPU/GPU 后端一致性。
- 配置字段增删要同步 `config.py`、`config_schema.py`、`config_normalize.py`、示例配置和相关 README。
- 新增诊断要同步保存字段、摘要、MATLAB 兼容性和字段单位说明。
- 修改电离模型前先阅读 [ionization/README.md](ionization/README.md)。

## 最低验证

```powershell
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
```

修改运行入口或配置读取时，再使用 `Filament_python/config_ref.json` 进行小网格运行。更完整的运行与调参说明见 [../README.md](../README.md)。
