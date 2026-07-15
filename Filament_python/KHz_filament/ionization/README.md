# ionization 子包说明

本子包将电离实现分为“模型 / 注册表 / LUT / runtime”四层，目标是在不改变传播主循环接口的前提下，支持可验证的模型切换与缓存复用。

| 文件 | 职责 |
| --- | --- |
| `models_ppt.py` | PPT/Talebpour reference evaluator |
| `models_popruzhenko.py` | Popruzhenko 原子 reference evaluator |
| `rate_registry.py` | 支持的 rate、历史别名与移除项 |
| `lut.py` | LUT 签名、构建、缓存和插值 |
| `runtime.py` | `make_Wfunc` 与传播阶段的运行时接口 |
| `common.py` | 单位、裁剪和数值安全工具 |

## 当前 rate 约定

- 推荐 runtime：`ppt_talebpour_i_lut`、`popruzhenko_atom_i_lut`。
- reference 对照：`ppt_talebpour_i_full_reference`、`popruzhenko_atom_i_full_reference`。
- 回归兼容：`ppt_talebpour_i_legacy`、`popruzhenko_atom_i_legacy`。
- `ppt_talebpour_i_full` 和 `popruzhenko_atom_i_full` 是历史别名，会分别映射到对应的 `*_full_reference`。
- `ppt_e`、`ppt_i`、`adk_e`、`powerlaw`、`mpa` 等已移除；多光子近似使用 `mpa_fact`。

N2/O2 使用原子模型时属于 atomic proxy，不能当作严格分子模型解释。物种参数、`Ip_eV_eff`、`Zeff`、`fraction`、`time_mode` 和限幅项会直接影响电子密度量级。

## 修改模型时的同步项

1. 在 `models_*.py` 实现或修改 reference evaluator，明确输入/输出单位。
2. 在 `rate_registry.py` 注册 rate、别名、模型族和 LUT 支持情况。
3. 若支持 LUT，在 `lut.py` 中确保物理参数和采样/参考精度进入缓存签名。
4. 在 `runtime.py` 保持主循环调用接口稳定；不要把模型选择散落到 `nonlinear.py`。
5. 同步更新 [运行指南](../../README.md)、本文件、测试或 LUT 验证命令。

## 验证命令

```powershell
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = (Resolve-Path Filament_python)
  python Filament_python/tests/ionization_selfcheck_min.py
} finally {
  $env:PYTHONPATH = $previousPythonPath
}
python Filament_python/tools/validate_ion_lut_runtime.py --config Filament_python/khz_config_lut.json --outdir Filament_python/lut_validation
```

`khz_config_lut.json` 是当前可运行的 LUT 配置；验证可能读取或构建 `cache/rate_tables`，不要将缓存纳入提交。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 电子密度偏低/偏高 | `Ip_eV`、`Ip_eV_eff`、`Zeff`、`fraction`、`time_mode`、`I_cap`、`W_cap`、强度单位 |
| LUT 与 reference 偏差大 | 强度采样范围、`interp_mode`、reference 参数与缓存签名 |
| 缓存未复用 | `rate_table`、`reuse_cache`、`force_rebuild`、`cache_dir` 和签名日志 |
| runtime 过慢 | LUT 是否启用、`cycle_avg_samples`、是否重复构建表 |
