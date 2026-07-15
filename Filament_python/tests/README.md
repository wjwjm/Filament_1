# tests 目录说明

本目录包含轻量回归测试、缓存/CLI 验证和手动自检脚本。测试命令默认从仓库根目录执行。

| 类别 | 文件 |
| --- | --- |
| 基础导入与运行 | `test_sanity.py` |
| 入射条件归一化 | `test_beam_input_modes.py` |
| LUT 缓存与 CLI | `test_ion_lut_cache_reuse.py`、`test_validate_ion_lut_cli.py` |
| 电离拆分等价性 | `test_ionization_split_equivalence.py` |
| 手动自检/基准 | `ionization_selfcheck_min.py`、`minimal_run.py`、`benchmark_ion_rate_eval.py` |

最低检查：

```powershell
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
```

电离手动自检需要以 `Filament_python` 为工作目录：

```powershell
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = (Resolve-Path Filament_python)
  python Filament_python/tests/ionization_selfcheck_min.py
} finally {
  $env:PYTHONPATH = $previousPythonPath
}
```

`minimal_run.py` 固定加载 `khz_config.json`，可能触发高分辨率计算；它不替代 `config_ref.json` 的轻量 smoke test。运行入口或配置加载改动后，优先执行：

```powershell
python Filament_python/test_run.py --cfg Filament_python/config_ref.json --out Filament_python/test_smoke.npz
```

验证结束后删除临时结果，勿将 `.npz`、`.mat`、缓存或基准输出提交。
