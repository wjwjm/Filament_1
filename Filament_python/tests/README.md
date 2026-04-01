# tests 目录说明

该目录存放基础回归测试与轻量验证脚本。

- `test_sanity.py`：最小可导入/可运行检查。
- `test_beam_input_modes.py`：入射参数模式兼容检查。
- `test_ion_lut_cache_reuse.py`：LUT 缓存复用行为测试。
- `test_ionization_split_equivalence.py`：电离拆分后的等价性测试。
- `test_validate_ion_lut_cli.py`：LUT 验证工具 CLI 测试。
- `benchmark_ion_rate_eval.py`：速率评估基准。
- `ionization_selfcheck_min.py` / `minimal_run.py`：快速自检脚本。

建议先跑：`pytest -q Filament_python/tests/test_sanity.py`。
