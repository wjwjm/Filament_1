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

## 新增或修改电离模型的同步流程
电离子包采用“模型 / LUT / runtime”分层。新增或修改 rate model 时，不应只改一个公式文件。

必须同步检查：

1. `models_*.py`
   - 放置具体电离率公式或 reference evaluator；
   - 明确输入强度/电场单位和输出单位；
   - 避免在模型内部读取全局配置文件。

2. `rate_registry.py`
   - 注册新的 rate 名称；
   - 明确别名映射；
   - 明确模型族和是否支持 LUT；
   - 删除或替换旧 rate 名称时，应给出清晰报错或迁移提示。

3. `lut.py`
   - 如果模型支持 LUT，需要纳入缓存签名；
   - 确保物理参数、采样参数和 reference 精度参数变化时会触发重建；
   - 检查插值模式是否适合速率动态范围。

4. `runtime.py`
   - 确保 `make_Wfunc` 或等价 runtime 接口能调用新模型；
   - 保持传播主循环调用接口稳定；
   - 避免把模型选择逻辑散落到 `nonlinear.py`。

5. 文档与测试
   - 在 `Filament_python/README.md` 中补充配置示例；
   - 必要时更新本 README；
   - 添加或更新最小测试、自检脚本或 LUT 验证命令。

## 电离模型修改后的最低检查
修改电离相关代码后，至少执行：

```bash
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
PYTHONPATH=Filament_python python Filament_python/tests/ionization_selfcheck_min.py
```

如果修改了 LUT 逻辑，还应执行或说明未执行原因：

```bash
python Filament_python/tools/validate_ion_lut_runtime.py --config Filament_python/config.json
```

如果本地环境缺少依赖、GPU 或算力不足，应在提交摘要中说明实际执行了哪些替代检查。

## 电离相关常见问题定位
| 现象 | 优先检查 |
|---|---|
| 电子密度整体偏低 | `Ip_eV`、`Zeff`、`fraction`、`time_mode`、`W_scale`、强度单位 |
| 电子密度整体偏高 | `I_cap`、`W_cap`、中性粒子密度、重复计算电离、单位换算 |
| LUT 与 reference 差异大 | `lut.py`、采样范围、`interp_mode`、reference 精度参数 |
| 缓存没有复用 | `rate_table` 配置、缓存签名、`force_rebuild`、`cache_dir` |
| N2/O2 结果异常 | 是否误用 atomic proxy、`Ip_eV_eff`、`Zeff`、模型族映射 |
| runtime 很慢 | 是否未启用 LUT、`cycle_avg_samples` 是否过大、是否重复构建表 |
