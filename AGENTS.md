# AGENTS.md（仓库级工作约束）

## 1. 项目目的
本项目用于高重频（kHz）激光在空气中的成丝（filamentation）数值仿真，关注：
- 线性传播（UPPE / paraxial）；
- 非线性效应（Kerr、自陡峭、拉曼、电离、等离子体相位与吸收）；
- 脉冲间热/密度慢时扩散累积。

核心目标是获得稳定、可复现实验趋势的传播诊断数据（如 `I_max_z`、`U_z`、`rho_onaxis_max_z`、`w_mom_z`、`fwhm_*` 等）。

## 2. 传播过程约束（开发与调参时必须保持）
1. 数值流程应维持“线性半步 -> 非线性整步 -> 线性半步”的分裂思想。
2. 新增物理项时应保证：
   - 不破坏现有能量诊断输出；
   - 可被配置开关关闭；
   - 在 CPU 与 GPU 后端行为一致（允许小数值差异）。
3. 修改传播核心（`propagate.py` / `linear*.py` / `nonlinear.py` / `ionization.py` / `raman.py`）后，必须至少做一次快速可运行检查。

## 3. 工作区读写边界（防误操作）
1. 仅允许在本仓库工作区内进行读写操作：`/workspace/Filament_1/**`。
2. 禁止改动系统路径、用户主目录下与本项目无关文件。
3. 删除/重命名文件前先确认引用关系，优先小步修改。
4. 任何批量改动前先保存可回滚状态（git status 清晰可追踪）。

## 4. 修改后快速检查（最低要求）
每次代码/配置改动后至少执行以下快速检查：
1. 语法检查：`python -m compileall Filament_python/KHz_filament`
2. 基础导入测试：`pytest -q Filament_python/tests/test_sanity.py`
3. 若改动了运行入口或配置加载，增加一次最小运行（可用小网格或最短路径）以验证不崩溃。

如环境受限（无 GPU / 无某依赖），需在提交信息中注明限制与替代检查。

## 5. 超算（HPC）相关约束
1. 不在登录节点直接长时间运行大规模仿真。
2. 提交作业前明确资源参数：GPU 数、CPU 线程、内存、运行时长。
3. 线程数应与作业参数一致（OMP/MKL/OPENBLAS 等环境变量保持一致）。
4. 大网格作业前先进行小网格 smoke test，确认参数与输出字段正确。
5. 对显存敏感任务优先使用分块/因子化选项（如 `full_linear_factorize`、`chunk_pixels` 等）避免 OOM。

## 6. 仿真结果的合理性约束（Sanity Envelope）
以下为“报警阈值/合理性检查”，用于识别明显数值失稳，不是严格物理定律：
1. `U_z`：整体应有限且无无故爆炸增长；若相对初值增长超过 10%（无增益机制下）需重点排查。
2. `I_max_z`：应出现可解释的聚焦/成丝峰值；若出现非物理级跳变（相邻步数十倍）应回查步长与裁剪参数。
3. `rho_onaxis_max_z`：通常应小于或接近中性粒子密度量级上限（空气约 `~1e25 m^-3`），超过需视为异常。
4. `w_mom_z`：应随聚焦先减后增（或平台），若出现剧烈锯齿通常意味着步长或边界处理问题。
5. `fwhm_plasma_z` / `fwhm_fluence_z`：应为正且连续变化，若频繁为 0 或 NaN 需检查诊断计算及阈值。

## 7. 提交要求
1. 提交前确保 `git status` 干净且改动目的单一。
2. 提交信息需说明：
   - 改动范围；
   - 快速检查结果；
   - 若有环境限制，明确说明。

## 8. 常见问题到代码位置的映射
当用户以自然语言描述问题时，agent 应优先按下列映射定位文件。

### 8.1 电子密度偏低 / 偏高

优先检查：
- `Filament_python/KHz_filament/ionization/`
- `Filament_python/KHz_filament/nonlinear.py`
- `Filament_python/KHz_filament/config.py`
- `Filament_python/KHz_filament/config_normalize.py`
- `Filament_python/config*.json`

重点排查：
- 电场与强度单位转换；
- `time_mode` 与 `integrator`；
- `species[*].rate`、`Ip_eV`、`Zeff`、`fraction`；
- `I_cap`、`W_cap`、`W_scale`；
- LUT 是否命中缓存、是否使用了预期 reference model。

### 8.2 成丝位置提前 / 延后

优先检查：
- `Filament_python/KHz_filament/runner.py`
- `Filament_python/KHz_filament/propagate.py`
- `Filament_python/KHz_filament/linear*.py`
- `Filament_python/KHz_filament/nonlinear.py`
- `Filament_python/config*.json`

重点排查：
- 初始能量或 `E0_peak` 反推；
- `w0`、`tau_fwhm`、`focal_length`；
- 薄透镜相位符号；
- `z_max`、`dz`、`focus_window_step`、`limit_focus_window`；
- Kerr、等离子体散焦、电离损耗是否同时被修改。

### 8.3 程序数值爆炸 / NaN / energy sentinel

优先检查：
- `Filament_python/KHz_filament/propagate.py`
- `Filament_python/KHz_filament/nonlinear.py`
- `Filament_python/KHz_filament/linear*.py`
- `Filament_python/KHz_filament/diagnostics.py`
- `Filament_python/config*.json`

重点排查：
- `dz` 是否过大；
- 非线性相位或吸收是否过强；
- FFT 轴、频率轴、传播因子是否被修改；
- 边界窗口是否过小；
- dtype、GPU/CPU 后端是否行为不一致。

### 8.4 LUT 构建慢 / 缓存未复用 / 速率模型不一致

优先检查：
- `Filament_python/KHz_filament/ionization/lut.py`
- `Filament_python/KHz_filament/ionization/rate_registry.py`
- `Filament_python/KHz_filament/ionization/runtime.py`
- `Filament_python/tools/build_ion_lut_cache.py`
- `Filament_python/tools/validate_ion_lut_runtime.py`

重点排查：
- `rate_table` 配置是否启用；
- `reuse_cache`、`force_rebuild`、`cache_dir`；
- LUT 签名是否因参数变化而失配；
- runtime evaluator 与 reference evaluator 是否匹配。

### 8.5 输出字段缺失 / MATLAB 后处理失败

优先检查：
- `Filament_python/KHz_filament/diagnostics.py`
- `Filament_python/KHz_filament/summary.py`
- `Filament_python/matlab/diagnose_khzfil_out.m`
- `Filament_python/matlab/compare_khzfil_out.m`

重点排查：
- `.npz` 中保存字段名是否变化；
- 诊断量维度是否与 MATLAB 脚本假设一致；
- 是否改变了 `z_axis` 的局部坐标 / 绝对坐标含义。

### 8.6 运行速度过慢 / 显存不足

优先检查：
- `Filament_python/KHz_filament/propagate.py`
- `Filament_python/KHz_filament/linear*.py`
- `Filament_python/KHz_filament/ionization/`
- `Filament_python/KHz_filament/raman.py`
- `Filament_python/config*.json`

重点排查：
- 网格规模 `Nx, Ny, Nt`；
- `full_linear_factorize`；
- `chunk_pixels`；
- LUT 是否启用；
- 输出频率和保存字段是否过多。

## 9. 不应自动修改或提交的内容
除非用户明确要求，agent 不应修改或提交以下内容：

- 大型仿真结果文件：`*.npz`、`*.npy`、`*.mat`、`*.h5`、`*.hdf5`；
- 缓存目录：`cache/`；
- 输出目录：`outputs/`、`figures/`；
- 参考文献 PDF：`references/papers/*.pdf`；
- 临时日志、调试输出、系统生成文件；
- 与当前任务无关的配置文件和历史结果。

如果任务确实需要更新上述内容，必须在修改摘要中说明原因、文件大小影响和是否可复现。

## 10. 修改某类功能时的同步更新要求
### 10.1 新增或修改配置字段

必须同步检查：
- `Filament_python/KHz_filament/config.py`
- `Filament_python/KHz_filament/config_schema.py`
- `Filament_python/KHz_filament/config_normalize.py`
- 示例配置 `Filament_python/config*.json`
- 相关 README 说明

### 10.2 新增物理模型或非线性项

必须同步检查：
- 是否有配置开关；
- 是否保持 CPU/GPU 后端一致；
- 是否影响能量诊断；
- 是否需要新增 sanity test；
- 是否需要更新 `Filament_python/KHz_filament/README.md`。

### 10.3 新增诊断输出

必须同步检查：
- `diagnostics.py` 中字段计算；
- `.npz` 保存字段；
- `summary.py` 是否需要显示；
- MATLAB 后处理脚本是否需要兼容；
- README 中是否说明字段含义和单位。

### 10.4 修改电离模型

必须同步检查：
- `ionization/models_*.py`；
- `ionization/rate_registry.py`；
- `ionization/runtime.py`；
- `ionization/lut.py`；
- LUT 验证工具；
- 最小测试或 selfcheck。
