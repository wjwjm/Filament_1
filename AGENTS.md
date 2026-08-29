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
1. 语法检查：使用专用环境显式解释器执行 `-s -B -m compileall Filament_python/KHz_filament`
2. 基础导入测试：使用 `Filament_python/tools/run_local_tests.ps1 -Mode sanity`
3. 若改动了运行入口或配置加载，增加一次最小运行（可用小网格或最短路径）以验证不崩溃。

如环境受限（无 GPU / 无某依赖），需在提交信息中注明限制与替代检查。

### 4.1 本地测试解释器强制规则

- 本地测试必须通过 `Filament_python/tools/run_local_tests.ps1` 执行；其默认解释器为仓库外的
  `C:\Users\wangj\.conda\envs\filament-local-test\python.exe`。
- 不得用裸 `python`、裸 `pytest` 或调用者临时设置的 `PYTHONPATH` 代替该入口；入口会固定
  `Filament_python` 模块路径并设置 `PYTHONNOUSERSITE=1`，隔离用户级 site-packages。
- 定向测试前必须先运行 `-Mode backend`；backend、import 或 sanity 失败时停止，不继续运行
  Raman 全算子或 targeted 测试。
- 允许的本地测试模式为 `import`、`backend`、`sanity`、`targeted`。`targeted` 只包含任务
 规定的轻量测试集合，不等价于 full pytest。
- 该环境仅用于 Windows CPU 软件/配置测试；CuPy、GPU 和真实传播仍以 HPC `scvi806` 环境
  为唯一权威。环境指纹和兼容性记录位于 `Filament_python/results/local_test_environment/`。

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

## 11. 项目级专用子 Agent 与科学决策边界

本项目使用以下项目级 Luna 子 Agent；主 Sol Agent 负责规划、物理/数值决策和最终验收：

- `filament_mapper`：严格只读地定位代码、配置、测试、诊断、冻结基准和数据流。
- `filament_worker`：只在边界明确后执行一次最小、局部的代码修改。
- `filament_numerical_reviewer`：最高优先级的只读数值、数学约定、物理闭合和诊断审查者。
- `filament_tester`：执行低成本测试、smoke、输入审计和输出完整性检查；写权限仅用于测试临时文件、缓存、日志和明确授权的测试产物。

并发与写入规则：

1. 默认最多并行 3 个子 Agent。
2. `filament_mapper`、`filament_numerical_reviewer` 和未执行昂贵计算的 `filament_tester` 可并行。
3. 任意时刻只能有一个修改型 Agent 写入生产代码；`filament_worker` 必须串行工作，不得让多个 worker 并行修改同一工作区、模块、物理过程、配置、测试或传播脚本。

Luna 子 Agent 可查代码和调用链、找配置和测试、整理证据、检查数值闭合、执行明确修改和低成本验证。以下决定必须保留给主 Sol Agent：为什么修改、物理模型是否合理、是否接受新数值策略、是否改变模型参数或冻结基准、是否启动完整传播或提交生产 Slurm Job、如何解释与 PyCAP/文献的差异，以及最终科学结论。遇到这些问题时，子 Agent 必须停止扩展并返回证据与待决问题。

安装或调用 Agent 不得修改物理/数值模型、生产配置或生产数据，不得重新生成或覆盖冻结结果，不得混同冻结物理基准 SHA 与后续只读分析/文档 SHA，不得自行改变坐标、电子密度 onset 或 PyCAP 比较口径，也不得因测试通过而宣称已复现论文。

<!-- BEGIN FILAMENT SUBAGENT ORCHESTRATION -->

# Filament_1 Scientific-Code Subagent Orchestration

## Scope

These rules apply to the Filament_1 ultrashort-pulse filamentation simulation repository.

They supplement the global code-engineering orchestration policy.

Where this project policy is more restrictive than the global policy, this project policy takes precedence.

The repository contains scientific and numerical simulation code. A change may be syntactically correct, software-correct, and test-passing while still being numerically or physically wrong. Scientific-code changes therefore require strict separation between:

1. software correctness;
2. numerical correctness;
3. physical correctness.

The parent Sol agent retains final scientific judgment.

---

## Project-Specific Agents

Prefer these project-specific agents over their generic global equivalents when operating on Filament_1 scientific code.

| Generic role | Filament_1 preferred role |
| --- | --- |
| `repo_explorer` | `filament_mapper` |
| `scoped_worker` | `filament_worker` |
| `code_reviewer` | `filament_numerical_reviewer` for scientific/numerical changes |
| `test_runner` | `filament_tester` |
| `debug_scout` | keep using global `debug_scout` |

These are preferred roles, not a requirement to spawn every role for every task.

---

## Named-Agent Runtime Compatibility

Use the configured named Filament agents when the runtime actually supports selecting them.

Do not claim that `filament_mapper`, `filament_worker`, `filament_numerical_reviewer`, or `filament_tester` was used unless the corresponding custom role was actually selected or loaded.

If named custom-agent selection is unavailable but generic subagent spawning works:

- generic read-only children may be used for bounded investigation or review;
- generic write children may be used only when their effective permissions and task scope are sufficiently controlled;
- preserve the role boundaries described below;
- report material loss of configured model, reasoning, sandbox, or custom-role guarantees.

Do not silently treat a generic spawned thread as a loaded Filament custom agent.

---

## Parent Scientific Authority

The parent agent is responsible for deciding:

- whether a proposed physics change is justified;
- whether a numerical strategy is acceptable;
- whether a model discrepancy has been explained;
- whether a reference comparison is valid;
- whether a baseline may change;
- whether an expensive propagation run or Slurm submission is warranted;
- whether scientific evidence is sufficient for a conclusion.

Subagents must stop and return evidence plus the remaining decision to the parent before acting on any item above.

---

## Project Routing and Sequencing

1. Analysis-only scientific requests: prefer `filament_mapper`; run `filament_numerical_reviewer` in parallel when mathematical conventions, conserved quantities, phase, energy, causality, or physical closure must be assessed. Do not start `filament_worker`.
2. Bounded scientific-code changes: first map the implementation chain as needed; let exactly one `filament_worker` write each overlapping code/configuration/test area; after integration, use `filament_numerical_reviewer` and `filament_tester` independently where practical.
3. Fault diagnosis: use `debug_scout` for the reproducible technical cause and `filament_mapper` for code/configuration provenance. A worker begins only after the parent resolves the diagnosis and change boundary.
4. Low-cost validation: `filament_tester` may run syntax, unit, smoke, input-audit, and output-integrity checks. Passing software tests do not establish numerical or physical correctness.

Parallel work is restricted to independent read-only mapping, numerical review, diagnosis, and low-cost validation. Serialize all production-code writes, configuration changes that affect the same run, numerical integration, and final scientific acceptance.

---

## Non-Delegable Safety Stops

No child agent may autonomously:

- submit, cancel, alter, or monitor a production Slurm job as a decision-making action;
- launch expensive or large-grid propagation beyond an explicitly approved bounded check;
- change model defaults, frozen baselines, reference-comparison definitions, physical parameters, coordinate conventions, onset definitions, or PyCAP comparison semantics;
- overwrite simulation outputs, caches, historical result artifacts, or frozen data;
- commit, push, create a pull request, or make an external publication claim;
- treat a passing test as proof of physical reproduction or scientific validity.

For these cases, stop and return the relevant code/data provenance, numerical evidence, estimated cost or risk, and the precise parent or user decision required.

## 12. Sol–Luna 与 HPC 执行可靠性

长期流程与本次错误复盘分别记录在：

- [`docs/experience/sol_luna_hpc_execution_playbook.md`](docs/experience/sol_luna_hpc_execution_playbook.md)
- [`docs/experience/2026-08-22_isaacs_eq27_c2_postmortem.md`](docs/experience/2026-08-22_isaacs_eq27_c2_postmortem.md)
- [`docs/experience/2026-08-24_hybrid_execution_postmortem.md`](docs/experience/2026-08-24_hybrid_execution_postmortem.md)
- [`docs/experience/2026-08-24_hybrid_0p60_result_archive.md`](docs/experience/2026-08-24_hybrid_0p60_result_archive.md)

强制规则：

- Sol 保留科学决策、基线接受、commit/push、HPC staging、Slurm 提交和最终验收；Luna 必须按 `task_boundary`、`evidence`、`files_changed`、`commands_and_exit_codes`、`tests`、`unverified`、`parent_decisions` 返回。
- 模型/连接失败后，先读 Agent 状态并审计共享工作树；确认已有写入和文件所有权前，不得重启第二个 writer。
- 无变量展开的单条只读 SSH 命令才可内联；遇到中文路径、管道、重定向、正则、命令替换、heredoc 或嵌套引号，必须使用 `Filament_python/tools/hpc_ops/Invoke-SshRemoteScript.ps1` 的脚本/参数数组入口。第一次转义错误后不得继续堆叠引号。`Invoke-PappRemoteScript.ps1` 仅作为显式 Papp 回退，不得自动切换。
- HPC GitHub 连接遵循代理优先、verified bundle 回退；代理值、token、认证 URL 和真实凭据不得进入仓库或日志。preflight 失败不得创建 run/lock 或调用 `sbatch`。
- 仓库内 `Invoke-SshRemoteScript.ps1`、`Invoke-PappRemoteScript.ps1` 和 `hpc_preflight.sh` 仅允许 `scvi806`；不得把其固定根目录或 `Filament_python` 环境套用于 `t0s000727`。其他账户必须使用独立审计的入口。
- 运行时终态和诊断报告分开核验；`PENDING`、`RUNNING`、编译通过或测试通过不能替代科学验收。

<!-- BEGIN FILAMENT LEAN EXECUTION AND FAILURE CIRCUIT BREAKERS -->

## 13. 精简执行与失败熔断

- “创建算例—提交—等待—比较”、已有结果后处理、状态检查和结果同步默认为
  `L0 direct execution`。若现有 runner、`hpc_ops`、launcher、postprocessor
  和 comparison 已覆盖任务，不得新建 campaign framework、替代绘图链或重复
  provenance 系统。
- quick engineering comparison 可以复用配置、执行 SHA、诊断字段和 provenance
  均满足当前比较要求的已验收 reference。最低资格是：scheduler/运行终态已核验、
  必要诊断有限且字段/坐标/threshold 定义一致、输入物理配置除候选字段外一致，且
  执行 SHA 相同或源码差异已证明不影响当前比较并由用户接受 quick 模式。只有用户或冻结协议明确要求同 SHA、
  同节点/同 allocation 严格配对时，才允许把 reference 重跑纳入执行计划；提交前
  必须报告串/并行关系和基于历史 wall time 的预计总时长。
- postprocess-only 阶段默认禁止修改 `propagate.py`、`linear*.py`、
  `nonlinear.py`、`ionization/`、`raman.py`、生产配置和冻结结果。必须先在固定
  `Filament_python` 环境调用已有后处理入口；本地缺少 PIL/matplotlib 不是修改
  生产后处理代码或重建分析框架的理由。
- 新增或修改生产 `.sbatch`/launcher 时，必须在任何 run 目录、execution lock、
  submission lock 或 `sbatch` 副作用之前运行
  `Filament_python/tools/hpc_ops/audit_batch_entry.py`。`bash -n` 失败或 conda
  激活前出现裸 `python`/`python3` 时立即停止。
- 同一工具、阶段和错误签名第二次出现时必须熔断。第一次发生 PowerShell/WSL/
  SSH/Bash 多层转义或变量丢失后，后续命令必须改用固定脚本、参数数组或 JSON
  manifest，不得继续加引号重试。工作更新中记录错误签名和 fallback；重读工具
  契约后实质修正的调用允许执行，只有换行/引号变化的等价 payload 不算新路径。
- raw NPZ、运行目录、scheduler evidence 和已冻结比较结果保持不可覆盖。技术失败
  与科学分类分开记录；算例尚未启动时不得给出物理结论。

<!-- END FILAMENT LEAN EXECUTION AND FAILURE CIRCUIT BREAKERS -->

## 14. 跨平台 provenance 与哈希范围

- 先分类再哈希；`Filament_python/tools/hpc_ops/provenance_v2.py` 是本仓库唯一权威实现，不得复制或另建哈希语义。
- Git tracked text 必须同时记录 Git blob OID 与 canonical-LF SHA256；不得把 Windows checkout 的 raw-byte SHA256 作为新 HPC 校验值。
- external 或 binary artifact 必须记录 raw-byte SHA256；新 manifest 顶层和每条 record 都必须声明正确的 `hash_scope`。
- legacy frozen receipt、lock、manifest 及其 `.gitattributes` CRLF/binary 例外不得迁移、规范化或重写；旧 v1 只保留兼容读取。
- 新 HPC campaign 必须在创建 run directory、lock、receipt 或调用 `sbatch` 前完成严格 provenance schema、Git blob 与 canonical-LF/raw-byte 校验。

<!-- END FILAMENT SUBAGENT ORCHESTRATION -->
