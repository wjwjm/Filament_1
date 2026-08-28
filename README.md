# Filament_1

用于 kHz 高重频激光在空气中的成丝（filamentation）数值仿真。核心实现覆盖线性传播、Kerr/自陡峭/拉曼/电离与等离子体效应，以及多脉冲热-密度慢时累积。

## 快速开始

在仓库根目录安装依赖并先运行轻量配置：

```powershell
pip install -r Filament_python/requirements.txt
python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json
```

启用 GPU 时，在 PowerShell 中使用：

```powershell
$env:UPPE_USE_GPU = "1"
python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json
```

程序默认写出 `khzfil_out.npz`；可通过 `--out` 指定结果路径。结果文件、`cache/` 与临时日志均不应提交。

## 配置档选择

| 配置 | 用途 | 特点 |
| --- | --- | --- |
| `Filament_python/config_ref.json` | 本地烟雾测试与基准 | 小网格、短传播距离、N2/O2 reference evaluator |
| `Filament_python/khz_config.json` | 常规高分辨率运行 | 512×512×384、BK-NEE、LUT runtime |
| `Filament_python/khz_config_lut.json` | 预生成/复用 Talebpour LUT | 启用 `rate_table` 和焦区窗口 |

从 `config_ref.json` 派生新配置；每次只修改少量参数，并先在小网格上确认 `U_z`、`I_max_z`、`rho_onaxis_max_z`、`w_mom_z` 和 `fwhm_*` 的趋势合理。

## 运行链路

```text
cli.py
  -> confio.py / config_normalize.py（读取、校验、派生量）
  -> runner.py（网格、初场、多脉冲编排）
  -> propagate.py（线性半步 -> 非线性整步 -> 线性半步）
  -> diagnostics.py / summary.py（诊断与 .npz 输出）
```

- `KHz_filament/linear*.py`：UPPE、paraxial 与 BK-NEE 线性算子。
- `KHz_filament/nonlinear.py`、`raman.py`、`heat.py`：非线性、拉曼/吸收和脉冲间慢时演化。
- `KHz_filament/ionization/`：电离模型、LUT 和运行时接口。

## 目录导航

```text
Filament_1/
├─ AGENTS.md                         # 仓库工作约束与检查要求
├─ README.md                         # 本页：总览、启动、版本核验
├─ Filament_python/
│  ├─ README.md                      # 运行、配置、输出与集群脚本
│  ├─ config_ref.json                # 轻量示例配置
│  ├─ khz_config.json                # 常规高分辨率配置
│  ├─ khz_config_lut.json            # LUT 配置
│  ├─ KHz_filament/
│  │  ├─ README.md                   # 核心包职责与修改边界
│  │  ├─ Config_explain.md           # 配置字段说明
│  │  └─ ionization/README.md        # 电离模型与 LUT 说明
│  ├─ tools/README.md                # LUT 构建与验证工具
│  ├─ tests/README.md                # 回归与自检脚本
│  └─ matlab/README.md               # MATLAB 后处理
└─ references/
   ├─ README.md
   └─ papers/                        # 参考文献 PDF；不自动改动
```

## 三端 campaign 管理

新运行的跨端身份由 `campaign_id`、执行 Git SHA、配置 SHA256 和产物清单
SHA256 组成。代码仍在 `D:\Filament_1` 中开发；完整本地派生结果放在被 Git
忽略的 `.artifacts/<campaign_id>/`，不会覆盖仓库内容。只有显式 allowlist
发布的 JSON/CSV/Markdown/图片等小型证据进入
`results/campaigns/<campaign_id>/`。

```powershell
python tools/campaign/manage.py init 20260825_demo_case_v01
python tools/campaign/manage.py check 20260825_demo_case_v01 --level lite
python tools/campaign/manage.py publish-plan 20260825_demo_case_v01 --allow metrics/*.csv
```

工具和分级检查说明见
[三端 campaign 管理规则](docs/project_management/three_end_campaign_management.md)
和 [campaign 工具说明](tools/campaign/README.md)。HPC 原始数据和完整运行
证据仍由 HPC 保留；GitHub 只接收去环境化配置和精选证据。

自 2026-08-25 起，HPC 新任务的唯一项目管理根为
`/data/run01/scvi806/user_Wangjimin/projects/Filament_1`。新 staging、campaign、
cache、archive 和 quarantine 均从该命名空间解析；账号根下的旧
`Filament_1` 仅作为 `legacy_compatibility_root` 保留，不得再用于启动新任务。
机器可读路径见
[`configs/project_management/hpc_namespace.json`](configs/project_management/hpc_namespace.json)，
阶段一证据摘要见
[`docs/project_management/2026-08-25_hpc_namespace_phase1_cutover.md`](docs/project_management/2026-08-25_hpc_namespace_phase1_cutover.md)。
第二阶段第一批的四个正式 legacy 运行已进入 `legacy/runs/<campaign_id>/`；迁移
路径、manifest SHA256 和 quarantine 状态见
[`configs/project_management/hpc_legacy_relocation_batch1.json`](configs/project_management/hpc_legacy_relocation_batch1.json)
及
[`第二阶段第一批迁移报告`](docs/project_management/2026-08-25_hpc_legacy_relocation_batch1.md)。

详细入口：

- [运行与参数指南](Filament_python/README.md)
- [核心包说明](Filament_python/KHz_filament/README.md)
- [配置字段说明](Filament_python/KHz_filament/Config_explain.md)
- [电离子包说明](Filament_python/KHz_filament/ionization/README.md)
- [工具、测试和 MATLAB 说明](Filament_python/tools/README.md)
- [Sol–Luna/HPC 执行手册](docs/experience/sol_luna_hpc_execution_playbook.md)
- [Isaacs Eq.27 C2 执行复盘](docs/experience/2026-08-22_isaacs_eq27_c2_postmortem.md)

## 最小检查

本地测试必须使用仓库外的专用 Conda 环境
`C:\Users\wangj\.conda\envs\filament-local-test`，不要直接调用裸 `python` 或 `pytest`。
统一入口会自动固定模块路径并隔离用户级 site-packages：

```powershell
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode import
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode backend
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode sanity
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode targeted
```

先运行 `backend`；若 backend、import 或 sanity 失败，不要继续 targeted 或真实 Raman 传播。
`targeted` 只运行指定的轻量测试，不是 full pytest。CuPy/GPU 与真实传播仍只能由 HPC
`scvi806` 环境验证。

```powershell
$py = 'C:\Users\wangj\.conda\envs\filament-local-test\python.exe'
& $py -s -B -m compileall Filament_python/KHz_filament
```

进行一次轻量运行时，建议显式给出临时输出文件；验证后删除该文件：

```powershell
python Filament_python/test_run.py --cfg Filament_python/config_ref.json --out Filament_python/doc_smoke.npz
```

若修改了传播核心、配置加载或电离模块，还应遵循 [AGENTS.md](AGENTS.md) 中的额外检查要求。

## Stage 1 自动比较

从 `Filament_python` 目录执行 `python submit_stage.py --spec stages/stage1_single_pulse_optimization.json`，可提交 40 fs 与 120 fs 的单脉冲、等峰值功率成丝比较。运行产物统一位于 `Filament_python/outputs/`，不提交到 GitHub。

## 本地—超算版本核验（只读）

GPU 成丝任务使用 SSH target `scvi-hpc`（远端身份 `scvi806@NC-N50R5`）；远端新任务源码位于
`/data/run01/scvi806/user_Wangjimin/projects/Filament_1/source/staging/<campaign_id>/Filament_1_<short_sha>/`，
运行证据位于
`/data/run01/scvi806/user_Wangjimin/projects/Filament_1/campaigns/<campaign_id>/`。
先在本地检查当前提交和 GitHub 分支：

```powershell
git rev-parse HEAD
git branch --show-current
git status --short
git ls-remote origin refs/heads/main
```

随后通过配置好的 SSH alias 执行远端只读查询。先用 `ssh -G scvi-hpc` 检查
本机映射，再核验远端身份；包含中文路径、管道、重定向、正则、命令替换或多层
引号的操作必须改用 `Filament_python/tools/hpc_ops/Invoke-SshRemoteScript.ps1`
的参数数组和脚本上传方式；先使用 `-DryRun`，再运行只读 `hpc_preflight.sh`。
简单、无变量展开的只读查询仍可直接执行：

```powershell
ssh -G scvi-hpc
ssh -o BatchMode=yes scvi-hpc "whoami; pwd; hostname"
```

```bash
cd /data/run01/scvi806/user_Wangjimin/projects/Filament_1/source/staging/<campaign_id>/Filament_1_<short_sha>
git rev-parse HEAD
git branch --show-current
git status --short
git log -1 --format='%ad %s' --date=iso-strict
exit
```

若任一工作区存在未提交改动或未跟踪运行产物，仅记录状态并先审查差异；不要直接执行 `git pull`、`git reset`、覆盖上传或清理命令。
旧路径 `/data/run01/scvi806/user_Wangjimin/Filament_1` 的绝对路径引用保持冻结，
只用于历史读取和路径兼容。

## 工作区边界

- 代码和文档改动不自动包含 `*.npz`、`*.npy`、`*.mat`、`*.h5`、`cache/`、`outputs/`、`figures/` 或参考 PDF。
- 大网格任务先做小网格 smoke test；不要在登录节点直接运行长时仿真。
- 修改物理核心时必须保持 Strang 分裂、能量诊断与 CPU/GPU 行为的一致性。
