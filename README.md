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

详细入口：

- [运行与参数指南](Filament_python/README.md)
- [核心包说明](Filament_python/KHz_filament/README.md)
- [配置字段说明](Filament_python/KHz_filament/Config_explain.md)
- [电离子包说明](Filament_python/KHz_filament/ionization/README.md)
- [工具、测试和 MATLAB 说明](Filament_python/tools/README.md)
- [Sol–Luna/HPC 执行手册](docs/experience/sol_luna_hpc_execution_playbook.md)
- [Isaacs Eq.27 C2 执行复盘](docs/experience/2026-08-22_isaacs_eq27_c2_postmortem.md)

## 最小检查

```powershell
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
```

进行一次轻量运行时，建议显式给出临时输出文件；验证后删除该文件：

```powershell
python Filament_python/test_run.py --cfg Filament_python/config_ref.json --out Filament_python/doc_smoke.npz
```

若修改了传播核心、配置加载或电离模块，还应遵循 [AGENTS.md](AGENTS.md) 中的额外检查要求。

## Stage 1 自动比较

从 `Filament_python` 目录执行 `python submit_stage.py --spec stages/stage1_single_pulse_optimization.json`，可提交 40 fs 与 120 fs 的单脉冲、等峰值功率成丝比较。运行产物统一位于 `Filament_python/outputs/`，不提交到 GitHub。

## 本地—超算版本核验（只读）

GPU 成丝任务使用 `scvi806@nc-n50r5`；远端项目根目录为 `/data/run01/scvi806/user_Wangjimin/Filament_1`。先在本地检查当前提交和 GitHub 分支：

```powershell
git rev-parse HEAD
git branch --show-current
git status --short
git ls-remote origin refs/heads/main
```

随后通过已有 papp_cloud 会话交互登录，再在远端执行只读 Git 查询。包含
中文路径、管道、重定向、正则、命令替换或多层引号的操作必须改用
`Filament_python/tools/hpc_ops/Invoke-PappRemoteScript.ps1` 的参数数组和
脚本上传方式；先使用 `-DryRun`，再运行只读 `hpc_preflight.sh`。简单、无
变量展开的只读查询仍可直接执行：

```powershell
wsl bash -c "~/papp_cloud/papp_cloud_linux_amd64 ssh scvi806@nc-n50r5"
```

```bash
cd /data/run01/scvi806/user_Wangjimin/Filament_1
git rev-parse HEAD
git branch --show-current
git status --short
git log -1 --format='%ad %s' --date=iso-strict
exit
```

若任一工作区存在未提交改动或未跟踪运行产物，仅记录状态并先审查差异；不要直接执行 `git pull`、`git reset`、覆盖上传或清理命令。

## 工作区边界

- 代码和文档改动不自动包含 `*.npz`、`*.npy`、`*.mat`、`*.h5`、`cache/`、`outputs/`、`figures/` 或参考 PDF。
- 大网格任务先做小网格 smoke test；不要在登录节点直接运行长时仿真。
- 修改物理核心时必须保持 Strang 分裂、能量诊断与 CPU/GPU 行为的一致性。
