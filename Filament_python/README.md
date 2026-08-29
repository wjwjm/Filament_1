# Filament_python 运行与参数指南

本目录包含可运行配置、主程序、后处理入口和集群脚本。命令默认从仓库根目录 `Filament_1` 执行，除非命令另有说明。

## 安装与运行

```powershell
pip install -r Filament_python/requirements.txt
python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json
```

GPU 运行：

```powershell
$env:UPPE_USE_GPU = "1"
python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json
```

CLI 支持 `--out` 指定结果文件和 `--dtype fp32|fp64` 指定计算精度：

```powershell
python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json --dtype fp32 --out Filament_python/khzfil_out.npz
```

## 配置档与适用场景

| 文件 | 用途 | 不应直接用于 |
| --- | --- | --- |
| `config_ref.json` | CPU/GPU 烟雾测试、基准与调试 | 长距离高分辨率生产运行 |
| `khz_config.json` | 标准高分辨率 BK-NEE 运行 | 未验证参数的首次测试 |
| `khz_config_lut.json` | 生成与复用 Talebpour LUT、焦区窗口运行 | 不启用 LUT 的 reference 对照 |

配置按 `grid`、`beam`、`propagation`、`ionization`、`heat`、`run`、`raman` 七个顶层段组织。字段含义见 [配置字段说明](KHz_filament/Config_explain.md)。

加载时，`config_normalize.py` 会：

1. 读取 JSON/YAML/TOML 并统一为标准结构；
2. 拒绝同时指定 `beam.energy_J` 与 `beam.P0_peak`；
3. 在 `E0_peak=0` 时由能量或峰值功率反推电场；
4. 为缺省 `grid.Twin` 补充 `8 * tau_fwhm`；
5. 归一化 `species.fraction` 并映射历史 rate 别名。

## LUT 构建与验证

`khz_config_lut.json` 是本仓库的 LUT 示例配置。先只构建缓存：

```powershell
python Filament_python/tools/build_ion_lut_cache.py --config Filament_python/khz_config_lut.json
python Filament_python/tools/validate_ion_lut_runtime.py --config Filament_python/khz_config_lut.json --outdir Filament_python/lut_validation
```

在 N50 GPU 队列预生成 LUT 时：

```bash
cd Filament_python
CFG=khz_config_lut.json sbatch sub_lut.sh
```

`sub_lut.sh` 已声明一张 GPU、8 个 CPU 线程和 30 分钟时限。该分区按 GPU 绑定内存；不要额外写入 `--mem` 或 `--mem-per-cpu`。运行完整仿真请检查 `sub.sh` 的 `CFG`、`OUT`、`DTYPE`、后处理开关，以及 Slurm CPU 线程数与 OMP/MKL/OPENBLAS 设置是否一致。

## 输出、Python 自动绘图与 MATLAB 后处理

主程序输出 `.npz`，包含坐标和诊断量，例如 `U_z`、`I_max_z`、`rho_onaxis_max_z`、`w_mom_z`、`fwhm_plasma_z` 与 `fwhm_fluence_z`。

Slurm 任务可在计算节点直接从 NPZ 写出六张 PNG 和 `diagnostic_summary.json`，再转换 MATLAB 文件。Python 自动绘图与本地 MATLAB 绘图是两条并存路径：前者适合只下载轻量的 `figures/<run_name>/`，后者保留 MATLAB 的交互分析和多结果比较能力。

推荐提交方式（从 `Filament_python` 目录）：

```bash
cd Filament_python
CFG=khz_config.json \
OUT=run_001.npz \
MAT_DIR=matlab保存数据 \
GENERATE_FIGURES=1 \
FIG_DIR=figures/run_001 \
FIG_SELECT=all \
FIG_DPI=200 \
Z_SHIFT_CM=-20 \
CONVERT_TO_MAT=1 \
REMOVE_NPZ=1 \
sbatch sub.sh
```

成功时输出为 `matlab保存数据/run_001.mat`、`figures/run_001/*.png` 和 `figures/run_001/diagnostic_summary.json`。`REMOVE_NPZ=1` 只有在已成功生成所有启用的 PNG/JSON 且 MAT 转换成功时才删除 `run_001.npz`；任一步失败都会保留 NPZ 并让任务以非零状态结束。

`sub.sh` 的环境变量如下：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CFG` | `khz_config.json` | 仿真配置 |
| `OUT` | `khzfil_out.npz` | 原始诊断 NPZ |
| `DTYPE` | `fp32` | 计算精度 |
| `GENERATE_FIGURES` | `1` | `1` 时生成节点端 PNG/JSON |
| `FIG_DIR` | `figures` | PNG 与摘要目录 |
| `FIG_SELECT` | `all` | `all` 或 `intensity,plasma,beam,energy,fwhm,rho_tz` |
| `FIG_DPI` | `200` | PNG DPI |
| `Z_SHIFT_CM` | `0` | 绘图 z 轴平移（cm） |
| `CONVERT_TO_MAT` | `1` | `1` 时写 MATLAB 文件 |
| `MAT_DIR` / `MAT_NAME` | `matlab保存数据` / 空 | MATLAB 输出目录 / 可选文件名 |
| `REMOVE_NPZ` | `1` | 仅在 MAT 与所有启用后处理成功后删除 NPZ |

只下载自动诊断结果的示例：

```bash
scp -r <user>@<cluster>:<remote-run-dir>/figures/run_001/ ./figures/run_001/
```

不经 Slurm 单独转换为 MATLAB 文件：

```powershell
python Filament_python/npz2mat.py --npz Filament_python/khzfil_out.npz --mat Filament_python/khzfil_out.mat
```

`npz2mat.py` 从不删除源 NPZ；需要安全清理时通过 `test_run.py --mat-dir ... --remove-npz`，由运行入口在所有后处理成功后统一执行。MATLAB 绘图入口见 [matlab/README.md](matlab/README.md)。

## 诊断与数值合理性

- `U_z` 在无增益机制下不应无故增长超过初值的约 10%。
- `I_max_z` 应有可解释的聚焦/成丝峰；相邻步数十倍跳变应检查步长和裁剪。
- `rho_onaxis_max_z` 不应明显超过空气中性粒子密度量级（约 `1e25 m^-3`）。
- `w_mom_z`、`fwhm_plasma_z`、`fwhm_fluence_z` 应连续；频繁 0、NaN 或强锯齿通常表示数值或诊断问题。

问题定位优先级见仓库根目录 [AGENTS.md](../AGENTS.md)。

## 快速检查

```powershell
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode import
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode backend
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode sanity
D:\Filament_1\Filament_python\tools\run_local_tests.ps1 -Mode targeted
$py = 'C:\Users\wangj\.conda\envs\filament-local-test\python.exe'
& $py -s -B -m compileall D:\Filament_1\Filament_python\KHz_filament
```

本地测试入口默认使用仓库外的 `filament-local-test` Conda 环境，并自动设置
`Filament_python` 模块路径、`PYTHONNOUSERSITE=1` 和显式 `python -m pytest`。
不要使用裸 `python`/`pytest` 运行本地测试，也不要在 backend 或 sanity 失败后继续
执行 targeted。该环境仅用于 Windows CPU 软件测试；CuPy、GPU 和真实传播仍以 HPC
`scvi806` 环境为权威。环境指纹见
`Filament_python/results/local_test_environment/`。

`tests/minimal_run.py` 会从 `Filament_python` 工作目录加载 `khz_config.json`，因此它不是低成本 smoke test；首次验证优先使用 `test_run.py --cfg config_ref.json`。

## 目录导航

- [核心包职责](KHz_filament/README.md)
- [电离模型与 LUT](KHz_filament/ionization/README.md)
- [可执行工具](tools/README.md)
- [回归测试与自检](tests/README.md)
- [MATLAB 后处理](matlab/README.md)

输出、缓存、运行日志和参考文献 PDF 不属于常规代码提交内容。

## Stage 1：40 fs / 120 fs 单脉冲比较

Stage 1 固定为 `run.Npulses=1`、`P0_peak=17 GW`、`energy_J=null` 的等峰值功率比较；仅改变 40 fs 与 120 fs 脉宽，不会自动宣布任一脉宽最优。

```bash
cd Filament_python
python submit_stage.py --spec stages/stage1_single_pulse_optimization.json
```

每个运行写入 `outputs/single_pulse_filament_optimization/<run_id>/`。两个 case 完成后，自动提交依赖它们 `afterok` 的后处理作业，生成比较图和 `reports/stage1_report.md`。
