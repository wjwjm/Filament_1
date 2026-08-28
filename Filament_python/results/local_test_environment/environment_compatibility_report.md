# 环境兼容性报告：HPC `Filament_python` vs 本地 `filament-local-test`

生成时间：2026-08-27
数据来源：
- HPC 指纹：`hpc_environment_fingerprint.json`（2026-08-27 人工只读采集，登录节点 ln07）
- 本地指纹：`local_environment_fingerprint.json`已于 2026-08-27 重建并验证

## 定位声明

本地环境 **仅用于** CPU 软件测试、模块导入、配置加载测试。
**不承诺** 等价于完整 HPC 运行环境。
GPU/CuPy 与真实传播验证的唯一权威环境仍是 **HPC scvi806@NC-N50R5 的 `Filament_python` 环境**。

## 对比表

| 项目 | HPC (Filament_python) | Local (filament-local-test) | 状态 |
|---|---|---|---|
| Python | 3.13.7 (conda-forge, GCC 14.3.0, linux-64) | 3.13.7 (conda-forge, win-64) | **match** |
| NumPy | 2.3.3 | 2.3.3 | **match** |
| SciPy | 1.16.3 | 1.16.3 | **match** |
| pytest | **unavailable（未安装）** | 9.1.1 | local_only |
| CuPy | 13.6.0 | 未安装（Windows 无兼容 CUDA 配置，不强装） | remote_gpu_only |
| OS | Linux (NC-N50R5 登录节点 ln07) | Windows | expected difference |

## 说明

### pytest：HPC 缺失

HPC 的 Filament_python 环境中 `import pytest` 报 ModuleNotFoundError。
这是预期内情况：定向单元测试本来就只在本地跑；HPC 用于 GPU/CuPy/传播运行验证，
依赖 HPC 现有环境即可，无需在 HPC 上补装 pytest（避免污染生产环境）。

### CuPy：remote_gpu_only

Windows 本地未安装 CUDA/CuPy wheel，按任务边界不安装、不降级 CUDA、不改驱动。
GPU/CuPy 行为真实性只能由 HPC 验证。

### 平台差异提示

- 本地 Windows conda-forge + OpenBLAS 与 HPC Linux OpenBLAS 数值行为允许存在浮点级小差异；
  科学数值等价性仍以 HPC 结果为唯一判据。
- `KHz_filament/__init__.py` 设计为惰性导入（不拉 heavy deps），故 import 模式在最小 CPU 环境可跑通。

## 结论

| 结论项 | 判定 |
|---|---|
| Python 主次版本一致 | ✅ |
| NumPy 完全一致 | ✅ |
| pytest 可用（仅本地） | ✅ local_only |
| CuPy 仅 HPC 验证 | ✅ remote_gpu_only |
| 未声称完整等价环境 | ✅ |

## 重建后验收

- backend：独立 NumPy `complex128`、reshape、64 点 FFT/IFFT 探针通过；最大重构误差 `1.588822e-14`。
- import：从仓库根目录和外部目录调用均通过；入口自动固定 `Filament_python` 模块路径。
- sanity：`1 passed`。
- targeted：通过受控入口运行六组指定测试，共 `56 passed, 3 skipped`。
- compileall：使用显式解释器完成，退出码 `0`。
- `python -s -B -m pip check`：`No broken requirements found.`

## 后端修复说明

原 MKL 变体在第一个真实全 Isaacs Raman 用例的 `numpy.linalg.norm` 调用中触发
Windows `0xc06d007f` 原生崩溃。按预定回退先进行 OpenBLAS solver dry-run，再以
`libopenblas 0.3.34`、`libblas/libcblas/liblapack *openblas` 重建；重建后 backend、
sanity 与当前 56 项 targeted 均通过，另有 3 项平台相关 skip，未修改 Raman 或生产传播代码。

入口在所有模式中设置 `PYTHONNOUSERSITE=1` 并使用 `-s`，避免用户级 site-packages
污染。CuPy 未安装，状态仍为 `remote_gpu_only`；未运行 full pytest、生产传播或 Slurm。

## 已弃用环境审计（2026-08-27）

在重建前，原 `C:\Users\wangj\.conda\envs\filament-local-test` 被判定为不可继续用于
本地测试：`conda list` 将 `scipy 1.18.0` 标记为 `pypi_0`，但 `conda list --explicit`
仍列出 `scipy-1.18.0-py313he51e9a2_0.conda`，说明包来源/元数据不一致。该环境的
BLAS/LAPACK 是 MKL 变体。

Windows Application 事件日志还记录了该解释器的 APPCRASH（异常码
`0xc06d007f`）：2026-08-27 16:31:14、16:36:11、16:37:25（UTC+08:00）。不保留
崩溃 dump；仅保留这一审计摘要。此环境将仅按任务授权删除并以纯 Conda-forge
环境重建，不能据此作出 Raman 或物理模型结论。

### 纯 Conda-forge 重建后的 MKL 回退失败

纯 Conda-forge 环境已验证 Python 3.13.7、NumPy 2.3.3、SciPy 1.16.3、pytest
9.1.1 与隔离 `pip check`；`conda-meta` 中 SciPy 的 URL、构建和 SHA 均来自
conda-forge。2026-08-27 18:46（UTC+08:00）的基础 complex128/FFT/IFFT 探针通过，
但同日的定向测试在第一个真实全 Isaacs Raman 用例中再次崩溃，退出码
`-1066598273`（Windows `0xc06d007f`）。Python 栈定位为
`numpy.linalg.norm` → `raman.isaacs_raman_stage` → Raman 全算子子步。

因此本地 MKL 变体不能作为可信 CPU 测试后端；该失败不是 Raman 物理结论，也未
触发生产代码修改。后续仅允许先验证并使用 Conda-forge OpenBLAS 变体；在该变体
完成前不再运行 targeted。
