# Filament_python 目录说明

该目录包含仿真主程序、配置、测试与工具脚本。

- `KHz_filament/`：核心 Python 包（传播、非线性、电离、配置与运行编排）。
- `tools/`：工程工具脚本（如 LUT 构建与验证）。
- `tests/`：pytest 与最小自检脚本。
- `matlab/`：MATLAB 后处理脚本。
- `*.json`：示例/参考配置。
- `sub*.sh`：集群提交脚本。

建议阅读顺序：
1. `KHz_filament/READMD.md`
2. `KHz_filament/ionization/READMD.md`
3. `tools/READMD.md`
4. `tests/READMD.md`
