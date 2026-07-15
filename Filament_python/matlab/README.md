# matlab 目录说明

该目录存放 MATLAB 后处理脚本，面向 `khzfil_out.npz` 转换后的 `.mat` 数据。

| 脚本 | 用途 |
| --- | --- |
| `diagnose_khzfil_out.m` | 单个结果文件的综合诊断 |
| `compare_khzfil_out.m` | 多个结果文件的单参数比较绘图 |

先在仓库根目录将 `.npz` 转为 `.mat`：

```powershell
python Filament_python/npz2mat.py --npz Filament_python/khzfil_out.npz --mat Filament_python/khzfil_out.mat
```

Python 节点端自动绘图（`plot_khzfil_out.py`）和本目录的 MATLAB 绘图并存：前者由 `sub.sh` 自动生成 `figures/<run_name>/`，适合只下载 PNG/JSON；后者继续用于交互分析和 `compare_khzfil_out.m` 的多结果比较。`npz2mat.py` 不会删除源 NPZ。若需要安全删除，请通过 `test_run.py --mat-dir ... --remove-npz` 运行，由它在所有启用的 PNG/JSON 和 MAT 输出都成功后统一删除。

若诊断脚本报字段缺失，先检查 `diagnostics.py` 的保存字段和 `z_axis` 的局部/绝对坐标含义；焦区窗口模式下，绝对坐标为 `z_abs = z_local + z_start`。

结果文件属于运行产物，不自动纳入提交。
