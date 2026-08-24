# Hybrid 0.60 m 结果归档索引

## 定位

- 归档标签：`hybrid-0p60-validation-2026-08-24`
- 归档分支：`codex/hybrid-propagation-validation`
- 归档 commit：`ebfe5c3afbeef4583f983e8dc9258c25c8d6c980`
- 正式执行 commit：`5ce3be1e4a74eff71dee219116e9a2f29aa3b34b`
- Slurm job：`222025`，终态 `COMPLETED/0:0`
- raw NPZ：保留在 HPC，本 Git 归档不包含 raw NPZ

该索引只让后续任务定位结果，不把 Hybrid 核心实现、大型轴向 CSV、图像或
一次性 campaign 工具复制到 `main`。精确结论必须以归档标签中的机器可读文件为准。

## 最终结论

- 机械分类：`hybrid_0p60_not_supported`
- 人工最终结论：`hybrid_0p60_partially_supported_for_acceleration`
- 适用范围：显式 opt-in 的低精度加速测试；不能替代从 `z=0` 开始的严格 reference
- case wall-time speed-up：`1.5363706412x`
- 峰值电子密度相对降低：`4.1135619%`
- 电子密度峰位置延后：`0.3500044 cm`
- `rho=1e22 m^-3` onset 延后：`0.3053482 cm`
- 强度峰位置延后：`0.4099965 cm`
- 密度峰数量：`1 -> 2`
- Hybrid/reference 能量漂移：`0.0632302 / 0.0630072`

这些差异是正式结果的一部分，不得描述为数值噪声、严格物理等价或默认生产路径。

## 核心归档文件与 SHA-256

以下哈希针对标签中的原始 Git blob 字节计算，不受 Windows/POSIX 工作树换行转换影响。

| 路径（相对归档标签） | SHA-256 |
| --- | --- |
| `Filament_python/results/hybrid_propagation_validation/FINAL_REPORT_222025.md` | `68374669d29c2825655aac714b371324d449dd88073861f729b2706ed7cf08b8` |
| `Filament_python/results/hybrid_propagation_validation/final_classification_222025.json` | `440c97351fec20c2862728c3b1d7c2ac5a3f5787fa303e0277f2e9ac3e330517` |
| `Filament_python/results/hybrid_propagation_validation/scheduler_terminal_evidence_222025.json` | `9664393270e33b1ae68a0fab1c5273813d1c2614f44cde440e1f1c3229685aa5` |
| `Filament_python/results/hybrid_propagation_validation/comparison_222025/hybrid_propagation_validation_comparison.json` | `16301dfc940b750c2cb8feff03a575afe9d2e1cf4833212665c598f4fa449669` |
| `Filament_python/results/hybrid_propagation_validation/postprocess_222025/hybrid_propagation_validation_audit.json` | `3dc1d59e88a703e286823948daf08403eb1ed5ff84daf7cfa2613b2889f5d603` |
| `Filament_python/results/hybrid_propagation_validation/postprocess_222025/reference_axial.csv` | `205f6b7c0b01b1d1966862a5f639faa624d0c39b5c44920728a138f3764f488a` |
| `Filament_python/results/hybrid_propagation_validation/postprocess_222025/hybrid_axial.csv` | `a2a37cdb8b7c3aed4c2921a5eb236ad6c7f0e5d2304000d9d876442b1f16a0cf` |

两份轴向 CSV 均为 15,001 行（表头加 15,000 条传播记录）。

## 读取方法

只读查看单个结果：

```powershell
git show hybrid-0p60-validation-2026-08-24:Filament_python/results/hybrid_propagation_validation/final_classification_222025.json
```

需要同时读取多个结果或重跑现有 postprocess/compare 时，建立独立 worktree：

```powershell
git worktree add ..\Filament_1_hybrid_archive hybrid-0p60-validation-2026-08-24
```

不得在归档 worktree 中覆盖结果、重新分类或把 partial pass 写成 strict pass。
