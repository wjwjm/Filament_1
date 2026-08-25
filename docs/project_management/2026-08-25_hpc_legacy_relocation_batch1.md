# Filament_1 HPC legacy relocation：第二阶段第一批

第二阶段第一批迁移已于 2026-08-25 完成，仅包含四个已经通过 namespace archive
receipt 的正式 legacy campaign。批次 receipt SHA256 为
`0095137fe36ba375c6723a94b09cf3f02b3779d0f39aa340aca82465ce457401`。

## 迁移结果

| Campaign | 文件数 | 总字节数 | Source manifest SHA256 |
| --- | ---: | ---: | --- |
| `20260820_historical_fr_mixture_final_v01` | 2,003 | 222,589,452 | `5afbe710cb62bb7ec9b1b40bd7f4097978e3989156545f1facab4e4b87a26f52` |
| `20260821_raman_off_kerr085_final_v01` | 2,047 | 241,795,212 | `cb932e812e578fb7f4546a2cff2d3cb1224111f55e565d1c48168172f3b06242` |
| `20260822_isaacs_complete_eq27_c2_v01` | 3,707 | 674,099,910 | `4987b09fca83bdaaf6ae14414f04086b1c36ba62a0d81e06cc964c9e2d215f08` |
| `20260824_hybrid_propagation_0p60_v01` | 5,535 | 1,068,790,480 | `ccc2cf795a98ae0f998a87d036d975ad98217a3d147f7cb5cfc9bb7d0c56c8f3` |

四个正式副本位于：

```text
/data/run01/scvi806/user_Wangjimin/projects/Filament_1/legacy/runs/<campaign_id>/
```

四个旧顶层源目录不再占用原路径，而是以相同 device/inode 原子移动到：

```text
/data/run01/scvi806/user_Wangjimin/projects/Filament_1/quarantine/
  relocated_legacy_sources_20260825/<旧目录名>/
```

每个正式目标均由独立 destination manifest 与 source manifest 逐文件比较。相对
路径、文件数、总字节数、文件大小和 SHA256 全部一致。quarantine 验证复用移动前
SHA256，并确认 device/inode、文件数、字节数及全部相对路径一致；没有执行第三次
全量哈希。

## 科学结论保护

- Hybrid 0.60 的机械分类仍为 `hybrid_0p60_not_supported`；人工实用分类仍为
  `hybrid_0p60_partially_supported_for_acceleration`，仅限低精度加速。
- Isaacs complete Eq.27 C2 仍为 `electronic_eq27_operator_not_supported`，并保留
  non-strict provenance 限制。
- Historical FR mixture 与 Raman-off Kerr 0.85 的原报告和验收文字未改变。
- archive gate 只证明管理证据完整，不构成新的科学有效性结论。

## 安全边界

- 迁移窗口内 `squeue` 和 `sacct` 均无作业；Slurm 操作数为零。
- 旧 dirty `Filament_1` 的 HEAD、status SHA256 和 stash SHA256 前后完全一致。
- `.secrets`、`.codex_ops` 的 device、inode 和权限前后一致。
- Phase 8B、Phase 8C 路径保持原位。
- 未删除 quarantine，未创建软链接，未运行仿真或后处理。

完整小型证据位于本地被忽略的
`.artifacts/<campaign_id>/hpc_relocation/`。Git 只记录
`configs/project_management/hpc_legacy_relocation_batch1.json` 中的路径与哈希摘要。
永久删除没有获得授权。
