# Filament_1 HPC 项目命名空间阶段一 cutover

阶段一已于 2026-08-25 14:49:47–14:51:04 UTC 完成。远端 receipt 状态为
`completed`、`ok=true`。本报告仅记录管理命名空间调整，不改变传播核心、生产
配置解析、物理结论或原始 NPZ/MAT。

## 路径结果

- 新项目根：`/data/run01/scvi806/user_Wangjimin/projects/Filament_1`
- 新源码 staging：`projects/Filament_1/source/staging/`
- 新 campaign 根：`projects/Filament_1/campaigns/`
- 旧兼容仓库：`/data/run01/scvi806/user_Wangjimin/Filament_1`，保持原位
- 观察区：`projects/Filament_1/quarantine/namespace_cutover_20260825/account_root_management/`
- 观察区包含：`staging/`、`campaigns/`、`cache/`、`archive/`、`quarantine/`

旧管理目录是移动到项目 quarantine 观察区，并未永久删除。`.secrets`、
`.codex_ops`、旧仓库和 19 个旧运行目录没有移动。

## 证据 SHA256

| 证据 | SHA256 |
| --- | --- |
| source management manifest | `41990245b3da6e9e0cb3c8c2e08241d82ed88582de758fcfca7152c959459a6d` |
| target pre-rewrite manifest | `10037b23dd37617913286e737739f29bb0df7136eef0ae09129448413c215334` |
| target final namespace manifest | `82f66dcb485da62c1c659863d7dfe9d8c5b9496dece7c3fcf3e57e211a32704d` |
| raw protection manifest, before | `23c762b26b2399d09f295e12c24af4723337d244865b1e275fbaad47deff44f8` |
| raw protection manifest, after | `23c762b26b2399d09f295e12c24af4723337d244865b1e275fbaad47deff44f8` |
| legacy roots manifest | `0c5945ce77da0e83b20bd3c05da4eeda843df2352df5f691d169ee5b397f2af6` |
| quarantine manifest | `c0be9225c47f60240e45c6363bb8ef8abd4be9a4533295f8efce91b32c5785ea` |
| namespace cutover receipt | `51c805bc14fc9e27bc63437ce15639ae058e0e01107c24b1e3f5525340efd700` |
| synchronized evidence bundle | `124d55efb6295e865312d4a2356758f81c42f96ca1fde7c43753b49e372712ad` |

完整小型证据已同步到本地忽略目录
`.artifacts/20260825_hpc_namespace_cutover_v01/`，不进入 GitHub。

## 验收结果

- 新 staging：`main@63651ba389e5166012b1297eebdcf46b69cc289d`，worktree clean。
- 四个正式 legacy campaign 可读取且生成新的 namespace archive receipt：
  - `20260820_historical_fr_mixture_final_v01`，job `215812`
  - `20260821_raman_off_kerr085_final_v01`，job `220822`
  - `20260822_isaacs_complete_eq27_c2_v01`，job `221822`
  - `20260824_hybrid_propagation_0p60_v01`，job `222025`
- 旧仓库 HEAD 在前后均为 `bb592ef1c16ee9c9572f041b2ba31db4d53b4582`；status SHA256 均为
  `b4fde2c627b82ae14674589662bb773a366928020ffa4d6c0748239f5c2f8b7e`；stash SHA256 均为
  `12790ce0a83289b1a935d5f7bfaa641757d1104e0099a0225175283954644a70`。
- 19 个旧运行目录的 cutover 前后目录快照一致，仍在账号根原路径。
- NPZ/MAT 保护清单前后相同：148 个文件，共 1,029,633,202 bytes。
- cutover 前后 `squeue -u scvi806` 为空，起点后的 `sacct` 无记录；未执行 Slurm 操作。
- 未创建软链接，未永久删除源目录，未移动或删除原始结果。

## 本地旧路径引用审计

全仓只读搜索将旧路径引用分为两类：

1. README、仓库结构和三端管理手册中的未来操作入口已更新为新项目根；机器可读
   配置成为新任务路径权威。
2. `results/campaigns/legacy_registry.json`、`Filament_python/results/**` 中的配置、
   日志、submission manifest 和结果报告均属于历史运行证据，继续保留冻结绝对
   路径，不进行替换或重新解释。

未发现仍把账号根 `staging/`、`campaigns/`、`cache/`、`archive/` 或
`quarantine/` 作为新任务入口的 tracked 操作说明。

## 阶段二计划（未执行）

剩余旧目录将按 campaign 逐个登记、通过 archive 门禁、复制并核对 SHA256；只有
验证通过后，单个原目录才可进入 quarantine。四个已登记 campaign 在迁移前也要
重新执行 archive 检查。永久删除不属于迁移授权，必须对每个目标单独报告路径、
manifest 与预计释放空间并请求人工授权；未获授权时 quarantine 无限期保留。
