# 当前项目状态 (Round 1 Restructuring)

- 日期: 2026-08-23
- 分支: `restructure_round1`
- 基线 HEAD (本轮开始前): `37f79794f8b1dd93b4431e11f21e70f7059c6492`

## 目标

建立清晰、可维护的目录与入口，为“单脉冲 + 多脉冲”开发提供放置规则；
本轮**只整理**，不新增多脉冲物理。

## 已完成（软件/结构）

1. 全仓库 inventory 与五分类：`docs/repo_layout/repository_inventory.{md,json}`
   - 786 files / 92,011,103 bytes；含 git_tracked 与 SHA256（>64 MiB 的文件不哈希）。
2. `configs/production/` 三份默认配置副本（与权威原件 SHA256 一致，原件未动）。
3. 4 个 phase 文档归档：`docs/architecture|physics_decisions|known_residuals`。
4. `修改记录/` 证据归档：`results/reference_evidence/修改记录/`。
5. old→new 路径图：`docs/repo_layout/path_map.json`。
6. 目标骨架目录已建：`configs|tools|tests|docs|results` 的子分类。

## 未完成 / 遗留（待后续轮次，需用户确认）

- `Filament_python/configs|stages|results|tests|tools` 的物理移动：
  被约 40+ 处 `ROOT/"..."` 硬编码引用阻断，本轮按约束暂缓，避免破坏入口。
- `configs/production` 目前是“借阅副本”；权威原件仍在 `Filament_python/` 顶层，
  待 ROOT 解析层建立后再统一迁移。
- `tests/` 拆分与 `tools/archive` 逐步归档尚未开始。

## 验证结论（严格分离软件与科学）

- pytest：基线 268 passed, 3 skipped（约 69.8 s，改动前）；**再验证 268 passed, 3 skipped**。
- 生产行为不变证明：`git diff 37f7979..HEAD -- <生产面>` 为空（零改动）。
- tiny Npulses=1 冒烟（fp32）两次独立运行：`U_z` 守恒至 0.01%，输出 NPZ
  SHA256 完全一致（bitwise）。
  - 此为运行确定性/等价性证据，**不构成任何新的科学结论**。

## 下一轮建议

见 `REPOSITORY_STRUCTURE.md` §5。
