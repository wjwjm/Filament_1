# 当前项目状态 (Round 1 Restructuring)

## 2026-08-30: HR-3C-A interpulse transverse diffusion

- HR-3C-A = **CLOSED**：冻结 authoritative `D_th=21.7e-6 m^2/s`，对 HR-3B
  interval-centered `delta_n_th[k,y,x]` 的单个二维 slice 施加
  `dt_interpulse=1/f_rep` 的谱横向扩散。该算子采用现有 `kperp2`、周期边界和
  `R_edge <= 1e-3` 的 fail-closed numerical-validity gate。
- 未实现 HR-3C-B 的 ping-pong/storage/performance，亦未实现 HR-3C-C 的
  `Npulses=N -> N-1` orchestration、checkpoint 或 crash consistency。

## 2026-08-30: air refractivity scaling repair

- `KHz_filament.air_dispersion` 的 Ciddor-simple refractivity 经验式已移除重复的
  外层 `1e-6` 缩放；这是既有光学背景折射率实现错误的修复，不改变 HR-3B 的
  `beta_th`、热力学参数来源、persistent slow state 或脉冲排序合同。

## 2026-08-30: HR-3A thermalization contract

- 开发分支：`HR-3`（尚未合并 `main`）。
- HR-3A = **CLOSED**：将 HR-2 权威、分机制、按 interval 的沉积图转换为独立的 HR-3A
  microscopic-thermalization ledger；完整热化是两篇核心参考文献支持的模型近似，
  而非 fs 瞬时平动升温结论。
- HR-3A-R：改为 interval-streaming、O(K) 标量账本与物理 z 稀疏二维诊断 sidecar；
  不再在 RAM 或主 NPZ 中保留 full-z thermal/deposition map history。
- HR-3A-R2：静态 sample-map 估算修正为 nominal `~501` 张 / `~501 MiB`（实际值以
  `build_physical_sample_plan(...).count` 为准）；T1/T2/T3 closure 状态独立，overall
  authority 为三者 conjunction。已知 HR-2E strict-float baseline failure 不阻止 HR-3A close。
- HR-3B = **CLOSED**：实现为显式 opt-in 的 post-acoustic reduced mapping。唯一 authoritative
  persistent slow state 是 interval-centered、disk-backed `delta_n_th[K,Ny,Nx]`；每个
  interval 读取旧 slice 用于本发相位，再在 HR-3A authoritative `q_thermal` 完成后原地累加。
  `Q2D/gamma_heat/dn_gas` 仅保留 non-authoritative legacy compatibility 模式。
- 未实现：persistent `Delta T`、persistent `delta rho`、显式声学/等压瞬态、扩散、
  传热传质及脉冲间慢状态传播；这些属于 HR-3C/HR-4。
- 保留：HR-2E = **DEFERRED**；production longitudinal schedule = **NOT FROZEN**；
  未提交 HPC 或 Slurm 作业。

- 日期: 2026-08-23
- 分支: `main`
- 基线 HEAD (本轮开始前): `37f79794f8b1dd93b4431e11f21e70f7059c6492`
- 本轮最终 HEAD 见 `git rev-parse HEAD`（以线头为准）。

## 目标

建立清晰、可维护的目录与入口，为“单脉冲 + 多脉冲”开发提供放置规则；
本轮**只整理**，不新增多脉冲物理。

## 已完成（软件/结构）

1. 全仓库 inventory 与五分类：`docs/repo_layout/repository_inventory.{md,json}`
   - 最终数字 **796 files / 92,430,286 bytes**；含 git_tracked 与 SHA256（>64 MiB 的文件不哈希）。
   - 唯一权威数字以 `docs/repo_layout/repository_inventory.json` 的 `total_files`/`total_bytes` 为准。
2. `configs/production/` 三份默认配置副本（与权威原件 SHA256 一致，原件未动）。
   - 新增只读校验器 `Filament_python/tools/audit/verify_config_production_copies.py`；
   - 新增 pytest 回归测试 `Filament_python/tests/test_config_production_copies.py`（CI/pytest 自动验证副本一致性）。
3. 4 个 phase 文档归档：`docs/architecture|physics_decisions|known_residuals`。
4. `修改记录/` 证据归档：`results/reference_evidence/修改记录/`。
5. old→new 路径图：`docs/repo_layout/path_map.json`。
6. 目标骨架目录已建：`configs|tools|tests|docs|results` 的子分类。

## 已冻结（不再主动搬动）

- `Filament_python/configs|stages|results|tests|tools` 的**大规模物理迁移已冻结**：
  `deferred_until_architecture_requires_it`（见 `REPOSITORY_STRUCTURE.md` §5）。
  仅在具体开发被旧路径阻塞时，才局部建 path resolver 并移动受阻塞部分。
- `configs/production` 目前是“权威原件 + 同步副本”双份状态；短期接受，
  由 SHA256 校验器 + pytest 测试防止漂移。

## 验证结论（严格分离软件与科学）

- pytest：基线 268 passed, 3 skipped（约 69.8 s，改动前）；**再验证 270 passed, 3 skipped（66.98 s，含新增配置副本一致性测试）**。
- 生产行为不变证明：`git diff 37f7979..HEAD -- <生产面>` 为空（零改动）。
- tiny Npulses=1 冒烟（fp32）两次独立运行：`U_z` 守恒至 0.01%，输出 NPZ
  SHA256 完全一致（bitwise）。
  - 此为运行确定性/等价性证据，**不构成任何新的科学结论**。

## 远端同步状态（待执行）

- 本地 `main` 领先 `origin/main` 的整理 commit 需在 Round 1.5 正常推送（**不 squash**，
  保留逐批迁移历史）。推送前重新核查：
  1. `git ls-remote origin refs/heads/main` 确认远端基点；
  2. 确认本地 6 个 restructuring commit 均在、生产面 diff 为空；
  3. `git push origin main` 后，`git ls-remote origin refs/heads/main` 应指向本地最新 HEAD。
