# REPOSITORY_STRUCTURE.md — Filament_1 目录结构与放置规则

> 第一轮整理范围：只做分类、inventory、低风险移动与文档化；**不新增多脉冲物理，
> 不改变任何公式、默认参数、operator、precision 策略或数值顺序。**

## 0. 当前状态（2026-08-25，branch `main`）

- 已完成：第一轮结构基线的完整五类 inventory + machine-readable manifest；
  4 个 phase 文档从 `Filament_python/docs/` 归档到目标 `docs/`；
  `修改记录/` 证据归档到 `results/reference_evidence/`；
  `configs/production/` 建立字节一致的默认配置副本（权威原件仍为唯一权威），
  并有只读 SHA256 校验器 + pytest 回归测试防止副本漂移。
- 已**冻结**：`Filament_python/configs|stages|results|tests|tools` 的物理移动，
  因为现有脚本/测试以 `ROOT/"..."` 硬编码引用它们（约 40+ 处）。这些路径的
  大规模迁移不在本轮范围，按 §5 冻结为 `deferred_until_architecture_requires_it`。
- 已完成 HPC 项目命名空间阶段一：新管理根固定为
  `/data/run01/scvi806/user_Wangjimin/projects/Filament_1`；账号根下五个旧管理
  目录已进入项目 quarantine 观察区。旧 `Filament_1`、19 个旧运行目录、
  `.secrets`、`.codex_ops` 和 NPZ/MAT 均保持原位。阶段二尚未开始。

## 1. 五个分类

| class | 含义 | 判断标准 |
| --- | --- | --- |
| `production_runtime` | 可复用运行库 | `Filament_python/KHz_filament/` 内 `.py` 与其说明 |
| `production_config` | 授权生产默认配置 | `config_ref.json`、`khz_config.json`、`khz_config_lut.json`（权威原件）与 `configs/production/` 副本 |
| `generic_tests_tools` | 通用测试/工具 | 不受单次 Phase 绑定的 runner / postprocess / audit / unit test |
| `historical_experiments_audits` | 历史因果实验与审计 | 文件名带 phase / model-审计 / frozen-stage 标志，且只在某一冻结阶段有意义 |
| `results_documentation_evidence` | 结果/文档证据 | 冻结结果、reference、论文 PDF、修改记录、项目文档 |

第一轮冻结基线清单：`docs/repo_layout/repository_inventory.json`（含 path / size /
class / git_tracked / SHA256）；人类可读版本为
`docs/repo_layout/repository_inventory.md`。后续新增内容由 campaign registry 和Git
历史补充，不把该冻结基线描述为实时工作树扫描结果。

## 2. 目标目录职责

```text
D:\Filament_1\
├── Filament_python\KHz_filament\   # 目标1：只放可复用运行库，禁止加入 Phase 编号/具体作业逻辑
├── configs\
│   ├── production\   # 目标2：授权生产配置（权威原件同步副本，SHA256 校验）
│   ├── validation\   # 同一物理设定下的校验/对拍配置
│   ├── experiments\  # 每实验一个子目录，写清冻结 SHA/config-hash
│   └── archive\      # 一次性 phaseX_* 配置，冻结读取
├── tools\
│   ├── run\          # 目标3：通用提交/运行入口
│   ├── postprocess\  # 通用后处理
│   ├── audit\        # 通用审计（build_repository_inventory.py + 配置副本校验器）
│   └── archive\      # 一次性 phaseX_* 脚本逐步移入
├── tests\
│   ├── unit\         # 目标4
│   ├── integration\
│   └── regression\
├── docs\
│   ├── architecture\        # 目标5：架构与数据流
│   ├── physics_decisions\   # 物理决策与因果实验结论
│   ├── known_residuals\     # 已知残余/未闭合项
│   └── repo_layout\         # inventory/manifest/path_map
└── results\                 # 小型可追溯证据（json/csv/md/小图）
    ├── campaigns\           # 显式发布的 campaign 证据与 legacy registry
    └── reference_evidence\
```

## 2.1 三端 campaign 目录边界

本地 `D:\Filament_1` 是代码和项目管理工具的连续工作区。完整派生结果放在
`.artifacts/<campaign_id>/`，该目录被 `.gitignore` 忽略，且不会自动反向
覆盖任何源码、配置或文档。`tools/campaign/manage.py publish-plan` 默认只
生成 dry-run 计划，只有显式的、非全局 `--allow` 模式配合 `--apply` 才能将
选定文件复制到 `results/campaigns/<campaign_id>/artifacts/`。

GitHub 管理代码、去环境化 requested/resolved 配置、工具和精选小型证据；HPC
管理具体运行配置、原始 NPZ/MAT/HDF5、完整后处理和 scheduler evidence。三端
通过 `campaign_id`、execution SHA、config SHA256 和 artifact manifest SHA256
对应。旧 `Filament_python/results/*` 保持原位，由
`results/campaigns/legacy_registry.json` 机械登记，不在本轮迁移。

HPC 新任务只使用以下命名空间：

```text
/data/run01/scvi806/user_Wangjimin/projects/Filament_1/
├── source/staging/<campaign_id>/Filament_1_<short_sha>/
├── campaigns/<campaign_id>/
├── cache/
├── archive/
├── quarantine/
└── legacy/
```

旧 `/data/run01/scvi806/user_Wangjimin/Filament_1` 是冻结的
`legacy_compatibility_root`，不是新任务 checkout。阶段一配置和 receipt 摘要分别见
`configs/project_management/hpc_namespace.json` 与
`docs/project_management/2026-08-25_hpc_namespace_phase1_cutover.md`。

## 3. 以后“新东西放哪里”

- **新生产运行代码** → `Filament_python/KHz_filament/`。只接受可复用、可开关、CPU/GPU 行为一致的库代码。
  禁止把 `phaseX_...`、单次作业、指定 Slurm job 逻辑写进核心模块。
- **新生产配置** → `configs/production/<name>.json`；默认参数先改权威原件
  (`Filament_python/khz_config.json` 等)，再同步副本，并保证解析结果不变；提交前跑
  `python Filament_python/tools/audit/verify_config_production_copies.py`（或在 pytest 中自动验证）。
- **新校验/实验配置** → `configs/validation/` 或 `configs/experiments/<实验名>/`。
  冻结后记 SHA256，并在 `docs/physics_decisions/` 留决策记录。
- **新工具** → 通用放 `tools/run|postprocess|audit`；一次性 `phaseX_*` 直接放 `tools/archive/`
  （或完成使命后移入）。
- **新测试** → unit/integration/regression 之一；只服务某个冻结阶段的测试放 `tests/regression/`
  并注明冻结 SHA，不要再混进主回归集合。
- **新文档** → `docs/architecture|physics_decisions|known_residuals` 之一。
- **新结果** → 小文件（JSON/CSV/Markdown/小图）放 `results/reference_evidence/` 并登记 manifest。
  大 NPZ / HPC 原始输出 **不进入 results/**，只保存
  `{路径, SHA256, 产生 commit, config-hash, Slurm job}` provenance。

## 4. 冻结引用原则（不可违反）

- 所有移动必须登记 `docs/repo_layout/path_map.json` 的 old→new。
- 冻结 SHA、Slurm job、config hash、result hash 的引用不得失效：先查引用，再移动；
  有引用的先保留原位或做向后兼容 shim（本轮选择保留原位）。
- 不删除任何已有科学证据，只移动并保留 git rename 历史。

## 5. 旧目录冻结：deferred_until_architecture_requires_it

`Filament_python/configs|stages|results|tests|tools` 的**大规模物理迁移已冻结**，
不做“为了整理而全仓改造”。只有当下述任一项被实际开发阻塞时，才**局部**引入 path
resolver 并移动被阻塞的那一部分：

1. 某项新功能必须引用目标骨架路径（如 `configs/production/`、`tools/run/`），旧路径不再满足。
2. 某份冻结 SHA / Slurm job / config hash 的引用因旧路径持续扩展而无法追溯。
3. 开发者反复跨旧目录搬文件，维护成本已被具体任务证实。

触发后按顺序做：先建立最小 `paths()` 解析层 → 只迁移受阻塞的文件 → 更新
`docs/repo_layout/path_map.json` → 跑全量 pytest 与 Npulses=1 等价性验证 → 单独 commit。

在触发之前，旧路径继续保持原样，仅做 phase 脚本/测试的逐步 archive（有引用先保留）。
