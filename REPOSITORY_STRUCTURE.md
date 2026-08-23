# REPOSITORY_STRUCTURE.md — Filament_1 目录结构与放置规则

> 第一轮整理范围：只做分类、inventory、低风险移动与文档化；**不新增多脉冲物理，
> 不改变任何公式、默认参数、operator、precision 策略或数值顺序。**

## 0. 当前状态（2026-08-23，branch `restructure_round1`）

- 已完成：完整五类 inventory + machine-readable manifest；
  4 个 phase 文档从 `Filament_python/docs/` 归档到目标 `docs/`；
  `修改记录/` 证据归档到 `results/reference_evidence/`；
  `configs/production/` 建立字节一致的默认配置副本（原文件仍为唯一权威，未移走）。
- 已按约束**暂缓**：`Filament_python/configs|stages|results|tests|tools` 的物理移动，
  因为现有脚本/测试以 `ROOT/"..."` 硬编码引用它们（约 40+ 处）。这些路径的
  import/deploy 重构不在“低风险移动”范围内，留待后续批次（见 §5）。

## 1. 五个分类

| class | 含义 | 判断标准 |
| --- | --- | --- |
| `production_runtime` | 可复用运行库 | `Filament_python/KHz_filament/` 内 `.py` 与其说明 |
| `production_config` | 授权生产默认配置 | `config_ref.json`、`khz_config.json`、`khz_config_lut.json`（权威原件） |
| `generic_tests_tools` | 通用测试/工具 | 不受单次 Phase 绑定的 runner / postprocess / audit / unit test |
| `historical_experiments_audits` | 历史因果实验与审计 | 文件名带 phase / model-审计 / frozen-stage 标志，且只在某一冻结阶段有意义 |
| `results_documentation_evidence` | 结果/文档证据 | 冻结结果、reference、论文 PDF、修改记录、项目文档 |

机器可读清单：`docs/repo_layout/repository_inventory.json`（含 path / size / class / git_tracked / SHA256）。
人类可读清单：`docs/repo_layout/repository_inventory.md`。

## 2. 目标目录职责

```text
D:\Filament_1\
├── Filament_python\KHz_filament\   # 目标1：只放可复用运行库，禁止加入 Phase 编号/具体作业逻辑
├── configs\
│   ├── production\   # 目标2：授权生产配置（借阅副本；权威原件仍在上层）
│   ├── validation\   # 同一物理设定下的校验/对拍配置
│   ├── experiments\  # 每实验一个子目录，写清冻结 SHA/config-hash
│   └── archive\      # 一次性 phaseX_* 配置，冻结读取
├── tools\
│   ├── run\          # 目标3：通用提交/运行入口
│   ├── postprocess\  # 通用后处理
│   ├── audit\        # 通用审计（含 build_repository_inventory.py）
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
└── results\                 # 目标6：只放小型可追溯证据（json/csv/md/小图）
    └── reference_evidence\
```

## 3. 以后“新东西放哪里”

- **新生产运行代码** → `Filament_python/KHz_filament/`。只接受可复用、可开关、CPU/GPU 行为一致的库代码。
  禁止把 `phaseX_...`、单次作业、特定 Slurm job 逻辑写进核心模块。
- **新生产配置** → `configs/production/<name>.json`；默认参数先改权威原件
  (`Filament_python/khz_config.json` 等)，再同步副本，并保证解析结果不变。
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

## 5. 下一轮（明确列为待办，不在本轮执行）

1. 建立 `ROOT` 解析层：把散落的 `ROOT/"configs|stages|results|tools"` 硬编码统一为
   `paths()` 解析器，再物理移动上述目录到目标骨架。
2. 移动 `Filament_python/tools/tmp` 与 phase 工具到 `tools/archive`（先逐一核对引用）。
3. `Filament_python/results` 拆分为 `results/reference_evidence` 与 HPC provenance。
4. tests 拆分 unit/integration/regression（保留 conftest 根，避免 pytest 行为变化）。
5. 为 `configs/production` 建立“权威原件单向同步”校验脚本，防止副本漂移。
</parameter>
