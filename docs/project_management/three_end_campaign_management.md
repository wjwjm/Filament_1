# 三端 campaign 管理规则

本规则只管理项目证据和路径，不改变传播公式、默认参数、配置解析或
数值策略。三端不要求保存同一套文件，而是通过不可变引用建立对应关系：

```text
campaign_id + execution_git_sha + config_sha256 + artifact_sha256
```

## 三端职责

| 端 | 权威内容 | 允许的工作 |
| --- | --- | --- |
| 本地 `D:\Filament_1` | 开发代码、测试、精选证据准备 | 编辑代码、审阅 HPC 派生结果 |
| GitHub `main` | 代码、去环境化配置、精选小型证据 | 版本历史和可审阅发布 |
| HPC | 具体 requested/resolved 配置、原始数据、完整日志和调度证据 | 按固定 SHA 执行与后处理 |

本地完整派生结果放在 `.artifacts/<campaign_id>/`，该目录被 Git 忽略，
因此不会覆盖代码，也不会自动进入 GitHub。需要发布的文件必须使用显式
allowlist 调用 `publish-plan --apply`，目标为
`results/campaigns/<campaign_id>/artifacts/`。

## HPC 项目命名空间

2026-08-25 阶段一 cutover 后，新任务的规范根为：

```text
/data/run01/scvi806/user_Wangjimin/projects/Filament_1/
├── source/staging/<campaign_id>/Filament_1_<short_sha>/
├── campaigns/<campaign_id>/
├── cache/
├── archive/
├── quarantine/
└── legacy/
```

账号根下旧 `/data/run01/scvi806/user_Wangjimin/Filament_1` 保持
`legacy_compatibility_root` 身份，不 pull、不 reset，也不再启动新任务。原账号根
`staging/campaigns/cache/archive/quarantine` 已移入
`projects/Filament_1/quarantine/namespace_cutover_20260825/account_root_management/`
观察区；这不是永久删除授权。

第二阶段第一批已将四个正式 legacy campaign 复制并逐文件验哈希至
`projects/Filament_1/legacy/runs/<campaign_id>/`，旧顶层源目录随后原子移动至
`projects/Filament_1/quarantine/relocated_legacy_sources_20260825/`。Phase 8B、
Phase 8C 和旧 dirty `Filament_1` 不在该批范围。具体路径与 receipt 哈希见
`configs/project_management/hpc_legacy_relocation_batch1.json`。

机器或脚本不得自行拼接 HPC 路径，应读取
`configs/project_management/hpc_namespace.json`。历史 campaign 中已经冻结的绝对
路径和 `legacy.source_root` 不回写；新 campaign 的 `paths.hpc_root` 必须位于
`projects/Filament_1/campaigns/<campaign_id>/`。

## Campaign 与普通任务

普通代码、文档和单元测试不必创建 campaign。已有结果的轻量后处理可关联
已有 campaign 并使用 `check --level lite`。新 HPC 运行、正式结果发布和归档
才需要完整 campaign 记录。

## 配置与安全

`init` 在 `configs/experiments/<campaign_id>/requested|resolved/` 建立工作配置
目录；`publish-config` 将审核后的 requested/resolved 快照写入
`results/campaigns/<campaign_id>/configs/`，同时登记文件 SHA256。输入文件保持
不变。token、password、credential、proxy 等 secret-like key、
带认证信息的 URL、Windows/HPC 绝对路径均拒绝进入可发布配置；完整原始
配置可留在 HPC，但不得把凭据写入仓库。

## 分级检查和缓存

```powershell
python tools/campaign/manage.py check <campaign_id> --level lite
python tools/campaign/manage.py check <campaign_id> --level submit
python tools/campaign/manage.py check <campaign_id> --level publish
python tools/campaign/manage.py check <campaign_id> --level archive
```

检查只在对应边界执行，不重新运行仿真或后处理。receipt 的指纹包含检查
级别、campaign JSON、配置/manifest、manifest所指文件、实时staging状态、batch
audit receipt和已发布evidence的实际哈希；输入不变时复用
`.artifacts/<campaign_id>/.validation/` 中的结果。

## 历史结果

旧的 `Filament_python/results/*` 不移动、不重命名、不重新解释科学结论。
`register-legacy` 根据冻结 inventory 机械登记全部 18 个顶层目录，状态统一为
`legacy_unclassified`。后续若要迁移，必须另行审计路径引用和冻结 provenance。

## 阶段二门禁

阶段二本次未执行。剩余旧目录必须逐个登记为 legacy campaign，并分别通过
archive 门禁后才能复制、验哈希和进入 quarantine。迁移授权不包含永久删除；每个
删除目标都必须单独报告路径、manifest 和预计释放空间，并单独获得人工授权。
