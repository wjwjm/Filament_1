# Isaacs Eq.27 C2 执行复盘

日期：2026-08-22 至 2026-08-23。本文只记录执行工程和协作经验，不重写
物理结论、冻结结果或 job `221822` 的 provenance。此前 C2 代码与比较结果
仍按其原始 non-strict 和 operator-order 限定解释。

## 结论摘要

本轮最终只提交了一个传播作业；所有 pre-sbatch 失败都发生在一次性 lock
消耗前。问题主要来自三类边界没有在工具层固化：

1. Sol/Luna 的连接、写入状态和实际运行时配置没有完全分离核验；
2. Windows PowerShell、JavaScript、WSL、SSH、Bash 多层解析被压进了长命令；
3. Linux/HPC 环境、代理路径和跨平台字节 hash 没有统一 preflight/provenance
   契约。

## 事件表

| 类别 | 症状/错误 | 根因 | 副作用与发现门禁 | 当时修复 | 永久控制 |
|---|---|---|---|---|---|
| 模型连接 | 子 Agent `c2_execution_lock_fix` 返回 `502 Bad Gateway` | 模型/服务链路中断，发生在它已经写入部分修改之后 | 无额外作业；共享工作树审计发现已有 diff | Sol 接管现有最小 diff，重新测试；没有盲目启动第二 writer | 先查 Agent 状态与 `git status/diff`，记录 writer 所有权，再决定续接或恢复 |
| 工具模式 | 调用 `request_user_input` 报 `unavailable in Default mode` | 只在 Plan mode 可用的工具被错误调用 | 无文件/HPC 副作用；调用返回即暴露 | 改为当前模式下的直接执行/报告 | 在执行前核对模式和工具可用性；把模式能力当作运行时事实单独验证 |
| JavaScript | `SyntaxError: Unexpected string` | JS 字符串、PowerShell、Bash 三层引号嵌套 | 未产生作业；命令构造阶段失败 | 改用模板字符串、短命令和脚本文件 | 出现一次 quote 错误即升级到脚本上传，禁止继续堆叠转义 |
| WSL 路径 | WSL 报 `C:Userswangj/papp_cloud... No such file or directory` | PowerShell 先把 `~` 或反斜杠解释，WSL 未获得预期路径 | 传输前失败 | 让 `~` 在 WSL 内展开，或传入显式 WSL 路径 | wrapper 内统一 `wslpath`，不手写 Windows/WSL 混合路径 |
| SSH/Bash 引号 | `unexpected EOF while looking for matching quote` | `find -printf`、命令替换和多层引号拼成长串 | 远端命令未执行 | 拆成短命令，最终采用上传 helper | 复杂命令只能走固定脚本和 JSON manifest |
| Slurm 查询 | `squeue: Unrecognized option: %T` | `-o` 格式字符串在跨层传递时被拆散 | 仅监控查询失败 | 改用默认 `squeue` 与独立 `sacct -P` | 监控脚本使用参数数组和固定格式；终态以 `sacct` 为准 |
| 正则/管道 | `numerical_admission is not recognized...` | PowerShell 把正则中的 `|` 当作管道/命令分隔 | 审计命令失败，无作业副作用 | 使用多个 `grep -e` 或脚本内匹配 | 远端匹配逻辑放 Bash/Python 文件，PowerShell 不承载正则 shell 语法 |
| 长 launcher | `bash: line 3: \: command not found` | 过长环境变量/续行命令的反斜杠在多层解析中失效 | preflight 失败，lock 尚未消耗 | 上传无密码的短期 helper，核验 SHA 后执行 | `Invoke-PappRemoteScript.ps1` 固定上传脚本、manifest、dispatcher |
| 变量展开 | 远端输出 `DIFF_RC=True` | `$?` 在本地 PowerShell 提前展开，诊断命令被污染 | 仅该诊断不可信 | 丢弃该次结果，改用独立远端脚本 | 不在双层字符串中使用 `$?`/`$()`；所有 native 调用检查本地 `$LASTEXITCODE` |
| 路径/通配符 | `Filament_python/slurm` 不存在或 Windows `rg` 通配形式不兼容 | 假定了不存在目录或混用了 shell glob 语义 | 只读搜索失败 | 查询实际目录，使用显式文件列表 | preflight/文档先给真实路径；跨 shell 不传裸 glob |
| Python 环境 | 登录节点默认 Python 报 `ModuleNotFoundError: numpy` | 默认解释器不是生产 Miniforge 环境 | 未提交作业；环境审计拒绝继续 | 显式激活 `Filament_python` | preflight 固定 Miniforge hook、conda env、NumPy/CuPy import |
| bundle checkout | clone 后 `remote HEAD refers to nonexistent ref` | bundle 没有默认 HEAD 或 branch 指针 | 无结果副作用 | 从 `refs/remotes/origin/codex/isaacs-raman-reclosure` 显式建本地 branch | 远端 checkout 同时核验 HEAD、branch、clean；不能依赖默认 HEAD |
| CRLF/LF | manifest Windows SHA 与 Linux checkout SHA 不一致 | 同一文本的工作树换行字节不同 | v1 provenance 审计 STOP；未改变代码/结果 | 明确记录 byte-level 差异；历史锁定文件按原 CRLF 物化 | v2 tracked text 记录 Git blob OID + canonical-LF SHA；external 仍 raw SHA |
| checkout-index | 设置 `core.autocrlf` 后 SHA 仍为 LF | 已存在工作树不会被该设置自动重写 | 只影响审计重物化 | 使用显式字节物化并重新核验 | v2 不依赖隐式 Git checkout 转换 |
| staging 状态 | 文件 SHA 正确但 `git status` 仍 dirty | `.gitattributes` 与独立锁定字节要求冲突 | staging 审计停止 | 仅对四个明确锁定文件使用 `assume-unchanged`，未隐藏代码改动 | v2 canonical provenance；不普遍使用 `assume-unchanged` |
| comparator audit | 报告被错误要求自带 `COMPLETED/0:0` | diagnostic report 与 scheduler live state 混为一谈 | 旧 audit STOP；无作业副作用 | 报告只校验 schema/hash/NPZ，终态由实时 `sacct` 提供 | preflight、scheduler、diagnostic 三类证据分层 |
| Bash 变量 | `FIXED_REMOTE_CAMPAIGN_ROOT: readonly variable`，随后 `KeyError: FIXED_CAMPAIGN_ID` | launcher 把校验输入声明为 readonly 又尝试覆盖/引用错误键 | pre-sbatch 失败；一次性 lock 未消耗 | 改为非 readonly validator 输入并严格比较固定 root/campaign | guardrail 测试覆盖变量 shadowing、固定字段和 receipt schema |
| fake Python 测试 | 32 项中 1 项 `manifest validator returned no derived config binding` | 回归测试 fake Python 未区分三种调用形状 | 仅测试失败，无生产文件副作用 | 按参数数量/validator 环境区分调用，恢复 32/32 | 测试 fixture 明确接口，不靠调用次数猜分支 |
| fixture 依赖 | 干净 checkout 报 `.git/codex-locks` `FileNotFoundError` | 测试隐含本机运行时 lock | 低成本测试失败 | `tmp_path` 生成最小占位 lock/provenance | 测试自带最小 fixture，禁止依赖个人目录 |
| WSL 可移植性 | 测试路径假定 `/mnt/<drive>` | Windows/WSL 专用路径进入通用断言 | 当前环境可通过，原生 Linux 风险未消除 | 记录为非阻断 portability 问题 | wrapper 先探测路径转换；测试按平台 skip/断言 |
| GitHub 传输 | HPC `clone/fetch/ls-remote` GnuTLS/超时 | 登录节点到 GitHub 的直接链路不可用 | 未能证明 direct GitHub provenance；候选未运行第二次提交 | 使用 SHA 核验 Git bundle，标记 `verified_bundle_non_strict` | 代理优先 `ls-remote`，失败才接受 bundle/ref/SHA/HEAD 全校验 |

## 门禁对应关系

| 门禁 | 目的 | 失败时允许的下一步 |
|---|---|---|
| 本地状态审计 | 防止误把已有写入当作无写入 | Sol 检查 diff/所有权后才续接 |
| TOML 与运行时角色核验 | 区分静态配置和真实 Agent 行为 | 修正配置或报告 runtime mismatch |
| wrapper dry-run | 在无网络时验证 account/root、参数数组和 JSON 形状 | 修复本地输入，不触碰 HPC |
| HPC preflight | 验证环境、工具、仓库和代理/bundle 来源 | 失败即 STOP，不创建 lock/run/job |
| SHA/manifest | 绑定代码、配置、脚本和外部产物 | 重新生成候选 provenance，废弃旧工件 |
| scheduler live state | 防止把 RUNNING 当完成 | 继续监控；不 postprocess、不分类 |
| numerical admission | 防止 `I_cap`/LUT 饱和污染比较 | STOP，交回 Sol 做科学判断 |

## 后续执行模板

```text
任务边界/冻结对象：
本地 HEAD/branch/clean：
Agent role/model/effort/sandbox：
只读 mapper/reviewer 证据：
唯一 writer 与文件所有权：
脚本 dry-run 与测试：
HPC preflight JSON/schema/source_class：
bundle 或代理的 raw/canonical SHA：
lock/receipt 是否消耗：
Slurm live state：
未验证事项与需 Sol 决策：
```

本复盘不授权重新提交 `221822`、覆盖历史结果、改变物理模型或自动轮换
凭据；这些均需独立的父 Agent/管理员授权。

## 可靠性整改验收记录

验收基线：本地分支 `codex/isaacs-raman-reclosure`，HEAD
`463e279a5d689fa61b7fe869880550420df72a00`；整改改动保持未提交、未推送。

- guardrail、C2、完整 Eq.27、Raman reclosure 和 sanity 合计 `60 passed`；
  Windows 权限/原生 POSIX 条件导致 `3 skipped`；WSL 覆盖了 symlink、代理
  probe 与 bundle-preflight 主路径，但 malformed-key 的 mode-600 拒绝仍仅在
  原生 POSIX fixture 中执行。
- `compileall`、三个 Bash `-n`、PowerShell AST、四个 Agent TOML、secret
  scan 和 `git diff --check` 通过。
- `Filament_python/results/isaacs_complete_eq27` 及 job `221822` 的 27 个
  tracked 文件无 diff；v1 provenance 未重写。
- 2026-08-22 的初次 `scvi806` live preflight 使用已核验 bundle，来源为
  `verified_bundle_non_strict`，没有把 bundle 误写成 direct GitHub 证明。
- 2026-08-23 将用户确认并实测有效的代理迁入用户自有、mode-600 的
  `/data/run01/scvi806/user_Wangjimin/.secrets/github_proxy.env`，并把旧的
  `更新并提交代码.txt` 替换为无凭据说明文件。随后通过该 secret 获取当前
  GitHub 分支 `463e279a5d689fa61b7fe869880550420df72a00` 的干净 checkout；完整
  `filament.hpc_preflight.v1` 的 `account_root/repo/tools/python_env/
  proxy_or_bundle` 全部通过，来源升级为 `strict_remote_verified`。
- live preflight 后 `.codex_ops` 为空；未创建仿真 run/lock，未调用 `sbatch`。
- 迁移和验收全过程未回显代理值；一次性源码 checkout 与 `.codex_ops`
  staging 已清理，未创建仿真 run/lock，未调用 `sbatch`。
