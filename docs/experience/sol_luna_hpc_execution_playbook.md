# Sol–Luna 与 HPC 执行可靠性手册

本手册把科学代码任务中的“规划、委派、执行、验证、外部副作用”分开。
它服务于 `Filament_1` 的后续任务，不改变物理模型、坐标定义、归一化、
冻结基线、PyCAP 比较口径或既有结果 provenance。

## 一、角色边界与状态机

Sol（父 Agent）保留任务解释、科学/数值决策、基线接受、外部写入、
Git commit/push、HPC staging、Slurm 提交和最终验收。Luna 子 Agent 只
执行边界明确的调查、局部实现、独立审查或低成本验证。

标准状态机如下：

1. Sol 读取任务书和仓库规则，记录工作树、当前分支、HEAD、运行环境、
   冻结文件和不可逆动作。
2. `filament_mapper` 做只读调用链/配置/测试映射；需要数学、单位、相位、
   能量或物理闭合时，并行使用 `filament_numerical_reviewer`。
3. 环境、连接、Shell、依赖或工具链故障先交给 `debug_scout`；不得让 worker
   在原因不明时猜测性修改生产代码。
4. Sol 根据证据确定一个变更边界和唯一 writer；重叠区域不得并行写入。
5. `filament_worker` 只修改批准的文件；`filament_tester` 和 numerical
   reviewer 在变更稳定后独立检查。
6. Sol 决定是否 commit、上传、提交作业或产生其他外部副作用；这些动作
   不由测试“自动升级”。
7. Scheduler 必须达到真实终态（例如 `COMPLETED/0:0`）后才能 postprocess；
   `PENDING`、`RUNNING`、单次 smoke 或测试通过都不是科学结论。

每次 Luna 返回必须包含下列字段，缺失字段应视为未完成交付：

```text
task_boundary
evidence
files_changed
commands_and_exit_codes
tests
unverified
parent_decisions
```

其中 `evidence` 要区分已核验事实、假设和证据路径；`parent_decisions`
要明确需要 Sol 选择的科学或不可逆动作。

### 模型/连接失败的恢复

502、超时、模型链接中断或子 Agent 没有最终回复时，执行以下顺序：

1. 先读取 Agent 状态和已返回的工具输出。
2. 再检查共享工作树 `git status --short`、`git diff`、未跟踪文件和文件
   所有权；“没有最终回复”不等于“没有写入”。
3. 确认已有 diff 的边界后由 Sol 集成最小修改，或只向原 Agent 发起可续接
   的 follow-up；未确认写入前禁止重新启动第二个 writer。
4. 单独核验静态 TOML、Agent 创建、实际 role/model/effort/sandbox 和
   child-visible payload；其中任何一个不能替代其他证据。

## 二、命令行与 Shell 可靠性

### 复杂度升级规则

没有变量展开的单条只读命令可以直接通过 SSH。出现中文路径、空格、管道、
重定向、正则、命令替换、heredoc、嵌套引号或第二层 Shell 时，改用
`Filament_python/tools/hpc_ops/Invoke-PappRemoteScript.ps1` 上传固定脚本
执行。第一次出现 quote/escape 错误后不得继续堆叠转义。

PowerShell 侧使用参数数组和 call operator；不使用 `Invoke-Expression`、
`cmd /c` 或额外 PowerShell 子进程。每次 native 命令调用后检查
`$LASTEXITCODE`。Bash 侧使用 `set -euo pipefail`、固定脚本参数、显式临时
文件和静态 JSON；禁止把用户参数拼接到 shell 代码中。

远端脚本应接收 JSON 参数 manifest，而不是把参数重新编码到一条长命令中。
`ReadOnly` wrapper 只能运行仓库内固定的 `hpc_preflight.sh`；其他脚本必须
明确使用 `Write + AllowRemoteWrite`。上传后先在 `${RemoteRoot}/.codex_ops`
下建立本次专属 mode-700 目录，脚本/manifest 为 600、dispatcher 为 700，
再核验 SHA256 并由固定 dispatcher 调用 `bash`。dispatcher 解析参数时必须
显式检查 Python 退出码，不得用吞掉失败的 process substitution。
任何错误输出都应脱敏，不能回显代理、token、认证 URL 或完整命令。

### 轻量命令顺序

先做本地 `git status`、脚本语法检查和 dry-run；再做远端只读 preflight；
最后才考虑 staging 或 scheduler。复杂操作的安全入口和报告 schema 见
[`hpc_ops/README.md`](../../Filament_python/tools/hpc_ops/README.md)。

## 三、HPC GitHub 连接：代理优先、bundle 回退

默认连接链为：

1. 读取权限为 `600`、当前用户所有的外部 secret 文件；只允许
   `http_proxy`/`https_proxy`（及大写、可选 `export`），不 source/eval 任意
   内容。
2. 设置大小写代理变量和 `GIT_TERMINAL_PROMPT=0`，要求 `timeout` 存在且
   秒数为 1–300 的正整数，以固定参数执行只读
   `git ls-remote URL REF`。
3. 代理通过后必须只接受唯一的 `<expected_head>\t<ref>` 行，随后核验指定
   branch、HEAD 和 clean worktree；来源标记为 `strict_remote_verified`。
4. 代理失败时，仅在提供 bundle、bundle raw SHA、expected ref、expected
   HEAD 且 `git bundle verify` 全部通过的情况下回退；报告标记
   `verified_bundle_non_strict`，不能写成 direct GitHub verification。
5. HPC 执行链只 clone/fetch；commit/push 保留在本地 Windows 工作区。

GitHub 源 URL 只允许无 userinfo/query/fragment 的
`https://github.com/<owner>/<repo>[.git]`。官方代理 URL 可以在外部 mode-600
secret 中带认证信息，但绝不能输出或写入 GitHub URL、仓库、日志或
manifest。项目约定的 secret 文件路径只是部署位置，不是仓库内容：

```text
/data/run01/scvi806/user_Wangjimin/.secrets/github_proxy.env
```

使用 `/data/run01/scvi806/user_Wangjimin/更新并提交代码.txt` 时，应先把其
操作说明迁移为不含密码、PAT、token-in-URL 或 `sbatch` 的版本，仅引用上述
权限受控 secret 文件和占位符；迁移前由用户/管理员人工轮换已经暴露的旧
凭据。Codex 不把任何真实凭据写进 Git、日志、测试 fixture、manifest 或报告，
也不在本手册读取该远端文件。

## 四、环境与 provenance

HPC preflight 必须显式检查并激活固定 Miniforge 安装下的
`Filament_python` conda 环境，再验证 Python、NumPy、CuPy、Git、
`sha256sum`、`sbatch`、`sacct`、`scontrol`。不得把登录节点默认 `python3`
的导入结果当作生产环境证明。

后续 provenance 使用 `filament.provenance.v2`：

- tracked text 记录 repo-relative path、HEAD 中的 Git blob OID 和
  canonical-LF SHA256；创建时要求 clean HEAD、已提交路径和 LF 工作树。
- external 文件记录 raw-byte SHA256；换行转换不能改变其 hash 语义。
- v2 validator 用同一 canonical-LF 算法，因此库层的非 strict 校验可以让
  Linux LF 与 Windows CRLF checkout 的 tracked 文本一致审计；CLI `validate`
  默认 strict，要求 repository identity、branch、HEAD 和 clean worktree，
  需要审计 CRLF 内容时显式使用 `--non-strict`。
- create 输出必须在仓库外、父目录已存在、目标不存在且不是 symlink；tracked
  path 也不得是 symlink。Git blob OID 不假定 SHA-1 的 40 位长度，并核验
  对象类型确实为 blob。
- 既有 v1 execution lock、receipt、bundle、二进制结果及 job `221822`
  provenance 保持原字节和原哈希，不重写、不覆盖。

## 五、失败门禁与交付

每个 preflight、staging 和 launcher 失败都要记录：症状、根因、是否产生
副作用、发现它的门禁、当时修复、永久控制和证据路径。失败发生在 lock
消耗或 `sbatch` 之前时，必须明确说明“未产生作业”；不能用成功的编译、
测试或 dry-run 推断物理有效。

交付前最小检查：

```powershell
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
bash -n Filament_python/tools/hpc_ops/hpc_proxy_env.sh
bash -n Filament_python/tools/hpc_ops/hpc_preflight.sh
git diff --check
```

新增 guardrail 测试应覆盖参数数组、account/root 拒绝、secret 脱敏、代理
失败与 bundle 回退、wrong-head rejection、canonical/raw hash、TOML 角色
权限、`hpc_git_source.sh` clone/fetch dry-run 和 v1 兼容性。测试
通过只证明软件门禁，不替代数值或物理审查。

## 六、设计依据

项目把不可违反的边界放在 `AGENTS.md`，把专责子 Agent 的角色、模型和
权限放在 `.codex/agents/*.toml`；详细操作模板留在本手册。该分层参考官方
OpenAI 文档：

- https://learn.chatgpt.com/docs/agent-configuration/agents-md
- https://learn.chatgpt.com/docs/agent-configuration/subagents
