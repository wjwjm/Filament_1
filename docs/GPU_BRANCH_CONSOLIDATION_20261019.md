> 2026-10-19 后续清理已完成：本项目 D:/Filament_1 现在直接使用 codex/gpu-optimization，无需新项目。
> GPU独立worktree已解除分支占用；统一分支已推送并跟踪origin。
> 已删除24个本地、54个远端已合并分支；保留CPU/main、含未提交改动的WP1及5个未合并远端分支。
> 历史worktree目录/证据均保留。恢复清单与已验证Git bundle：
> D:/Filament_1/review_evidence/GPU_BRANCH_CLEANUP_20261019T101733901/
> 下文是清理前的整合记录，其中旧分支名/SHA用于历史追溯。
# GPU 开发分支整合（2026-10-19）

统一分支：codex/gpu-optimization
工作区：D:/Filament_1/.worktrees/gpu_optimization

以最新GPU正式执行源码0d812434a4d6e73117674fdae4d5d192c1d00c15为第一父节点，
正常merge codex/hr4e5s-s5-lifecycle（23983a88fa84e196670f452a60e427a01c8d0c84），不squash、不改写历史。
其余HR-4/E5分支的提交已经是N3祖先。本次补入S5独立的14个提交，包括bootstrap/科学输入身份、
本地审计、阶段0可观测性入口及历史收口证据，同时保留N3当前journal/digest实现。
S5的before_hydro_block可选回调位于durable claim之后、manifest锁之外；E5默认不传该回调。
旧分支带入的Agent模型配置及AGENTS.md变更未采用，保留N3配置与本聊天最新指令。

## 当前任务边界

275230于2026-10-19 07:14:57 +08:00 FAILED / 1:0。
已知首错：H2 commit_next生命周期锁获取超时；底层持锁原因尚未确定。
三对actual exact、继承/promotion、逐发reclaim及科学terminal尚未完成；
E5_1C/E5_2保持OPEN。n3监测PAUSED。合并不是科学PASS或重提授权。
证据：D:/Filament_1/review_evidence/E5_N3_FORMAL_RETRY_20261018/TERMINAL_REPORT_20261019T093640023.md。

## 保留与后续使用

- 未来GPU开发从本统一分支继续；已有阶段分支/worktree作为历史证据入口保留。
- CPU分支与main保持原状；main独有的543499d是本地artifact保存提交，不并入GPU源码。
- 本次没有删除旧分支、worktree或证据，没有push或HPC操作。
- 当前整合仅本地生效。原分支名仍可用于定位旧执行SHA，不再需要逐一合并。
- 远端引用已经git fetch origin刷新；下表ahead/behind是相对整合前N3基线的提交计数。

## 分支归属

|引用|整合前SHA|N3独有 / 该引用独有|处理|
|---|---|---|---|
|HR-4E|d4bd96ca21c5815b57f1a3c2ee8f8cec54dfb220|55	0|已包含于N3历史，无需重复merge|
|codex/e5-1b-wp2-canonical-source|a22292d0860199a4ff36d919537203db4fafa963|16	0|已包含于N3历史，无需重复merge|
|codex/e5-1b-wp2-final-qualification|f04035db505b9e519d1a7cd6951c6562f90a7290|12	0|已包含于N3历史，无需重复merge|
|codex/e5-1b-wp2-gpu-binding|a22292d0860199a4ff36d919537203db4fafa963|16	0|已包含于N3历史，无需重复merge|
|codex/e5-1b-wp2-portable-identity|f04035db505b9e519d1a7cd6951c6562f90a7290|12	0|已包含于N3历史，无需重复merge|
|codex/e5-1b-wp2-recovery-redesign|49dd43b8cabbfcd24fa7a2fbc201233997a626a7|19	0|已包含于N3历史，无需重复merge|
|codex/e5-1b-wp2-source-rename-local-qualification|1120a47d4bb4ec2966a1502e73ef6ac89edae6af|13	0|已包含于N3历史，无需重复merge|
|codex/e5-n3-120h-admission|0d812434a4d6e73117674fdae4d5d192c1d00c15|0	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-formal-admission|c8f3adaf41aa46c104feda6d34de402dd501d436|10	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-formal-wait-binding-20261016|d8300006cd2f9382c494a0aa596240bc3837d006|5	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-job273606-h3-fix|0009679112f59d536c8b70921bbc997a6055e1a5|4	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-joint-cache-ledger-diag|7a3d8f0fa48cedec9bed2d9749e93f48d25929f2|8	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-joint-cache-ledger-import-fix|77edba869382bc3b3817e889a9aa01812f67ef83|7	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-streaming-journal-r1|46b6a446754e44b8b126911e759954720fe9438c|3	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-v5-ledger-accounting|706f6ac508bdbf6f25a9c034c71742def6c4f184|6	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-v5-ledger-accounting-r4|706f6ac508bdbf6f25a9c034c71742def6c4f184|6	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-v5-ledger-accounting-r5|706f6ac508bdbf6f25a9c034c71742def6c4f184|6	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-v5-ledger-accounting-r6|706f6ac508bdbf6f25a9c034c71742def6c4f184|6	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-v5-ledger-accounting-r7|706f6ac508bdbf6f25a9c034c71742def6c4f184|6	0|已包含于N3历史，无需重复merge|
|codex/e5-n5-v5-ledger-accounting-r8|706f6ac508bdbf6f25a9c034c71742def6c4f184|6	0|已包含于N3历史，无需重复merge|
|codex/hr4e5-e5-1a-streaming-implementation|079deb38f92bd5d25b952561440f85ac15efb34c|23	0|已包含于N3历史，无需重复merge|
|codex/hr4e5-e5-1b-wp1|079deb38f92bd5d25b952561440f85ac15efb34c|23	0|已包含于N3历史，无需重复merge|
|codex/hr4e5s-s5-lifecycle|23983a88fa84e196670f452a60e427a01c8d0c84|37	14|通过本次双亲merge整合|
|origin/HR-4E|d7e77cd380d48dafdab04d307a06264159afa958|100	0|已包含于N3历史，无需重复merge|
|origin/codex/e5-1b-wp2-final-qualification|0bb3866af90a08e5539c74e19f570c4529572219|14	0|已包含于N3历史，无需重复merge|
|origin/codex/e5-1b-wp2-portable-identity|f04035db505b9e519d1a7cd6951c6562f90a7290|12	0|已包含于N3历史，无需重复merge|
|origin/codex/e5-n5-formal-admission|c8f3adaf41aa46c104feda6d34de402dd501d436|10	0|已包含于N3历史，无需重复merge|
|origin/codex/hr4e5-e5-1a-streaming-implementation|079deb38f92bd5d25b952561440f85ac15efb34c|23	0|已包含于N3历史，无需重复merge|
|origin/codex/hr4e5s-s5-lifecycle|57da4d6665358ec4b706e871a67ccb66a2525f92|37	13|通过本次双亲merge整合|
|origin/codex/phase5-ionization-propagation|7c43efff481486276988a10ba0952cfb35b35a20|301	0|已包含于N3历史，无需重复merge|

## 整合检查记录

- 合并前最新N3与S5三方合并无文本冲突；人工检查发现S5 ready回执及clean audit仍以旧checkpoint文件哈希比较journal状态哈希，需要兼容修正。
- 初次较宽定向集合出现S5相关失败，并进入耗时的持久化三对fixture；已主动中止，不能记为通过。
- compileall、backend、sanity通过。
- journal / streaming / N3定向回归：95 passed，3 skipped，625 deselected（49.08秒）。
- 原始本地日志：D:/Filament_1/review_evidence/GPU_BRANCH_CONSOLIDATION_20261019/journal_regression_1006.log。
- 所合并的artifacts仅为S5历史小型文本/JSON和batch证据，未重写或生成冻结科学数组；原有提交保持可追溯。
## 最终验证与合并兼容修正

S5 ready回执现在使用lifecycle.state_sha256并显式绑定hash scope；
clean audit比较同一scope下的authority状态身份，末尾重新刷新校验。
仅旧raw格式允许缺省scope；journal缺失或错误scope拒绝。冻结历史回执不改写。
回归覆盖journal ready、scope篡改、legacy ready/audit兼容和负向缺失产物。
测试夹具改为读取authoritative snapshot；需要破坏文件的负向测试仅修改临时副本。

- S5 final/audit/phase0：62 passed，666 deselected（57.17秒）。
- journal/streaming/N3：95 passed，3 skipped（49.08秒）；跳过保持原测试条件，不代表GPU验证。
- 最终compileall与sanity通过；backend通过。独立只读审查未发现修正后的阻塞问题。
- WSL调用Windows worktree的旧脚本复现测试通过调用环境GIT_DIR/GIT_WORK_TREE和WSLENV路径映射运行；未改生产脚本。
- N3 journal实现、生产驱动及传播核心与合并前一致。未修复275230锁超时，未运行HPC或科学生产验证。
- 日志目录：D:/Filament_1/review_evidence/GPU_BRANCH_CONSOLIDATION_20261019/
  - merge_compat_fix_targeted_r3.log：最终S5集合
  - journal_regression_1006.log：journal/streaming/N3集合
  - wsl_worktree_test.log：WSL环境修正后的单项检查
历史失败/中止日志保留，不计入通过结果。