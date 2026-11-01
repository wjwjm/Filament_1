# Eq10/P48 原尺寸波动原因诊断

用户于2026-11-01授权增加测试及多次CPU作业提交，目标是解释既有原尺寸Eq10/P48结果的不稳定性。本研究独立于已结束的S4-6F，不改写其负结果、不合并新旧性能样本、不切换生产默认。

从干净ec354f345b31e1e88b422757f05c2ac7dfb697c2建立 `codex/cpu-eq10-variability-20261101`。保持科学子树821fec5bdacd832b0347e1f2e84285cb54172e57、冻结生产输入/PRE、[384,351,301] complex128、空1.5GiB线性缓存、原数学与48线程Eq10实现。

首轮复用6F完整14窗入口和比较器，新增独立于传播子进程的5秒标量观察，15秒采集NUMA驻留页分布。记录PID/TID调度、线程CPU与上下文切换、缺页、CPU频率可用描述、节点上下文及采样器自身成本。无科学数组读取或保留，无sysctl/affinity/NUMA策略变更，无生产配置或公开API改动。

6F引擎的结果/计数/时钟仍按原定义输出，但整个新作业标记为DIAGNOSTIC_ONLY：其旧资源字段和性能门作为复用引擎的记录保留，本研究授权与资源预算以新AUTHORIZATION/DIAGNOSTIC_CONTEXT和资源账为准。新结果不替换S4-6F的四对样本，也不自动成为生产资格。

资源逐次审批由父级执行：最大在途1个；首个作业96物理核、128GiB内存、2小时、0GPU，工作截止7080秒及实际allocation/cgroup较小值80%内存线。根据上轮5964秒，预计约95–100分钟。后续最多再做2个有明确检验因素的诊断作业，每次冻结合同并复核资源；无自动重试或requeue。正常诊断预留上限576 allocated核时，实际按scheduler终态记账。

磁盘按当前单个作业逐项估算加20%余量；初始账户剩余约68.6GiB。旧四作业的128GiB磁盘总预留已结项，不冒充本轮单作业预算。保留所有历史证据；下一作业空间不足时停止提交，不能自动删除旧数据。

软件测试先规定runner backend/sanity，再观察器定向测试、实际W1微网格工程入口与语法。独立审查后冻结SHA、provenance_v2和真实batch-entry，才可提交。科学非exact、原生命周期失败、来源不符或资源保护线触发立即停止；采集失败单列工程/诊断失败。缺失内核指标明确UNAVAILABLE，不填零冒充有效测量。

只将观测关联解释为关联。线程CPU/wall、NUMA驻留页和CPU MHz采样不能单独证明GIL、NUMA访问带宽或频率因果。需要完整helper/FFT profile时另设对称OFF/PROFILE验证，不能静默放宽6D的profile=False合同。

首轮Job8860011已完成14窗exact及离线复算：body中位改善4.9442%，应用四对平均净省13.7506秒，未重现旧慢配对。缺少可用runqueue/PSI计数，worker节点位置变化只提供下一项检验线索。结果和限制见OBSERVATION_A01_FINDINGS.md。

第二轮私有模式MEMORY_POLICY_CROSSOVER只改变整个子进程的default/interleave-all内存策略。实际14窗由独立POLICY_PLAN.json及其context raw hash绑定，不能冒充旧6F矩阵。资格I:R2/X2；早晚R/R为D；正式两块D:R/X、I:X/R，然后I:R/X、D:X/R，平衡policy、角色和执行顺序。原科学入口/比较器/数学不改，D/I都须有实际body内策略及CPU/Mems允许集合证据。策略变快、变慢及不超过R/R漂移的未决分别记录，不按旧3%门接受混合策略。

第二轮原application/PROPAGATION/process_service时钟不移动、不用body总和替代传播跨度；额外launch锚从Popen前开始，覆盖numactl前缀与进程启动。源码、context与新输出根另行冻结，源增量从首轮3846d179d6fdcefd5ab631a8822883a7e5745243构建；合同见NUMA_POLICY_DIAGNOSTIC_CONTRACT.md。结果仍为DIAGNOSTIC_ONLY。

当前进度读取STATE.json与MONITOR.md；旧记录回溯在validation/old_eq10_scalar_audit。DEFAULT=I0，PRODUCTION_QUALIFIED=false，MANUAL_REVIEW_PENDING。
