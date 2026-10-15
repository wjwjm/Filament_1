# CPU-Q3 私有两发与独立重载执行器

任务：FILAMENT-CPU-Q3-BI-TWO-PULSE-RELOAD-T0-20261015。基线 b75becfe3a9b0ff6e381727fb516aa761d37c051；科学树 821fec5bdacd832b0347e1f2e84285cb54172e57。

## 固定执行合同

执行顺序 B_P1 → B_H1 → B_P2_CONT → B_P2_RELOAD → I_P1 → I_H1 → I_P2_CONT → I_P2_RELOAD。两个 normal-chain 各在独立解释器中串联；每个 reload 必须在对应 normal-chain 退出后由另一个解释器从磁盘 NEXT1 重建。仅六个真实八区间窗口、两批各八个真实 HR-4 截面，每截面1000步。B/I各自计算并保存NEXT1。

B/I固定时间FFT48、空间FFT24；I内核24、NtB、block32、generic/workqueue；BLAS1。CPU-only，一节点96核320G2h，无requeue。Q3自身无sbatch调用，外部唯一提交脚本以独占intent守卫提交；不自动重试。

## 复用与私有适配

- `cpu_p1r2_j1._run_window`：完整冻结八区间传播；Q1私有observer克隆在完整接受边界取证。
- `q1_t0_real._observer_class / _stage_endpoint_comparison`：首边界、终点完整字段和有界比较。
- `cpu_o1.make_joint_propagator_factory`：B/I私有符号分派；不改科学模块、np.fft或xp。
- `cpu_p1r2_j3._process_hydro_workers`：八个exec工作进程，绑物理核后加载，单线程执行原HR-4。
- `_prepare_q3_context`：私有复制J3上下文函数命名空间，仅把历史解释器常量绑定到已通过`_check_expected_python`的本轮锁定解释器；原J3/P1R模块不变。
- Q1提交前门私有副本只更换Q3存储cap校验，硬上限16GiB/50000文件；保留原14对象、源码、环境和allocation门。

## 状态与比较

每窗重新校验、加载sealed source并创建副本。P1绑定PRE0；P2绑定本分支NEXT1全部三场及八截面，实际传播入口逐场hash再次核对。NEXT1通过临时写入/校验→不可覆盖发布→receipt→COMPLETE标记发布；内存与磁盘数组、实际读取的磁盘元数据直接比较。P2前后和reload后检查不变。

`q3_comparison_contract.json`在执行前冻结31组；光学105/235字段模板源自已审Q2。所有原始行需实现/case/reference/stage身份及完整唯一字段。I的NEXT1相对B比较在I_P2_CONT开始前完成。有限非exact也停止；不放宽六项NaN、四项计时例外。HR4新增科学数值必须有限，None不能作为元数据豁免。

## 监督与时限

父进程直接文件日志、每秒进程树RSS/PSS及cgroup采样；各阶段以单调时钟计时。首区间900s、单窗2700s、H1外层900s（原worker批次180s更严格）；allocation保留900s收尾。RSS80%锁住后续阶段，90%有界终止整个子进程组。所有部分结果保留。正常链中每个窗口开始前按已有完整服务时间估计或2700s上限检查预算。

## 前门与调用

登录节点仅`--dry-run`、`--entry-smoke`、静态身份/依赖/ABI/batch审计。`--run`必须在唯一allocation中；先执行节点前门、B/I固定48/24/24接线tiny及Q1/Q2/Q3回归，再进入真实窗口。

本地测试通过仓库 `run_local_tests.ps1 -PythonExe <cpu_o1_py312> -Mode backend`、sanity、targeted。目录级注册已包含新测试，无需重复注册。tiny模式使用既有full Eq27/Heun/Strang传播与真实HR4、全新解释器重载；它只使用合成小数组，不访问封存真实输入。其内核线程受本地Numba线程上限约束；真实窗口始终24。

阶段回执：`<case>/q3_case_receipt.json`；`stages/<branch>_CHAIN/{h1_receipt,chain_receipt}.json`；`stages/<branch>_RELOAD/reload_receipt.json`；监督与实际命令在各stage目录。完整汇总`coordinator_receipt.json`和`NUMERIC_GATE.json`；失败也写回执并退出非零。核验还需独立scheduler终态、全部原始比较和库存，不能只看exit0。

服务包含source/PRE校验、初始化、传播、持久化及自身验证，跨身份比较单列；service总量减cross是同口径非cross服务量。传播含首边界比较；不可把包含时间与子项相加。哈希阶段内存不受64MiB指标chunk承诺约束；未知计量不记零。

最终仅`Q3_READY_FOR_MANUAL_REVIEW`或明确失败/部分状态。CPU_N5_PRODUCTION_QUALIFIED=NO；不授予CPU/GPU等价、焦区、长程、多发生产或任意中断接管资格。