# CPU-O1 下一轮 HPC 申请草案

状态：NOT_SUBMITTED / NOT_AUTHORIZED。本轮提交为0；P1系列11/11和P1-R2的4/4已用完，历史189.20分配核时不得结转授权。

|阶段|固定任务|申请上限|
|---|---|---|
|Q1|tiny入口/依赖检查；真实 accepted 区间热点输入；BASE_CPUFFT与+I局部及合法prefix8回接|1节点96核、2小时|
|Q2|REF_NUMPY独立对照；BASE_CPUFFT和主候选 B→C→C→B 独立fresh source完整8区间|1节点96核、2小时|
|Q3|BASE_CPUFFT与合格候选各两发+一次第二发独立reload，真实8截面HR4一次|1节点96核、2小时|

默认串行，最多3次submission，无自动重试，失败计数；合计6节点时/576 allocated core-hours。实际队列和完工ETA未知，不能沿用历史排队。内存拟320G但须现场核验。仅t0s000727/bscc-t/v6_384/normal，无GPU；账户合计不得超过10节点960核。

新增持久数据申请≤8GiB、系列累计申请≤40GiB；这是新申请，旧32GiB不自动扩容。提交前按实际inventory/quota/inode重新核算；环境、cache与数组均计入实际预算，审阅包另列。无额度/输入身份不全则阻断，不删旧证据。

Q1 kernel线程1/24/48，只有48较24的中位kernel时间降低且无错误时才试96；每档预热1、计时2，最佳补足3次。FFT固定既有48/24，不重复61条微基准。不以局部最优代替完整服务。首accepted区间在合法8区间传播内观测，不构造prefix1。

本轮只实现I；Raman+R/组合只有在另行完成实现、接口审查和tiny数值门之后加入必要消融，否则SKIPPED_NOT_IMPLEMENTED。不得把设计完成当成+R已可运行。

Q2优先完成基线和候选配对；每窗独立进程、同封存输入。保存所有样本，比较传播和完整服务两个层面，编译/转置/线程池/校验均纳入适当成本。Q3每实现fresh source两次、optical两发、hydro一次，不在第二发后追加hydro；同实现reload exact。跨实现有限差异只允许诊断，不产生科学容差。

首区间上限15分钟、每8区间45分钟；剩余15分钟不启动新长case，且需实测预测覆盖下一case与收尾。输入/后端/物理硬门/拓扑/线程违规立即停止；RSS80%不再开case，90%有界终止并保留局部证据。提交失败或不明确先查本人accounting，不自动重提。

预提交须验证Linux实际解释器/动态库/线程层/ABI及缓存新身份、小数组全算子、batch-entry审计、bash -n、真实子进程入口、sbatch --test-only；本轮均未执行。

同时报告REF_NUMPY→CPU_O1和BASE_CPUFFT→CPU_O1。335s/255s仅研究里程碑，非承诺或科学门。EARLY_PREFIX8不能代表焦区、后续脉冲、K8048或FULL15000；本方案不制备缺失远z样本，不开展Hydro节点矩阵。
