# CPU-S4-3G3-2 检查点与大尺寸确认记录

源码检查点已保存并普通推送到 `checkpoint/cpu-opt-g3-smallgrid-20261026`。
原小场源码检查点为 `98194d2e07991ac60a1cfb24f002b7cdfe8ab388`；
本次第二次大场执行源码为 `0349f82f8600cbbc16a8b16dea7d84c13574181d`。
科学子树仍为 `821fec5bdacd832b0347e1f2e84285cb54172e57`。
实际 Git HEAD、历史 deployment 和科学子树按各自范围绑定，没有伪造版本查询。

原 R14 包的 24,702 个成员、原字节身份、CRC、路径和 manifest 覆盖已核验；
原只读复算器及独立原精度 CSV 复算通过。原小场增量中位为 6.7521672435%，
累计中位为 13.2148794300%，仅属于历史小场。未重跑小场 HPC 性能矩阵。
独立干净 checkout 原检查点 Windows/Linux 各 47 项通过；
追加大场入口与兼容检查后，当前代码 Windows/Linux 各 50 项通过，无失败或跳过。
本地实际 workers=1，不是 W48 科学资格。

| 范围 | 本轮结果 |
| --- | --- |
| G2_FULL 连续入口八区间 | 已执行，8/8 持久化 POST |
| G2_FULL 对原合格 G2 的终态 | 1120 行：1026 数值 exact、90 元数据 exact、4 原计时例外；native layout/alias 通过 |
| G2_FULL 首边界 | 历史 JSON 读取触发 `BOUNDARY_SCALAR_ORDER`，未进行全元素比较 |
| G3_FULL 大场传播与资格 | NOT_RUN |
| B1/B2/B3 增量计时 | 全部 NOT_RUN，无中位收益 |
| 完整大场科学资格 | NOT_ESTABLISHED_BOUNDARY_SCHEMA_BLOCKED |

历史首边界的同七个标量键按 JSON 字典顺序保存为
`U0_baseline,U_now,adaptive_rejection_count,dz_try,interval_index,safety_mode_trigger_count,z`，
读取器要求 `z,interval_index,dz_try,U0_baseline,U_now,adaptive_rejection_count,safety_mode_trigger_count`。
此次失败发生在原数组比较之前，不是已观察到的科学数值 non-exact。
同键集合不构成完整科学 exact；未改历史 receipt、oracle、读取守卫或容差。
没有新增非计时差异例外，也未使用存储 bool bridge。

| JobID | 源码 | 终态 | allocated 秒 × CPU | 核时 |
| --- | --- | --- | --- | --- |
| 8851763 | a17215b | FAILED 1:0：Linux CSV checkout 原字节守卫 | 74 × 96 | 1.9733333333 |
| 8851774 | 0349f82 | FAILED 1:0：历史边界 JSON 键序守卫 | 478 × 96 | 12.7466666667 |

合计 14.72 allocated 核时，两次均为单节点、96 物理核、0 GPU、320 GiB、75 分钟上限。
最大同时在途 1；两作业终态及无在途已核验。2/2 JobID 已用满，未提交第三个作业。
第二次实际 cgroup high-water 为 65,727,205,376 bytes，低于 274,877,906,944 bytes 停止线。
Slurm step 未重复计费。没有完成的正式 application_service_s 样本，不能以进程耗时替代。

工程修复保留为独立提交和私有启动收据：真实 Git 只读兼容查询、实际 producer 模块名
进入私有缓存身份、固定映射文件 CRLF checkout、实际 source-root 环境绑定。
冻结数学内核与 G3 controlled_callsite 保持原字节；未开发新的数学优化。
属性变更后切换已有 checkout 不一定重写未变化的 CSV，应在未运行的新 staging 中用
真实 Git 重新取出该路径并验证其冻结 raw SHA，不能改守卫。

下一步优先回到固定小基准复现边界 JSON 键序兼容问题。本任务未实施该修复、
未重新比较失败后的大数组，也未启动新的大场确认。

DEFAULT=I0。FULL_DISTANCE / FOCUS / MULTIPULSE / PRODUCTION=NOT_QUALIFIED。
REVIEW_GATE=MANUAL_REVIEW_PENDING。大场 G3 增量收益未知；没有同期 I0 性能，
不推算大场累计收益。旧 QA 与历史失败记录均保留。
