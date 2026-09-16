# Formal E5 路线与授权图

**E5-0 状态：** `CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED`（2026-09-16 人工审核接受）。跨发设计为 `DESIGNED / NOT_IMPLEMENTED`；E5-1 未开始，实施与执行仍未授权。

`architecture_choice=USER_FIXED_EXISTING_STREAMING`；`intra_pulse_optical_hydro_overlap=REQUIRED`；
`production_hr4c_replacement=false`；`implementation_authorized=false`；`formal_execution_authorized=false`；
`resource_execution_gate=NOT_RELEASED`；`s5_entry_gate=ENTRY_BLOCKED_BY_S5_FINAL`。

原准备合同的源码审阅及 start HEAD：`cbfe2e38172b5221d739df80b032e4dee9fab7e3`；S5 execution SHA（历史回执）：`cd456ff8413cbc041d2d60b9b64007a1554028a1`。
两者之间只有文档/旧包差异，本轮未实时查询 S5，244700 的 PENDING 仅为原 HPC 快照。
原审核包绑定文档 SHA `c377178e909d795b1fce0d4d162ab14845c9c264`；本次仅同步人工接受状态，不重打包、不改包内历史状态，不得把文档 SHA 当执行 SHA。

| 阶段 | 本次明确状态与进入条件 |
|---|---|
| E5-0 | Streaming纠偏、接口/PRE0/连续prefix N3/exact/资源保留合同完成并人工接受，CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED |
| E5-1 | NOT STARTED / IMPLEMENTATION + RESOURCE GATES PENDING；工程三发、两次rollover、末发POST、一次跨allocation恢复；前置工作包是已单列的glue实施/测试，无新增长期阶段 |
| E5-2 | 正式规模/耐久与科学结论NOT_STARTED；N、终点、C、资源和科学问题另审 |

当前生产权威为既有Streaming；每screen POST后可马上入队hydro，光学继续。全域barrier/promotion后才下一发。
HR4C仅隔离Batch reference，旧主导方案 SUPERSEDED / NOT CURRENT PLAN。新根复制CURRENT、outer index只引用promotion；末发不hydro。
推荐PRE0=E1B delta_n prefix0..8047 + float64零vx/vy；N=3、z0..0.8047999999999277、K8048，
直接切原schedule，不改步长。此为同域工程candidate，非全z科学等价。
两轨科学payload=669,587,300,928 B (623.601769 GiB)；含保留/临时/余量的建议可用容量=851,693,145,010 B (793.201053 GiB)；resource_execution_gate未释放。

派生input/保留方案的准备合同已人工接受；当前等待单独实施授权、真实资源批准和另行提交授权；S5 PASS与preflight是证据门。
本轮无runtime/science/job变更。S5状态仅原快照，不因设计收口而关闭或干预。

审查入口：[closeout matrix](E5_0_CLOSEOUT_MATRIX.md)；接口、freeze JSON、resource CSV、glue草稿和E5-1文档为同一合同。


正式结论：[E5-0 closeout](E5_0_CLOSEOUT_20260916.md)。
