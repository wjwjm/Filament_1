# E5-1前最小glue实施任务草稿（本轮不执行）

**E5-0 状态：** `CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED`（2026-09-16 人工审核接受）。跨发设计为 `DESIGNED / NOT_IMPLEMENTED`；E5-1 未开始，实施与执行仍未授权。

`architecture_choice=USER_FIXED_EXISTING_STREAMING`；`intra_pulse_optical_hydro_overlap=REQUIRED`；
`production_hr4c_replacement=false`；`implementation_authorized=false`；`formal_execution_authorized=false`；
`resource_execution_gate=NOT_RELEASED`；`s5_entry_gate=ENTRY_BLOCKED_BY_S5_FINAL`。

原准备合同的源码审阅及 start HEAD：`cbfe2e38172b5221d739df80b032e4dee9fab7e3`；S5 execution SHA（历史回执）：`cd456ff8413cbc041d2d60b9b64007a1554028a1`。
两者之间只有文档/旧包差异，本轮未实时查询 S5，244700 的 PENDING 仅为原 HPC 快照。
原审核包绑定文档 SHA `c377178e909d795b1fce0d4d162ab14845c9c264`；本次仅同步人工接受状态，不重打包、不改包内历史状态，不得把文档 SHA 当执行 SHA。

正式收尾结论与 G1–G6 门见 [E5-0 closeout](E5_0_CLOSEOUT_20260916.md)。

## 唯一实施边界

所有拟名均为 NEW_GLUE_REQUIRED / NEW_GLUE_DESIGNED_NOT_IMPLEMENTED，不虚构现有API。
新外层文件建议 `Filament_python/KHz_filament/hr4e5_formal_entry.py`，只衔接既有Streaming；
不创建HR4C adapter或第二执行器。S3/S4R/S5原入口和guard继续原样。

| 具体拟改文件/符号 | 必要性、源码依据 | 最小测试 |
|---|---|---|
| new formal_entry `StreamingPulseReadView` | S3:342–356当前读NPY；改用Streaming.current_fields绑定本发PRE，read/update协议与科学调用不变 | 实际读PRE、NEXT不可串入、速度继承、输入只读 |
| same `prepare_pulse_root` | Streaming.create:342–406需要不存在根/三整卷；host三数组G装载前代NEXT，create后逐screenexact并写READY | PRE0半成品不可见；完整但缺receipt可补；partial根转新attempt |
| same `resume_lineage` | open/current_fields/promotion:409,828,1228；index仅引用权威pointer | pointer先于index、两次换代、冲突fail、restart_epoch不推进pulse |
| same `run_pulse` | S3:409–474构场/透镜/真实propagate；选原schedule前缀直接传入、fresh copy、非末发复用hook/worker | source不alias、dz逐值相同、2次rollover、同发overlap事件 |
| same `final_post_hook/validate_final_post/resume_final_post` | commit_post与has_authoritative_post可复用；S3 hook总enqueue、barrier总需NEXT，不能照用末发 | 末发不enqueue/无hydro/NEXT；partial replay保留POST；receipt恢复幂等 |
| existing Streaming模块仅新增窄只读POST验证封装（如需） | 私有 `_validate_record_provenance/_assert_no_staged_or_orphaned_artifacts` 受锁；不手改manifest | 孤儿/临时文件、重复/漏screen、坏hash必须拒绝；不改变非末发guard |
| new formal_entry reference/comparison helpers | S3:495–520/571–627隔离Batch与四项exact；参数化K、继承PRE速度、末发没有NEXT | reference不清零后续速度；42K+27+3比较计数；故意mismatch定位 |
| existing launcher参数衔接草稿（实际文件依据实施时固定） | S5_final.sbatch串行bootstrap/identities/worker启动顺序；增加pulse namespace与有序allocation续接参数 | batch-entry audit、无early裸python、old writer quiescence、无重复bootstrap |
| new `tests/test_hr4e5_formal_entry.py` | 仅mock/小数组CPU contract tests | N3、所有新中断窗口、末发bootstrap、zero mismatch门；不重跑S5矩阵 |

末发bootstrap不能调用会重排POST到hydro的常规reconstruct_queue；新增只读/光学恢复分支仅末发启用，
Batch对照的装配/导出按已有 `write_staging_batch/read_authoritative_batch` 用8-screen分块，避免
旧run_batch_hydro的全卷zeros_like；其非首发POST速度来自真实PRE，两个工作slot仍完整计盘。
读取CURRENT的返回数组要设只读或保持内部独占，不能把可写别名暴露给hook。K=8048须整除8；拒绝不足block的candidate。
与正常非末发路径隔离。现有make_post_commit_hook/selected hook默认enqueue保持原行为。
不修改propagate、HR2/3/4、ionization、Raman、精度、默认配置、LUT或源数组。
若需要改变这些行为则BEHAVIOR_CHANGE_REQUIRES_REVIEW，停止实施并返回证据。

## 验收门与顺序

人工准备合同已接受；仍须 S5-FINAL terminal PASS，并另获实施授权；一次最小实现→专用解释器compile、wrapper backend/sanity→
小型定向tests→独立数值/接口审查。新增launcher必须audit_batch_entry，远程strict provenance在run/lock/sbatch之前。
这份草稿既不授权实现也不授权PRE0物化/提交。工程门：finite、字段/坐标/shape/dtype一致、array exact零mismatch，
每pulse K POST、非末发K NEXT、计数3/3/2、无漏/重、PRE0/lineage/末发幂等、overlap事件成立。
资源门按CSV：配额/空间+25%，RAM/VRAM实测≤80%，time模型适配有效QoS。物理趋势与性能加速比不作硬门。
