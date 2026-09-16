# E5-0 authority/interface：用户固定 Streaming 主线后的本次有效合同

**E5-0 状态：** `CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED`（2026-09-16 人工审核接受）。跨发设计为 `DESIGNED / NOT_IMPLEMENTED`；E5-1 未开始，实施与执行仍未授权。

`architecture_choice=USER_FIXED_EXISTING_STREAMING`；`intra_pulse_optical_hydro_overlap=REQUIRED`；
`production_hr4c_replacement=false`；`implementation_authorized=false`；`formal_execution_authorized=false`；
`resource_execution_gate=NOT_RELEASED`；`s5_entry_gate=ENTRY_BLOCKED_BY_S5_FINAL`。

原准备合同的源码审阅及 start HEAD：`cbfe2e38172b5221d739df80b032e4dee9fab7e3`；S5 execution SHA（历史回执）：`cd456ff8413cbc041d2d60b9b64007a1554028a1`。
两者之间只有文档/旧包差异，本轮未实时查询 S5，244700 的 PENDING 仅为原 HPC 快照。
原审核包绑定文档 SHA `c377178e909d795b1fce0d4d162ab14845c9c264`；本次仅同步人工接受状态，不重打包、不改包内历史状态，不得把文档 SHA 当执行 SHA。

正式收尾结论与 G1–G6 门见 [E5-0 closeout](E5_0_CLOSEOUT_20260916.md)。

## 1. 本次有效结论与证据等级

Streaming CURRENT/POST/NEXT、逐 screen 提交、队列、claim、恢复、barrier 和 promotion 保持生产权威。
同一发 optical–hydro 重叠必须保留；下一发等待前发全域 barrier/promotion。
HR4D 仅提供已冻结生命周期和时间步规则，HR4C 只用于隔离 Batch 对照。
旧 c92118f 的 HR4C 主导方案及全 POST 后 hydro 顺序现已 **SUPERSEDED / NOT CURRENT PLAN**；旧 ZIP/Git 记录保留，
旧方案 generation-0 装配问题撤出当前任务，不再建立修复工作包。

`EXISTING_VERIFIED` 表示下面固定源码确有接口/既有资格有其限定范围，绝不表示新增 N=3 路径已 PASS。
`NEW_GLUE_DESIGNED_NOT_IMPLEMENTED` 是本次设计完成但尚未实施；`PENDING_EXTERNAL_EVIDENCE` 是 site/运行事实待证。

## 2. 真实调用表（全部路径相对 Filament_python/KHz_filament，固定 SHA `cbfe2e38172b5221d739df80b032e4dee9fab7e3`）

| 环节 / 符号及行段 | 实际入参 → 产物 | 权威 / 分类 |
|---|---|---|
| `hr4e5s_streaming.py:342–406` `StreamingLifecycle.create` | root、三个 `[K,Ny,Nx]` arrays、screen_records、current_generation、dx/dy、queue_depth → CURRENT NPZ + manifest | EXISTING_VERIFIED；拒绝已存在根；np.asarray 需要整卷接口，非迭代器 |
| `hr4e5s_streaming.py:409–556,828–834` `open/current_fields` | root / ordinal → 校验 pointer 后 CURRENT 或 NEXT 三场 | EXISTING_VERIFIED；promotion 不移动文件 |
| `hr4e5s_s3.py:342–362` `S3ReadOnlyCurrentState` | 原始 NPY路径 → read_interval / update_interval | EXISTING_VERIFIED；旧入口一直读原始 NPY，不能直接当下一发 PRE；需新只读 view |
| `hr4e5s_s3.py:409–474` `run_optical_path` | manifest、out_dir、stream root、fp64 → 构场/透镜、真实 propagate 调用 | EXISTING_VERIFIED；固定单发/full source 守卫；正式 outer entry 调用同构场和 propagate，不偷改 S3 guard |
| `propagate.py:233–307,627–634` `propagate_one_pulse` | E、longitudinal_schedule、thermal_slow_state、hr3b_parameters、post_commit_hook 等 → final E / diagnostics | EXISTING_VERIFIED；支持既有 schedule 对象且用 interval.dz |
| `hr4e5s_s3.py:348–356,373–384` read/update/hook；`hr4e5s_streaming.py:857–966` | interval → PRE.delta_n + HR3 increment → commit_post_from_delta_n → enqueue_post | EXISTING_VERIFIED；速度从本发 CURRENT copy，不置零；耐久 POST 先于入队 |
| `hr4e5s_streaming.py:1028–1125` `claim_block/run_one_hydro_block/commit_next` | actor、dt_hydro、n_hydro_steps、chi/nu/n0/gravity/CFL → 互斥 block 与 NEXT | EXISTING_VERIFIED；复用 frozen block=8、queue=16、worker ownership |
| `hr4e5s_s3.py:523–561` `consume_streaming/finalize_streaming` | lifecycle_root、hydro、producer_complete → worker 循环、barrier/promotion | EXISTING_VERIFIED；非末发复用；末发不可调用 finalize_streaming |
| `hr4e5s_streaming.py:1178–1266` `validate_barrier/promote_next_to_current` | 完整 K records → barrier + authoritative_generation.json | EXISTING_VERIFIED；不因末发而放宽此门 |
| `hr4e5s_s3.py:80–255` bootstrap/receipt；`hr4e5s_streaming.py:836–847,1127–1176` | 静止旧 writer、记录/manifest → 单次串行 reconstruct + durable POST reuse | EXISTING_VERIFIED；新跨发 epoch 绑定 NEW_GLUE_DESIGNED_NOT_IMPLEMENTED |
| `hr4e5s_s5_final.py:66–91,145–204` identities/bootstrap；`tools/hr4e5s_s5_final.sbatch:105–118` | quiescence + effects → bootstrap 再启动 workers | EXISTING_VERIFIED 的 S5 接口/顺序；其故障 case-specific receipt 不冒充 E5 跨发 receipt |
| `hr4e5s_s3.py:495–520,571–627` Batch / compare_exact | 光学 POST档案 → HR4C evolve + NEXT；shape/dtype/hash/array exact | EXISTING_VERIFIED；硬编码48与速度置零必须在隔离对照 glue 中改为 K/真实 PRE.v，旧 S3不改 |

## 3. 本发 PRE、POST 与重叠

NEW_GLUE_DESIGNED_NOT_IMPLEMENTED：新增 `StreamingPulseReadView`（拟名，不是已有函数）在本发根
打开 CURRENT，绑定 current_generation/current_content_sha256、source_index/z；read_interval(k) 从
`current_fields(k)['delta_n']` 获取只读 slice；update_interval 返回 PRE+increment，不写 CURRENT。
本发禁止 promotion，读 view 在每次读取校验相同身份。保留原科学调用的热参数、deposition contract、精度和算子。
真实PRE读取位于 `propagate.py:960`，HR3 map/update/hook位于 `1182–1217`；不是仅在receipt里声明PRE。
原 `runner.build_transverse_input_field` 位于 `runner.py:123–202`，输出轴序为 `[Nt,Ny,Nx]`。
构造一次不可变 E_source（原输入面、原薄透镜），每发 E_source.copy()；输出绝不回灌下一发。

非末发 hook 沿用 deposition_finalized → commit_post_from_delta_n → enqueue_post(wait_for_capacity=True,timeout_s=None)。
K=8048为8的整数倍，正常claim无需新增尾块行为。重叠证据使用既有 `HYDRO_BLOCK_START` 或
`HYDRO_SCREEN_START` 事件（不是假定存在名为HYDRO_START的接口），与同pulse OPTICAL_COMPLETE比较。
提交的 `vx/vy=PRE.vx/vy`，不能复用旧 Batch 每发零速度。workers 与 optical 同时运行；不增加全 POST 门。
必须捕获同一 pulse 的 OPTICAL_START、首次 HYDRO_START、OPTICAL_COMPLETE，满足 hydro_start < optical_complete；
事件缺失则新路径 overlap 验证未通过，不用人工 sleep 或加速比代替。

## 4. 唯一跨发 lineage 与新根（NEW_GLUE_DESIGNED_NOT_IMPLEMENTED）

推荐 `candidate/pulse_000/attempt_000/lifecycle`、`pulse_001/attempt_000/lifecycle` 等独立根。
每发根只执行一次既有 CURRENT→POST→NEXT；不清空旧根，不在 promotion 后原根再次 create。
跨发衔接采用 **实际复制**，不假设 hardlink、symlink 或零成本共享：在仅 coordinator 活跃时从前代
`open(...).current_fields(k)`（已指向 NEXT）分 screen 读出，填三个独立 float64 host arrays，
逐字段 exact 核对前代，传给现有 create。新 CURRENT 的数组身份等于前代 NEXT，文件字节哈希可因 namespace 元数据不同而不同。
内存计入 G，复制出的新 CURRENT 计入磁盘 G；父代所有根仍保留。

generation 是既有 opaque string：PRE0=`E5:<id>:PRE0`，后代 current_generation 取前代 pointer.authoritative_generation；
next_generation 由既有 `:next` 规则生成。pulse_index 是0-based调度序号，restart_epoch/initialization_attempt 是独立字段，
source_generation 是前代权威凭证；绝不强套 HR4D 的 2p。
外层 `pulse_lineage.json` 只缓存 previous pointer/file hash、new root/current content hash、ready receipt 和执行进度；
不能独立 promotion 或使 index 超前。消费入口只接受已验证 ready 引用。

| 中断窗口 | 幂等恢复规则 |
|---|---|
| pointer 已落盘、前代 manifest promotion 未落盘 | open + 原 promote_next_to_current 补齐原事务；不得第二次推进 |
| 前代 promotion 已落盘、跨发 index 未写 | 静止性与 barrier 校验后，由 pointer 重建同一个后继身份；index 是派生记录 |
| 已创建部分 next-root、未产出 ready | 不可作为 PRE；保留隔离的 attempt；新 attempt 在新不存在路径 create，记录旧 attempt。不删除、不覆盖；额外失败副本进入预算再获授权 |
| 新根完整、ready 或 index 未落盘 | 从完整 manifest 逐文件验证 + 与父NEXT逐数组exact 后补 ready/index；若无完整 manifest 则按上一条 |
| index 指向不存在/冲突根或分叉 lineage | fail closed；不可根据最大目录编号猜测权威 |
| 中途 optical 故障 | 原 deterministic optical replay；`has_authoritative_post` 保留并校验已提交 POST，不能默认丢弃；新诊断输出独立 attempt |

初始化三场和 PRE0_READY 同样遵守上述规则。create 自身会先建目录、最后写 manifest；ready 之前整个根为私有候选。
所有 ready/terminal/link receipt 原子写并 fsync，绑定 schema/runtime/config/LUT/source/schedule 与每个 screen 的身份。
链接凭证不是第二种可写慢介质；只有 Streaming manifest/pointer 决定物理状态。

## 5. 末发闭合（NEW_GLUE_DESIGNED_NOT_IMPLEMENTED）

N=3：3次 optical、3次逻辑全域POST完成、2次 hydro/promotion，最终 POST_final。
末发仍逐 screen 原 commit_post；独立末发 hook **不 enqueue**，不启动 hydro，不调用既有 NEXT barrier/finalize/promotion，
不伪造 NEXT。新增 `validate_final_post`（拟名）在相同锁/读取校验封装下确认 K 个 POST_COMMITTED，
逐record三场hash/finite/坐标/HR3 authority正确，无 queue/backlog/claim/NEXT/pointer/临时孤儿，光学与九条账本完整、writer退出。
该只读检查生成 POST_final receipt，引用现有 durable POST artifacts；不改记录状态，不重写 Streaming authority。
恢复时有有效终态 receipt 则验证后返回；无 receipt 则复核/补写；不完整则恢复 deterministic replay，仅补未提交POST。
末发恢复不调用会把 POST 重入 hydro queue 的 reconstruct_queue：需要末发专用 bootstrap 封装，复用静止性、POST校验和 replay，
不改变非末发恢复。该区别必须单独定向测试，不能宣称 S5 已覆盖。
旧writer静止性来源还包括 `Filament_python/tools/monitor_hr4e5s_s5_final.py:69–75,170–184`：
squeue/sacct终态加worker identity/live actor证据；本轮不运行该monitor。未来有序跨allocation退出复用该检查逻辑，
不调用其中注入故障或提交恢复作业的分支。

## 6. 资格继承与最小增量

| 项 | 分类 | 继承范围 / 新增验收 |
|---|---|---|
| per-screen POST/NEXT、queue/backlog、claim | REUSED_UNCHANGED | 固定源码、S3/S4R；仍有 block8/queue16，不能扩大queue躲恢复问题 |
| 串行 bootstrap、durable POST replay、旧writer静止性 | REUSED_UNCHANGED | S5单发资格范围；S5-FINAL终态仍待证；跨发epoch绑定另测 |
| barrier/promotion及pointer中断补齐 | REUSED_UNCHANGED | 非末发；既有guard不变 |
| Streaming实际PRE读取与完整prefix hook | NEW_CROSS_PULSE_GLUE | fresh-source/不读NEXT/字段速度继承/全域覆盖 |
| 两次新根换代、ready/index恢复 | NEW_CROSS_PULSE_GLUE | 2次rollover、缺失/重复0、幂等恢复窗口 |
| 末发及末发bootstrap | NEW_CROSS_PULSE_GLUE | 3 optical/2 hydro、无NEXT、POST_final恢复 |
| 改现有科学算子、既有barrier或丢弃durable POST | BEHAVIOR_CHANGE_REQUIRES_REVIEW | 本任务未推荐、未授权；若实现发现需要则停止 |

完整输入、exact、资源与后续任务分别见同目录 freeze_candidate JSON、resource_budget CSV、minimal_glue 草稿及 E5-1 candidate。
