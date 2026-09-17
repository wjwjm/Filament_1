# E5-1 连续前缀三发工程验证候选

**2026-09-17 本地实施交付：** `E5_1A_PARTIAL / BLOCKED_BY_FORMAL_DRIVER_AND_RECOVERY_QUALIFICATION`。已完成代码和小型验证见 [实施报告](E5_1A_IMPLEMENTATION_REPORT_20260916.md)；正式总控的输入/预算/恢复凭证绑定仍有代码缺口，A 未验收、B/C 未释放。

**E5-0 状态：** `CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED`（2026-09-16 人工审核接受）。2026-09-16 新任务已授权 E5-1A 隔离本地实施与 CPU fixture 验证；B/C 科学执行仍未授权。

`architecture_choice=USER_FIXED_EXISTING_STREAMING`；`intra_pulse_optical_hydro_overlap=REQUIRED`；
`production_hr4c_replacement=false`；`e5_1a_local_implementation_authorized=true`；`formal_execution_authorized=false`；
`resource_execution_gate=NOT_RELEASED`；S5-FINAL 是未来科学执行门，不阻止已授权的本地 A。

原准备合同的源码审阅及 start HEAD：`cbfe2e38172b5221d739df80b032e4dee9fab7e3`；S5 execution SHA（历史回执）：`cd456ff8413cbc041d2d60b9b64007a1554028a1`。
两者之间只有文档/旧包差异，本轮未实时查询 S5，244700 的 PENDING 仅为原 HPC 快照。
原审核包绑定文档 SHA `c377178e909d795b1fce0d4d162ab14845c9c264`；本次仅同步人工接受状态，不重打包、不改包内历史状态，不得把文档 SHA 当执行 SHA。

正式收尾结论与 G1–G6 门见 [E5-0 closeout](../e5_0/E5_0_CLOSEOUT_20260916.md)。

## 推荐域与合法性（设计，不是运行结果）

`E5_1_CONTIGUOUS_PREFIX_N3_CANDIDATE`：N=3，K=8048，source indices 0..8047，
z=0 至 0.8047999999999277 m（展示约0.8048 m；精确hex `0x1.9c0ebedfa4173p-1`），
每场 `[8048,351,301]` float64。原完整域 K=15000、z_max=1.3 m。
同一冻结输入面构场/薄透镜，不用内部峰处冻结光场；完整覆盖每个实际 optical interval。
原 peak index8022、z=0.802249999999928 m、min(delta_n)=-0.00014098281502767653；
front index7827、min=-2.843616836977868e-05，均由 E1B post_reference_manifest 的 screens 记录提供。
因此 prefix 有已知非零/非平凡慢介质，并覆盖旧48窗口末端8045。未来启动仍验证沉积有效且finite，
不预言新三发沉积幅度；该域只用于同域编排 exact，不能宣称 full-z 科学等价或选定 E5-2 终点。

域终点按 `ceil((8045+1)/8)*8=8048` 选择，共1006个完整block；原48窗口终点仍为8045。
`StreamingLifecycle.claim_block`（streaming.py:1028–1034）在正常非恢复模式拒绝不足8个screen的尾块，
故不能选择K=8046。这里多保留两个原interval，既不补假screen，也不改变block/claim语义。

原48点不能单独组成传播域。前缀优于 full-z 作为本次推荐，因为光学是向前传播且已有 schedule 接口可接收
原始连续区间，无需改上游输入、算子、精度或步长。full-z 仍是可另审保守候选，不是唯一数学合法域。
元数据检查使用现有 LongitudinalSchedule，仅生成小型元数据：原schedule切片 validate通过。
**禁止重新 build 到缩短终点**：其最后 dz 变成9.999999999998899e-05，原值0.0001；
必须直接传 full.z_edges[:8049]、dz_intervals[:8048]、intervals[:8048] 构造的 schedule，
propagate 接受 supplied schedule 并直接遍历 interval.dz。effective z_max 与 schedule span一致。
仅候选说明覆盖 run.Npulses=3、effective z_max、全prefix采样；生产配置原件不改。

## 输入绑定

原配置路径、完整显式参数、LUT签名和hash见 freeze_candidate/evidence；隐式默认 pin 到源码 SHA，
包括 rho0=1.23 kg/m³、Cv=1000/1.4 J/(kg K)、air_T=293.15K、linear_precision_strategy=baseline_complex64。
fp64 输入不意味着把该内部策略改成全complex128。
PRE0.delta_n 是 E1B 原数组的0..8047前缀；原文件raw SHA `70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467`，
原整数组canonical SHA `5990da24bec80937bf3be9985777b3797be9adffedb776dd2f2e840b118d8ad9`。这两个值不是派生前缀的哈希。
vx/vy 是正float64零；每screen按既有 sha256_array 口径验证零内容，未来记录持久化文件raw hash。
不生成巨大数组来填空。PRE0为可追溯工程初态，不代表某自然脉冲列的完整三场历史。
optical x=-1.5..1.5mm、y=-1.75..1.75mm，dx=dy=10µm；E1B hydro标签平移y+0.75mm是相同索引重标，
不是把光场/数组搬移0.75mm。全z身份按原interval index/midpoint/edge绑定，p=0，source_generation=E1B。

## 串行对照与逐数组 exact

推荐先跑隔离 serial Batch reference，再跑 candidate，两条轨迹独立PRE演化、相同source/PRE0/schedule/config/LUT。
Batch复用 `run_batch_hydro` 的 HR4C/evolve_hr4_full_z 路径，仅作为对照；外层适配完整K和多发PRE速度，
不套用旧 S3 的48或每发速度清零。科学算子共用；独立性在编排、队列与存储路径，不是独立物理模型。
reference存PRE/POST/NEXT三场快照；非末发HR4C双slot工作目录额外保留；candidate保留新根CURRENT/POST/NEXT。
双方每发六个sink archives全部保留；state_after与POST delta冗余也不省略。末发没有NEXT。

比较沿既有 compare_exact 的四项AND：shape、dtype、sha256_array、np.array_equal；另要求finite。
sha256_array是 dtype+NUL+canonical JSON shape+NUL+C-order字节；NPZ文件raw hash只作传输/provenance。
新增比较外层参数化K/N/末发和PRE，而不把exact降为hash。

| 对象 | 数量 | 形状/口径 |
|---|---:|---|
| PRE/POST每发三场 + 非末发NEXT三场 | 193152 | 24K个 `[351,301]` float64 arrays |
| ion/ib/raman逐机制沉积 | 72432 | 9K个float64 arrays |
| qthermal/increment/state_after诊断 | 72432 | 9K个float64 arrays，完整捕获/比较 |
| screen科学数组总数 | 338016 | 42K，不套用432/432 |
| 九条scalar ledger × N | 27 | 每条 `[K]`，字段集合恰等于 S3 _ledger_payload 九名 |
| final optical | 3 | `[Nt,Ny,Nx]` complex128，按实际输出shape/dtype门检查 |
| fresh source检查 | 3 | 原始source内容不变、工作场与source不alias；每次调用前记录 |
| 各轨迹 NEXT→下一PRE绑定 | 48288 每轨 | 独立于跨轨比较，3字段×2换代×K |

现行 A 顺序为 R1 → C1 → Exact1 → 顺序后继交接/回收，再继续下一对；每对回收前完成实际数组 exact。
失败记录 trajectory/pulse/namespace/source_index/field、首个不同坐标和两份文件；原科学数组继续留存。
调度provenance单独核验epoch、claims、bootstrap、barrier/pointer、无漏screen/重复提交、两个rollover、末发计数；
不要求两轨jobid/timestamp/queue顺序相同。全体exact零mismatch，任何缺失/NaN/shape/dtype不符即FAIL。

## E5-1A 当前实施与恢复边界

本次新授权替代原全历史保留及跨 allocation 计划：第一对 exact、双方自包含后继 READY 与安全回收后，退出科学/state-holder 进程，全新本地进程只凭持久文件继续第二、三对。验收名称为 `NEW_PROCESS_PERSISTENT_RESUME`，实际跨 allocation 验证留后续授权阶段。

原 E5-0 收尾及 ZIP 保留历史快照。本次只新增外层 Streaming 编排、配对 exact、容量保护和自产中间文件回收；不改科学算子，不执行 S5/Slurm/GPU。Candidate 仍须保留发内 optical–hydro 重叠，末发仅 POST。

## 现行预算与门

单 campaign 硬上限为 **322122547200 bytes（300 GiB）**，最终输出默认 **68719476736 bytes（64 GiB）** 且包含在总额内。原 793 GiB 全保留估计属于已被本次授权替代的历史策略，不是当前限额。

逐发配对、实际 exact 后顺序建立后继与回收；保留 candidate 最终完整 CURRENT+POST 根、两轨逐发光学输出及完整证据。只允许回收本任务自产、exact 完整、恢复点 READY、writer 静止且无未来依赖的白名单文件。详细合同见 [E5-1A storage/retention contract](E5_1A_STORAGE_RETENTION_CONTRACT.md)。

本地实现与测试状态由本次实施报告记录；小型 CPU 资格不释放正式运行门。B 仍须核对 S5 终态、代码兼容性、输入物化哈希、现场 quota/空间/QoS/显存与单独执行授权。