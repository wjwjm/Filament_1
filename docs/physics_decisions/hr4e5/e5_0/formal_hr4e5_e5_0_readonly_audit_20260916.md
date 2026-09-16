# Formal E5-0 定向审计纠偏与准备收口

**状态：** `E5_0_STREAMING_CLOSEOUT_READY_FOR_WEB_REVIEW`。仅准备合同完成，尚非 CLOSED / implementation PASS / execution PASS。

`architecture_choice=USER_FIXED_EXISTING_STREAMING`；`intra_pulse_optical_hydro_overlap=REQUIRED`；
`production_hr4c_replacement=false`；`implementation_authorized=false`；`formal_execution_authorized=false`；
`resource_execution_gate=NOT_RELEASED`。

源码审阅及 start HEAD：`cbfe2e38172b5221d739df80b032e4dee9fab7e3`；S5 execution SHA（历史回执）：`cd456ff8413cbc041d2d60b9b64007a1554028a1`。
两者之间只有文档/旧包差异，本轮未实时查询 S5，244700 的 PENDING 仅为原 HPC 快照。
文档提交 SHA 在提交后由新包 INDEX/README 绑定；不得把文档 SHA 当执行 SHA。

## 1. 审计继承与纠偏

原审计的真实入口缺口、S3全optical/48采样区别、参数、source与site缺失继续保留。
撤回其 HR4C 主导生产与全域POST后才hydro的解释。当前唯一有效设计见
[authority/interface](formal_hr4e5_e5_0_authority_interface_design_20260916.md)。旧版由Git及原ZIP追溯。
只读源码确证 S3 optical读原NPY、selected hook只捕获48点；跨发必须增加Streaming PRE view，不能只外套循环。
本轮未读取/改写远程运行目录，S5状态沿用历史快照且不自动关闭。

## 2. 输入、末发与域

推荐完整输入面连续前缀0..8047、N=3，保持原schedule步长逐值，覆盖E1B peak8022，
同域 exact 不宣称full-z等价。PRE0为E1B delta_n前缀加两个零速度场；新前缀/文件哈希留待preflight。
三光学/三POST/两hydro，末发新POST收尾与恢复封装尚未实现，不能套用NEXT barrier。

## 5. Frozen parameter source matrix

The table identifies values available for direct inheritance from the frozen
source.  `INHERITED_FROZEN` means a future approved contract can bind the
named value and hash without asking the user to re-tune it.  It does not grant
execution authority.  `QUALIFICATION_ONLY_NOT_FORMAL` is intentionally not
promoted to a formal campaign value.

| Parameter | Effective available value | Source / key | Status |
| --- | --- | --- | --- |
| wavelength | `8.0e-7 m` | `real_post_input_config.json`, `beam.lam0` | INHERITED_FROZEN |
| pulse duration / definition | `1.2e-13 s` FWHM | config `beam.tau_fwhm` | INHERITED_FROZEN |
| input amplitude definition | `P0_peak=1.7e10 W`, `E0_peak=0`, `energy_J=null` | config `beam` | INHERITED_FROZEN |
| beam / focus | flat-top-cosine, `w0=radius=1.979e-3 m`, edge fraction `0.9`, `f=0.95 m` | config `beam` | INHERITED_FROZEN |
| optical grid | `Nx=301`, `Ny=351`, `Nt=384`; `Lx=3.01e-3 m`, `Ly=3.51e-3 m`, `Twin=9.6e-13 s` | config `grid` | INHERITED_FROZEN |
| optical z schedule | `dz=1e-4 m`, `dz_focus=5e-5 m`, focus stepping on, `z_max=1.3 m` | config `propagation` | INHERITED_FROZEN |
| numerical precision | requested `fp64`; slow state float64 | post-reference and S3 manifests | INHERITED_FROZEN |
| gas / linear constants | `n0=1.00027`, `n2_air=7.8e-24` | config `beam` | INHERITED_FROZEN |
| ionization | full-time RK4, `tdiff`, N2/O2 `ppt_talebpour_i_lut`, rate table enabled | config `ionization` | INHERITED_FROZEN |
| Raman | enabled; `isaacs_rot_sinexp`, full Isaacs Eq.27, IIR / Heun | config `raman` and `propagation` | INHERITED_FROZEN |
| source field construction | runner input field plus thin lens; fresh `E_source.copy()` exists in runner | `runner.py`; S3 calls `build_transverse_input_field` then lens | INHERITED_FROZEN |
| HR-2 / HR-3 selection | HR-3B true; HR-3C false; selected S3 uses authoritative HR-3A/B POST | config `heat`; `hr4e5s_s3.py` | INHERITED_FROZEN |
| hydro geometry | `dx=dy=1e-5 m`, collocated nodal source | S3 input manifest / post reference | INHERITED_FROZEN |
| hydro material constants | `chi=2.17e-5 m2/s`, `nu=1.5e-5 m2/s`, `n0=1.00027`, gravity `(0,-9.81)` | S3 input manifest | INHERITED_FROZEN |
| hydro integration | `dt_hydro=1e-6 s`, 1,000 steps, CFL `1.0` | S3 input manifest | INHERITED_FROZEN |
| queue / block values | block `8`, queue depth `16` | S3 input manifest | QUALIFICATION_ONLY_NOT_FORMAL |
| repetition rate | `f_rep=1000 Hz` | config `heat.f_rep` | INHERITED_FROZEN |
| exact interpulse construction | `build_interpulse_step_schedule(f_rep, dt_hydro)` | `hr4d_pulse_lifecycle.py` | INHERITED_FROZEN |
| source state / PRE identity | E1B HR-3B source hashes named in §2 | prepared input manifest | INHERITED_FROZEN |
| `Npulses=1` | qualification-only single pulse | config `run.Npulses`; S3 guard | QUALIFICATION_ONLY_NOT_FORMAL |
| full-z `K=15000` state shape | historical source state shape `[15000,351,301]` | post-reference manifest | QUALIFICATION_ONLY_NOT_FORMAL |

上述原始来源继续成立，但本次工程候选已经明确 N=3/K=8048/prefix；full source 15000不是canary K。
queue16/block8作为本次明确推荐沿用；科学E5-2终点未选，不影响E5-0完成。

## 6. Streaming资源与保留总账（本次设计模型）

G=3*K*351*301*8=20,406,701,952 B (19.005222 GiB)；F=G/3；O=351*301*384*16=649,119,744 B (0.604540 GiB)。
candidate每发独立根且复制CURRENT：两个非末发各3G、末发2G，共8G。
reference三场PRE/POST/NEXT快照8G，非末发Batch双slot工作store共4G（额外保留）；
两轨六种诊断×三发共12G；final optical共6O。加现有full source F_full，科学payload合计32G+6O+F_full=669,587,300,928 B (623.601769 GiB)。
没有把HR4C的2G写成生产稳态，也没有使用3G..5G作为campaign界限。
推荐无删除/无共享，先reference再candidate；顺序执行减少同时运行资源，但两套证据同时占盘。

| 对象/事件 | 生命周期与同时存在关系 |
|---|---|
| candidate p0初始化 | host三数组G + mmap source最多F_full驻留；create逐screen写CURRENT G；partial root不可被发现为ready |
| p0/p1 optical+hydro | CURRENT、逐渐增加POST/NEXT，完成3G；临时NPZ最多(1+4)×3A；manifest atomic旧+新同时存在 |
| p0→p1初始化 | p0完整3G保留，p1CURRENT增加G；共4G。p1完成后6G；p2CURRENT后7G；finalPOST后8G |
| 前代→后代 | 明确复制，不共享；create输入为完整三RAM数组G，读取前代NEXT按screen填充；绝不算成零成本 |
| reference | 8G exact快照+4G Batch工作slot；最后一发没有Batch hydro store/NEXT |
| 六诊断 | qthermal/qion/qib/qraman/increment/state_after每发各F；全部保留，含冗余state_after |
| checkpoint/history | C=0额外副本；既有保留根就是一次allocation延续的证据。每新增完整副本另加G |

CSV各数值从bytes生成。旧 block8 的 `(6*8+12)*351*301*8` 与 optical双场换算已修正，
HR4C工作集行仅为参考；checkpoints_and_history有完整列，不再缺列。
具体更正：50,712,480 bytes / 2^30 = 0.047229677 GiB（旧CSV为0.047236328125）；
1,298,239,488 bytes / 2^30 = 1.209079742 GiB（旧CSV为1.2090814113616943）。
原正文block工作集约48.363 MiB本身并未错，错的是CSV的GiB换算。
CSV的meta/header/log预留=3,164,602,368 B (2.947266 GiB)，NPZ并发临时=12,678,120 B (0.011807 GiB)，LUT/scalar等另预留8GiB；
合计再留25%余量，提交前同时要求可用空间和项目配额≥851,693,145,010 B (793.201053 GiB)。
若已有full source在另一个filesystem，应逐挂载拆账；不能把同一共享空间的旧文件从总账隐去。
本预算没有失败attempt副本；出现额外attempt或诊断重放，准确加其残留bytes并重新过25%门，不自动删旧目录。

### CPU、GPU与I/O

CURRENT输入独立三host数组G；原S3 advanced-index selected+sharedzero=2F（vx/vy引用同一zero），
不是create没有整卷开销。原source mmap虚拟F_full、实际驻留0..F_full；reference memmap两slot虚拟2G，
页缓存不是另一份必然常驻分配，也不是免费。hash按screen进行，避免整卷tobytes瞬时再复制F。
每worker block8三场数据=24A，算子工作集/host-device传输/诊断另计；不能以block估计冒充RSS。
规划 optical/coordinator 64GiB RAM、四worker各16GiB，共128GiB；以每进程80%实测峰值为运行门，
优先1+4、每task8CPU，总40CPU。调度器内存分配须确保optical独占64GiB，不能机械平均。
source+working GPU下界=1,298,239,488 B (1.209080 GiB)；可追踪stage/非线性/FFT数组按12O=7,789,436,928 B (7.254478 GiB)作条件工作额，
不是严格峰值。FFT计划、CuPy pool、LUT、library workspace未知。建议optical GPU≥16GiB、hydro每卡≥8GiB且
实测used/reserved均≤80%容量；若不满足即不释放资源门，不能保证不OOM。源码内部precision默认保持不变。

科学写入至少上述payload（现有source不重写），读取还包括初始化、每screen hash、barrier扫描、replay和逐数组exact；
可按 `write + 2*comparison_payload + source/binding_reads` 建初始流量账，额外扫描按实际次数追加。
**manifest整体重写是独立I/O成本**：源码每次_save重写records和事件；K放大后不能当作恒定小metadata。
令一根最终manifest M、每screen保存次数s，则写入保守模型≤s*K*M（附加轮询/恢复事件另算），
推荐规划s=20、M=16384*K bytes，则candidate三根条件写量=63,671,799,644,160 B (59298.984375 GiB)，
这是累计I/O，绝非磁盘驻留。metadata峰值计旧/新atomic双份，已含在64KiB/screen保留额中。
未来小型纯metadata入口检查需测M、s、write_seconds并冻结；若超预算或filesystem吞吐不足，资源门保持关闭，
不在本任务重写manifest协议或扩大queue。运行telemetry每次phase/每pulse汇总RSS、VRAM、I/O字节与写耗时。

### 时限模型（条件预测，不是full-z实测）

历史S4R optical active为10714.485..13425.556s；这是完整1.3m光学轨迹，**不乘K/48**。
本prefix每发先用同一10.7..13.4ks光学包络作保守规划占位，新PRE可能改变负载，非上界证明。
1+4 hydro固定48点速率0.575570 screens/s、1+2为0.320888；按同screen工作量和吞吐不变的限定假设，
本prefix hydro h4=K/0.575570=13982.7s，h2=25080.4s。
候选非末发compute位于max(o,h)..o+h；三发 `2*nonfinal+o` 为约10.7..19.0h（不含I/O、排队、恢复）。
serial Batch仅粗按最保守单worker有效速率0.16screen/s（由1+2总速率除2再下取整）建模，
`3*o+2*K/0.16`约36.9..39.1h；这是假设，不冒充单worker实测。
另加初始化/比较/metadata I/O时间 `B_read/R_read+B_write/R_write+T_metadata`，再留25%时间余量。
一次candidate allocation在p0后结束，下一次跑p1/p2；reference可在pulse边界正常续跑以适配已确认QoS，
代表性恢复验收只针对candidate一次。若site有效时限不能容纳公式请求且无已审checkpoint，禁止提交。
不把S5的12h请求或partition UNLIMITED当site上限；不承诺48点耗时能预测新状态实际耗时。
本次无queue/运行/等待测量；本地工作仅文档工程与小元数据检查，未产生生产修复/重跑时间。

### 保留、比较、未来回收与E5-2公式

保留顺序：reference所有pulse输出→candidate每pulse输出及comparison→全部exact/provenance签收→人工审查。
既有根CURRENT、POST、NEXT、Batch双slot、六诊断、final field、ledger、ready/pointer/bootstrap/log都留到签收。
candidate下一CURRENT复制完成也不解除父代lineage引用；不能只凭下一发开始就删父NEXT。
若未来单独授权回收，精确列 `reference/pulse_p/batch_store/*`（比较通过且无恢复/后代引用），
以及已停止且未被任何ready引用的 `candidate/pulse_p/attempt_a/`；须全writer静止、exact签收、依赖扫描、
parent ready/pointer及对照证据外存后才可以按清单执行。常规root不自动回收，无通用GC/压缩/存储重构。
单Streaming无删除保留 ` (5N-1+C)G + N*O + F_full + allowances`；
双轨本方案 ` (12N-4+C)G + 2N*O + F_full + allowances`，C是额外完整三场副本总数。
E5-2的N和C需科学/资源另审，末发仍无NEXT。

## 7. 历史资源证据与外部门

原 [HPC快照](../../hr4e5_hpc_audit_pack_20260916.md)保持原文不改：S3 batch03:06:26，
S4R1+4 02:59:28、1+2 03:44:29，1+6 NOT TESTED/RESOURCE_UNAVAILABLE；S5 clean03:03:53。
本轮有限补查仅本地已有config/source/LUT manifest与该快照，未实时登录HPC；日期2026-09-16，方法/范围记录于包内。
原快照 quota命令缺失、nvidia-smi登录机缺失、sacct MaxRSS/IO空、QoS数字未显；均不能用requested替代observed。
约180GiB历史可用空间低于本方案建议值，且不是配额。PENDING_EXTERNAL_EVIDENCE：

| 类别 | 已明确关闭方式 |
|---|---|
| 设计阻塞 | 当前无；prefix合法性已做metadata检查、接口缺口已有明确glue合同；大数组内容仍需preflight |
| 提交前site | df与管理员quota分别≥CSV容量门；purge覆盖保留期限；effective QoS/TRES/时限可承载公式；GPU/主机容量分配满足请求 |
| S5证据 | live squeue+sacct+controller+persisted exact/provenance及F01–F06跨SHA继承；不能用户投票豁免 |
| 启动/执行验证 | 固定Filament_python interpreter/imports、device mapping/model/driver/CUDA、每task RSS/VRAM/I/O；80%内存、25%磁盘余量；用prologue和每pulse receipts留存 |

## 8. 最少待批准集合

人工接受派生prefix/PRE0及保留合同；单独授权最小glue实施/定向测试；确认实际资源预算并另行授权HPC提交。
Streaming路线已由用户固定，不再提请选型；S5 PASS是证据门。准备就绪不等于E5-0 CLOSED，更不等于作业许可。

## 9. 本轮小型输入回执实核与历史区别

配置raw `eaec83ad...` 与source manifest raw `3e2c7557...` 对上原审计。
本地 prepared input raw为 `b18ddd46410e329fe61781a2349062526837f0c6a1fcedcea16dbd4aa99bdcfe`，
LUT workspace raw为 `e290579027983c0f13da391f5677cfbe356f3f458c6f779d8e1e84b9616c8ed3`；
它们不是原HPC快照引用的 `280d8d1f...` / `5ef79cc1...` 回执。包内完整保留实际本地小文件，
INDEX source_commit=null、raw_bytes范围并注明较早资格快照；仅继承其配置、hydro参数与LUT候选身份，
不把它们升级成S5-FINAL当前输入或PASS。两套值在freeze JSON分别命名，原HPC快照不改。
