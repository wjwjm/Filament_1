# CPU-S4-6A 人工审阅报告

**终态：WORKLOAD_MISMATCH_IDENTIFIED / NO_MATCHED_SPEEDUP_CLAIM。** 共同输入和工具已建立，但 CPU 小网格在允许的一次技术修复后仍被汇总入口阻断，未释放 GPU 和原尺寸阶段。DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。

## 先回答五个问题

1. **同工况完整应用、完整窗口和第2～8区间分别差多少？尚不能回答。** 两端三次正式 OFF 样本均未形成，三个 CPU/GPU 比值均为 NOT_COMPARABLE。原尺寸没有新测量，不能解释历史330秒窗口的全部差距。CPU仅完成辅助 REFERENCE 和一个 QUAL_OFF；后者应用 84.319451 s，不作为 P1–P3 代表值。
2. **前三大绝对差距是什么？当前无法排序。** 缺少真实 GPU 和有效三对数据，GAP_CONTRIBUTIONS.csv 的各范围保留未测/未归因；不会用旧 PROFILE 秒数或局部倍率填入前三名。
3. **CPU低并行、GPU活动及传输各占多少？只有一个 CPU 资格窗的有限观察。** 其应用 process CPU 176.541123 s、CPU占用当量 2.093718；首完整区间 14.060010 s、后续2～8区间 37.131707 s，来自原整数ns。当前 OFF 未保留整窗 PROPAGATION shell，不能补造窗口值。节点为 Xeon Platinum9242、96物理核/4NUMA；真实 G3 路径通过 nH16、fresh32、128数学 dispatch 和48 worker记录。占用当量不等于固定核心数、可并行比例或96/n加速上限。GPU未分配，没有设备活动、传输、显存峰值证据。
4. **哪些因素已有证据？** 历史“约27倍”的时间窗口不一致已复原；工具入口还暴露可选细分 profiler、CPU专用依赖和汇总 schema 问题。没有证据把余量归因于带宽、GIL、NUMA或硬件饱和。同步/启动可能改变测量，但本轮没有发布它们贡献的跨后端秒数。
5. **下一项CPU优化是什么？目前没有安全的数值候选。** 唯一后续合同是补齐合法匹配窗口与时钟证据；不实施CPU数学优化，不重开Hydro/MPI。详见 NEXT_CPU_OPTIMIZATION_CONTRACT.md。

## 历史口径与共同工作量

原 GPU CSV 的48个 POST（screen7998–8045）跨度63.47607327159494 s，原公式48/跨度得0.7561904435175519 screen/s；同一次 optical_active 为10890.363696183078 s，选中窗前5778.935226830654 s、后5047.952396080829 s，窗内还有10.787464395165443 s背压。完整长前缀与选定POST跨度覆盖不同范围，并不矛盾。CPU历史约36.11秒/POST发生在第二线性半步前，也不是完整区间末端。相乘得到的约27.3不能作为本轮同工况或纯硬件倍数。

两种尺寸均已在CPU端只封存一次 E/PRE、坐标/频率和LUT原字节；全尺寸复用已接受L1原输入。complex128、入口8区间0至0.0008 m、原BK-NEE/Raman full Isaacs Eq27/IIR/Heun及科学诊断保持。GPU工具保留原CuPy核心；CPU保留已接受G3/L1缓存，未合入5C/5D。新比较不要求跨后端bitwise，也没有新增epsilon。

## 实际执行、故障与修复

- 仪器v1沿用可选BK细分 profiler，GPU会逐表达式同步，不满足OFF要求。仅本任务 job8854173 被取消，保留全部旧source/plan及输入；无正式性能样本可用。
- v2关闭该细分 profiler，保留原宏观标量钟和科学诊断。GPU缺Numba/psutil，工具已解除CPU factory依赖，未改共享环境。两次显式内存请求被站点策略拒绝后停止数字试探；当前默认1GPU/8CPU、126000MiB（123.046875GiB）预检通过，但未提交GPU作业。
- CPU技术修复 job8854203 的 QUAL_OFF 完成8区间；终态、Raman stages和112字段首边界均按原规则 exact，原布局/alias/寿命和4计时/NaN例外保留。随后父汇总器把真实 raw_context 当作 context 读取而失败；复核也发现 OFF 的整窗scope未启用。故障是测量工程失败，既不是数值失败，也不能升级为完整矩阵成功。
- 本地提交6577000已修正这两项，并把同一提取器纳入真实2/8区间微测。**该修复未在HPC执行**；本地测试状态另见 LOCAL_COLLECTOR_ACCEPTANCE.json。科学子树仍为821fec5bdacd832b0347e1f2e84285cb54172e57。

## 资源与停止门

|主JobID|终态|分配秒|CPU核时|
|---|---|---:|---:|
|8854173|CANCELLED|175|4.666667|
|8854203|FAILED 1:0|319|8.506667|
|合计|||13.173333|

两次排队各1秒；batch/extern不重复收费。GPU为0 JobID、0 GPU-hours、0附属CPU核时；本任务在途allocation为0。剩余core-hours和第三JobID空位不放宽“最多一次CPU技术修复”及B阶段前门，因此没有第三次small、GPU或原尺寸提交。主动本地工程、监测等待和修复工时未独立计量，记UNMEASURED；原收据保留。

## 证据边界与交付

轻量导出437份元数据，下载后raw SHA逐项一致；大数组留原端，未声称本地重读或重新核验原科学数组。审阅包内保留原调度终态、原比较JSONL、输入/源码清单、失败版本、时钟和离线replayer。replayer应复现“数据不完整、没有匹配比值”，不能输出科学PASS。人工审阅后才决定是否另行授权补证。
