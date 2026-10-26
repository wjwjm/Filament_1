# CPU-S4-3G3-1 小网格人工审阅报告

SMALLGRID_INCREMENTAL_GAIN_PASS / MANUAL_REVIEW_PENDING

固定 [Nt,Ny,Nx]=[384,117,129]，complex128、C layout，入口连续八区间。仅删除受控私有 E_stage1 内容封签与内部复制比较，改用 controlled_lifetime_v1；保留三次真实 Omega、公开 strict、两级 fresh Raman、FFT、Heun/Strang、沉积、Qacc/POST 与正常输出。没有改生产科学代码或 G2 冻结源。

缺失的 382-byte git 原件按原哈希取回；414 原部署成员和必要 overlay/support 共 480 成员恢复。开发 HEAD e40a3ea、部署 HEAD 66801ad、科学子树 821fec5 与各 factory/候选来源分别记录。新部署是精确外部 materialization，不冒称干净 Git checkout。

## 科学与覆盖

I0/G2/G3 各一次八区间资格及九个正式窗口全部完成。G3 为一次资格、三次正式、三次一完整区间 warmup；本机负向测试的严格回退不计为 G3 执行资格。G2/G3 每八区间实际 nH=16、32 fresh 响应、128 dispatch，真实 worker IDs 0–47、mask 24→48→24；G2 E_stage1=16，G3 E_stage1/E_apply=0，双方 Omega=48，fallback/rejected=0。

首边界112项沿原108 exact＋4 timing规则；末端全元素、原生布局/alias，G2/G3 首 pre/post stage 与 I_R/RHS exact。没有扩展 NaN、timing、storage 例外。输入直接在小尺寸生成并固定 ambient PRE，后续区间使用前态；真实响应与 RHS 非零，源/PRE保持只读。详见 EXACT_INDEX、WORKLOAD_AND_ROUTE_AUDIT 和逐字段原始记录。

Windows/Linux 各30项最终受控链测试通过，再各2项最终入口检查通过。微型2/8区间、Nt384微空间和主形状首边界已检查。Linux为实际独立解释器/缓存，使用同一 hash-bound 源文件；本地实际单线程，不宣称本地 W48。R12 metadata erratum：早期 Linux execution receipt 的 local_pool=4 是标签错误，实际环境和 factory mask=1；原记录保留，已修正后续桥接标签。没有观察到科学 nonexact。失败工程 fixture 和修复前像保留在 REPAIR_HISTORY。

生命周期合同仅针对冻结的合作式受控调用链；本执行每个子进程一个 role factory，不宣称任意同 token 多 factory 共存或任意 native/反射并发篡改安全。

## 三批正式秒数与配对收益

|批次|I0 application|G2 application|G3 application|相对G2增量|相对I0累计|
|---|---:|---:|---:|---:|---:|
|B1|51.388772|48.283825|45.023621|6.7522%|12.3863%|
|B2|51.849633|48.020605|44.369318|7.6036%|14.4269%|
|B3|52.025624|47.965240|45.150501|5.8683%|13.2149%|

应用增量中位 6.7522%，正向 3/3；累计中位 13.2149%。完整 compute/application 原秒数、min/max、配对结果、首次 runtime 初始化和服务成本见三个 CSV/JSON。小样本仅作描述，不声称显著性；不与历史 G2 6.75% 相加。

## 资源与收束

Job 8851211，cc1102，96物理CPU、0GPU、32GiB、单节点，终态 COMPLETED / 0:0，实际 921 秒，allocated 24.560000 core-hours；无本任务在途，不重复计算 batch/extern。RSS采样峰值 8.8230 GiB；kernel cgroup high-water 9.2295 GiB（含 charged cache），低于停止线。

没有运行完整 [384,351,301] 场、真实全轨迹或 Hydro，没有证明真实光学吞吐与介质匹配。FULLSIZE_NEW_CANDIDATE=NOT_RUN，PRODUCTION_QUALIFIED=false，DEFAULT=I0。生产默认及旧 G2 资格不变；未 commit/push/merge。

人工审阅本包后，再决定是否单独授权大尺寸确认。

## 复核与保留

解压后运行 `python -I -B replay_readonly.py`：仅标准库、只读、不联网，核验包内原字节、从已保存全量比较 metrics 重放原 numeric gate、检查角色计数、配对收益和顶层 allocation 费用。它不重新执行数组比较或传播。

一份小 oracle、其112字段首边界和必要 G2/G3 fresh-stage 资格数组保留。正式 raw arrays 和正常输出留在原远端，逐件原字节 hash/size 与完整比较日志保留。PORTABLE_PATH_MAP/REVIEW_INDEX 使用相对链接；原收据绝对路径不改写。未清理任何历史目录或本轮 raw arrays。
