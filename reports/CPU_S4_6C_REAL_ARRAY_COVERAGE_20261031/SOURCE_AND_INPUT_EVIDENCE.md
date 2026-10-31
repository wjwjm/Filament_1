# S4-6C 来源、工作范围与计时解释

执行 CPU 科学基线 ed8754041625549a80959db7a8abd804d5d219ae；KHz_filament 子树 821fec5bdacd832b0347e1f2e84285cb54172e57。A2 工具执行提交 420ed7e82f09180da9a1b546f4ac69daeb795f59。后续报告提交不替代这两个身份；生产源未改。

## 实际链与真实输入

runner.fresh_factory 调用已有 cpu_s4_4a_linear_cache.candidate.build_role(PURE_OFF_SMALLGRID)，取得私有 L1 的 r8_state.stage 与原 raman_namespace['xp'].fft。由原 R8 public stage 重建四份响应；不向 retired G3 capability 注入保存的 stage。原 stage 的完整输出逐字段与 CPU 8855018 QUAL_OFF 捕获比较。零 reference fallback、输入身份与布局是实际运行门。

输入为同一 [384,351,301] complex128 原尺寸入口的首 pre/post field、predictor，分别为 STAGES/ARRAYS.zip 的 0000、0005、0010、0015.npy。原 E 为 C contiguous；响应原生 view 的 ownership/base depth 按旧捕获元数据恢复，不重排、不复制成新布局。原 archive 4,730,695,744 字节留在 t0，路径及逐数组指纹见 REAL_INPUT_INDEX.json；科学大数组不进入审阅包。四份输入只覆盖首区间两半步，不能代替八区间分布。

每份新 stage 产生的 I/response 为孤立 Eq10 重放的只读输入；这不构成传播路径的 Raman 响应缓存。C 相位/吸收没有合法对应输入，记 UNAVAILABLE。

## 两条完整局部链

| 链 | 冻结原函数位置 | 保留内容 | 允许变化 |
|---|---|---|---|
| A | raman.py isaacs_raman_stage:253 | 原 complex abs、square、标量 scale 与 float64 输出 | 各操作的独立输出分工；P_LOCAL 同序小块 scratch |
| B | 同函数:283–299 | 全局 abs/max、axis=0 原 SciPy FFT/IFFT W48、原 sum(axis=0)、能量 scale、正部裁剪 | 频谱乘法及 divide/divide/product 的独立输出分工；后者的小块 scratch |

参考函数体从冻结原 AST 提取，FFT 对象直接来自 L1 的实际工厂。没有采用未接入的 IR/R 物理函数。旧八区间的 A/B 调用数32、成本12.528/34.388秒仅作历史候选排序锚；本轮没有重新测量八区间。B 的 FFT 约8.610秒及原归约保留，不能把旧25.778秒非FFT全部称作可改成本。

P 每个逐点操作保留分离全场中间量和操作间 barrier；常驻 ThreadPoolExecutor 的1/8/24/48个工作线程只读共享输入，写入分区唯一。P_LOCAL 保留相同算术顺序和 ufunc，固定16384元素块、每worker两个float64 scratch；不使用 fastmath、乘倒数、复数abs改写或归约重结合。

## 时钟与可作出的因果判断

TIMINGS.jsonl 的 wall 是完整局部调用，包括检查、分配、ufunc、FFT/归约、线程提交、barrier/join和标量活动记录。输出指纹、psutil和NUMA快照在计时外；其缓存扰动属于所有配置共同的孤立实验环境。线程池初始化单列；每次调用仍分配自身scratch，不跨调用缓存物理数组。没有新增GPU测量。

REF→P1 包含表达式求值、分配/回收、显式 out 与线程管理差异，不能全部归因为并行。P1→P8/24/48 才隔离同组织下的线程数；P→P_LOCAL 只能用同线程 MATERIALIZATION 轮配对，不能用 THREAD 轮P作替代。

P_LOCAL 同时把 A 的3次任务/barrier变成1次、B的4次变成2次，并增加块内Python/ufunc调用。这是“减少全场物化的这项实现”的净效果，无法单独分离缓存读写、分配器、任务数量和块循环各自的秒数；不能把全部变化命名为内存带宽收益。

实际参与worker数来自每个目标操作的 native thread ID；process CPU/wall 包含该链的其他活动，尤其 B 的原 FFT。worker CPU 为各线程CPU时间之和，与 wall 不可相加。没有记录线程执行绝对时间线，不能据参与人数断言全程同时跑满这些核。

逻辑全场中间量读写量见 LOGICAL_ARRAY_TRAFFIC.json，不是实测DRAM流量。NUMA页面只给进程聚合分布，没有数组级首次接触归属。perf 的固定6项附加重放在正式矩阵之后，含捕获边沿及主机工作；cycles/instructions/cache-misses也不能证明DRAM带宽饱和、NUMA瓶颈、GIL瓶颈或GPU占满。

exact、输入不变与旧返回值寿命只授予本次真实输入局部回放资格。L1小网格2/8区间和112首边界、完整body收益另属条件门；未执行则不得通过。
