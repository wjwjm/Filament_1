# 唯一下一CPU实施合同：稳定能量诊断的表达式临时物化

本文件是下一任务的具体合同，**本次不实施、不申请新作业、不改变默认或科学资格**。当前终态仍为TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED。

## 选择及证据

只针对C3私有Raman namespace中`KHz_filament.raman.stable_field_fluence_and_loss`（raman.py:479–514），减少chunk级平方、差和积表达式的临时物化。先评审复用已有`tools.cpu_o1.cpu_o1._stable_field_fluence_and_loss_cpu_o1`（:1472）/`stable_expression` kernel（:1247）的可行性，保留原NumPy `sum(axis=0,dtype=float64)`；不要直接改成手写时间归约。

当前C3替换product、normalized derivative、RHS、Heun predictor/corrector五个helper，已做prange，未安装上述stable helper。有效电离路径也已是O1 `numba_parallel`、kernel_threads24、block_pixels32、layoutNtB，不再把“新增电离空间并行”作为方向。运行时binding JSON未直接序列化电离closure身份，身份依据为源绑定factory链；不能用Raman worker计数冒充电离busy时间。

CPU full有效LIGHT的`merged.records`中，筛选`label=RAMAN_ENERGY`且`raw_context.stage=none`，每窗16个非重叠调用，三次wall为24.556077038、24.739376383、24.893584588秒；中位24.739376383，范围0.337507550，MAD0.154208205。不能使用所有RAMAN_ENERGY父子记录的inclusive和。它是Raman/Heun内一个约8.3%整窗份额的持续成本，范围大于此前约2%的候选收益范围，但没有预测实际收益。

## 允许的唯一变化与硬边界

- 仅CPU、NumPy、complex128、full_isaacs_eq27、iir/exact_piecewise_linear、Heun及原chunk_t=32；只在私有namespace装载候选，原实现保留为oracle。
- 保持每chunk实部再虚部、`before_sum`后`loss_sum`、float64 cast/输出/归约、最终scale与独立输出。原式为`x*x`和`(x-y)*(x+y)`；负loss保留，禁止abs、截零、nan_to_num、fastmath或FMA重排。
- 候选只融合点表达式/可复用scratch，原NumPy时间归约、chunk累加次序及全局归约保持。输入只读、不覆盖E或Heun响应，scratch归属factory并隔离两次/并发调用，不缓存与输入有关的数学结果。
- 不改IIR、FFT、Heun两stage fresh响应、场更新、target q_raman、Qacc或全局closure。拒绝complex64；其projection会使用actual loss修正场幅，不属于本合同。

O1已有stable-expression实现不自动等于当前C3资格。其既有tiny测试使用`require_exact=False`，不能替代本合同的严格门；任何声称其已数值exact或已加速的表述均无依据。

源级逻辑规模：原尺寸一个32×351×301的float64 chunk为27,046,656字节；每次stable helper有12个chunk、real/imag共24组表达式，16次调用即384组。现有O1点表达式输出两个chunk，共54,093,312字节/组；原差积包含减、加、乘的逻辑输出。这里是源码可证明的数组规模与求值次数，实际allocator峰值、物理DRAM流量和带宽饱和仍未测。

## 下一任务的最小、可证伪验收

1. 先在既有Windows/真实Linux环境做原helper、现有O1、唯一候选的tiny complex128三方对照：近等幅/极小loss、正负loss、signed-zero、NaN/Inf传播、readonly、非C连续与alias fixture；finite正式输入和特殊值测试分开。要求shape/dtype、输出独立性、输入字节、逐元素/符号位与诊断规则一致。
2. 保留原`np.sum`，源级核验表达式次序、float64 cast与scale。若归约分组或特殊值行为无法证明，停在UNVERIFIED_NUMERIC_DIAGNOSTIC_ONLY；不新增容差、不换公式、不扩大为另一候选。
3. 通过后才按后续独立授权，在现有小网格真实连续八区间验证原终态/112首边界科学透明性和正常输出；固定一次热身、三次局部测量，编译/初始化另列。必须核验实际candidate closure、kernel路径及不重复的调用次数。观察不到该叶下降就停止，不为达门扫描线程/tile/精度。
4. 完整小网格exact、内存及计时干扰门通过后，才申请原尺寸确认。本任务已用满CPU/GPU各4 JobID；没有本任务内重提或追加样本权限。

## 保守成本模型

从**同一次有效CPU LIGHT**窗口/应用时间减去该叶节省量，不把PROFILE伪装成OFF逐秒重建：`T_new = T_old - L*(1-1/k) + C_added`，新增成本`C_added=UNKNOWN`。下表只列`C_added=0`的局部2倍/4倍上限情景；不以GPU时间为CPU下界。

|局部假设|每窗节省秒（三次范围）|窗口加速范围|应用加速范围|
|---|---:|---:|---:|
|2倍|12.278–12.447|1.04246–1.04358|1.03702–1.03780|
|4倍|18.417–18.670|1.06508–1.06683|1.05657–1.05779|

原始逐次输入和算式见[NEXT_CPU_COST_MODEL.json](NEXT_CPU_COST_MODEL.json)。Raman整类约106.9秒不能全部归为本候选可改成本；也不能推算96/CPU占用当量的剩余加速。

DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。
