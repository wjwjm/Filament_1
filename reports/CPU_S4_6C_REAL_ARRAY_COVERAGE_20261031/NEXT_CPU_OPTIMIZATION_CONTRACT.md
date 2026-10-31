# 唯一后续候选：B / P48

状态：DESIGN_ONLY / SMALLGRID_NOT_RUN / MANUAL_REVIEW_PENDING。选择依据是 PLAN_A2.json 的冻结规则与 REPLAY_FULL.json 的重算；不是实现或作业追加授权。

1. 只将 chains.eq10 的 P48 四个逐点段接入当前L1私有工厂：频谱乘法、逆变换实部除全局scale、response除同scale、二者乘积。不得叠加强度A、P_LOCAL、S4-5C/5D或其他候选。
2. 原式global abs/max、axis=0 SciPy FFT/IFFT W48、sum(axis=0)、energy_scale与正部裁剪保持；complex128、float64、native布局及原参数解析不改。禁止倒数代除法、fastmath/FMA重结合、分块归约或响应缓存。
3. 工厂私有常驻有界48线程池；作用域退出必须join，输入只读、写区唯一；异常/退役关闭池，不保持live E/stage/PRE可写引用。来源绑定、R8/G3能力取得与退役、mask24恢复及现有collector不放宽。
4. 原执行基线ed875404…，本轮局部实现420ed7e…。后续必须新工具版本和新干净研究checkout，重新做工程入口与来源验证；不得将报告tip写为历史执行版本。
5. 固定[384,117,129]先真实2区间，再8区间同期L1/P48单候选。资格/性能分开；112首边界、完整终态/能量/Qacc/POST、布局/alias、旧输出寿命、原缓存工作量均保留。主性能门为完整body第2–8区间中位至少改善3%、方向一致与原应用护栏；首POST、相邻POST和完整应用均报告，启动成本计入应用。
6. 单链局部B最差输入×区组比0.534215；旧34.388秒锚得到16.017秒排序投影。该数不是置信下界，不代表32次全程同样收益。24/48平均差约2%，未证明48稳定最优；不得看到下一轮数据后再挑线程档拼接样本。
7. 当前2个JobID已消耗，剩余预算不能转成第三个JobID；本轮不提交后续作业，不自动原尺寸传播、GPU、Hydro或MPI。只有新的明确执行授权后才进行条件接入。

未完成事项：私有传播接入、真实小网格2/8区间、112边界、完整body/应用收益门。任何局部PASS均不得代替上述事项；DEFAULT=I0、PRODUCTION_QUALIFIED=false。
