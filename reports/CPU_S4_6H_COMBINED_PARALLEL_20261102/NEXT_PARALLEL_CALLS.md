# 三项联合之后的实际调用清单

源码基准：5388bdba5965f779ec7bcc2232b7e935a0d215cf，科学子树821fec5bdacd832b0347e1f2e84285cb54172e57。以下是源码定位和待测对象，不是已经实现或已经测得的收益。后续传播基线使用通过联合exact的三项版本。

历史三份原尺寸CPU LIGHT记录：准备父范围独占37.178661887秒/16次，沉积父范围独占24.773103509秒/8次。不能将父范围成本直接分摊给表中任一helper，亦不能相加成新的预期收益。

| 先后 | 实际调用/位置（Filament_python/KHz_filament内） | 待拆的逐点工作 | 必要真实输入及保留的语义 |
|---|---|---|---|
| 1 | propagate.py:690 → ionization/common.py:84 intensity | complex abs、square、标量缩放、原位maximum/minimum | pre-Raman之后的E、n0、I_cap；这是独立于ISAACS stage强度的另一调用，不能复用旧强度值。保持原NaN传播、dtype、裁剪次序和新输出寿命。 |
| 2 | propagate.py:816-837，nonlinear.py:28 ib_alpha | rho缩放/裁剪、photoionization_energy_rate的清理、强度分母加法和alpha除法 | 当前rho、I、photoionization_energy_rate、I_floor、sigma_ib及实际开关。不得用近似或缓存替代当前输入。 |
| 3 | propagate.py:943-951，nonlinear.py:14-26 | alpha_total两次加法、Kerr/plasma相位的乘除、最终相位相加 | 当前alpha_ib/alpha_ion_applied、delta_n、rho、k0/omega0/dz；保持语句及括号顺序。相位组装仍与已并行的两次exp更新区分。 |
| 4 | propagate.py:1117-1134 → deposition.py:23-58、121-131 | 功率源nan_to_num/maximum、alpha_ib*I | 当前三维源和标量dt/dx/dy/dz；time-axis sum与全局sum使用原调用。qmap能量和直接3D能量参考保持独立，禁止复用同一已归约结果冒充闭合。 |
| 5 | propagate.py:1219-1236 → heat.py:6-37 | term_ib乘法、Q相加、clip、nan_to_num | I、alpha_ib、ion_source、dt；原sum(axis=0)*dt与Qacc提交顺序保持。thermalization/slow_state另计，不能归并其成本。 |

执行顺序：先对上述具体调用取独占标量计时并确认真实分支命中，再对最大的一条在第1/4/8区间捕获合法输入。捕获运行与无捕获联合参考终态exact；大数组留HPC。局部对照复用固定REF/P1/P24、两个反序区组，恢复输入与指纹检查在计时外；分离操作、轴零粗分区、无小块Python循环搜索。最后才将有效局部候选单独叠加到联合基线，测原尺寸完整传播及完整应用。

当前不并发电离递推与Raman递推：它们共用后端线程资源，尚缺实际依赖、线程安全与独占成本证据。优先并行已定位的逐点表达式，既有归约和区间提交次序不改。任何局部加速都不直接折算成应用收益。
