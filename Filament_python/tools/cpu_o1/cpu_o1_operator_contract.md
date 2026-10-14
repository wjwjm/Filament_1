# CPU-O1 活动算子合同

任务：FILAMENT-CPU-O1-CONTRACT-AND-LOCAL-PROTOTYPE-20261014。
范围：局部 tiny CPU 原型，非生产执行器；不改变原科学 API。
科学参考 c8f3adaf41aa46c104feda6d34de402dd501d436；science tree 821fec5bdacd832b0347e1f2e84285cb54172e57；工具父提交 3d4cbb54bc216c3fcfcd85bdd9c77a250e4bb0c3。以下行号绑定该父提交。

## 活动路径与证据

full_isaacs_eq27 + exact_piecewise_linear IIR + Heun + Strang。独立电子 Kerr、自陡峭、电离和沉积保持原配置。不是 full_isaacs_eq27_complete。

|候选边界|输入和输出|读写、并行及全局依赖|源位置与状态|
|---|---|---|---|
|LUT/runtime rates|输入 float64 [Nt,Ny,Nx]；每物种同形速率|只读输入；复用 W_runtime，保留原 rate cap/scale/插值|ionization/runtime.py:64–218；lut.py:113；原活动实现复用|
|组分常量|fractions,N0,beta_rec,Ip → fractions,N0_j,beta_j,Ui|在外层按原 sum/归一化一次计算，非正/非有限总和按原均匀回退；禁止块内重新归一化|runtime.py:267–276；活动|
|full-time RK4|速率、dt、beta_j → u_j，再生成 rho_j float64|每像素时间有因果依赖；只并行空间块；每次从 u[0]=0 开始|runtime.py:278–298；首个编译目标|
|组分输出|rho_sum,Wt_total,net derivative,positive photo energy,rho_by_species|原物种顺序；每像素独占输出；保留全部可选返回及稳定诊断|runtime.py:320–365；首个编译目标|
|非 RK4 / quasi-static|原接口的其它模式|非 RK4 的 Wmax_frame 跨物种、全空间决定 n_sub；原型拒绝，不 tile 化|runtime.py:299–318、367起；原型不支持|
|总 rho 截断及沉积|rho、species terms → 原传播诊断|调用者的 [0,N0] clip 保持原位置，不移入 RK4|propagate.py:767–820；只读|
|Raman intensity/response|complex128 E → float64 intensity/I_R|保持 abs(E)**2 路径；IIR 每空间列完整时间链；IIR 系数与因果次序不变|raman.py:137–245、246–265；仅接缝设计|
|Raman product/RHS|I_R,E,Omega → complex128 RHS|时间 FFT 为完整 Nt；保留 prefactor 和运算次序；全场 norm、IR max 在外层|raman.py:266–282；仅接缝设计|
|Raman signed energy|intensity,response,Omega → u_R_signed,q_R_positive [Ny,Nx]|全场 max(abs(intensity)) 统一 scale；沿完整时间求和；零尺度走原零输出分支|raman.py:283–300；仅接缝设计|
|Heun|stage1、预测场、stage2 → E_next 与能量账本|预测场是新版本，必须重算强度、IIR、RHS；禁止响应跨版本复用|raman.py:591–676；仅参考测试|
|稳定能量 / 全局闭合|before,after → fluence/loss及全域 sum|稳定分量差分，不以两大数直接相减替换；global sum/max/norm 保留原 NumPy 语义|raman.py:479–516、636–675；只读|
|complex64投影|全局幅度缩放|当前 complex128 不执行；不能借此删去当前全局能量诊断|raman.py:517–588、626–633；当前非活动|
|横向衍射 FFT|完整 [Ny,Nx] 平面|不可用小 XY 块 FFT 替代全场；复用已有 CPUFFT 私有分派|tools/cpu_p1r2/cpu_p1r2_j2.py:158–241；已有活动组件|

## RK4 精确离散合同

W1=Wt[it]，W4=Wt[it+1]，W2=0.5*(W1+W4)。保留原 k1/k2/k3/k4 和 u2/u3/u4 表达式、乘法结合顺序。中间级仅观测，不裁剪；u_next 完成后才 clip 到 [0,1]。fastmath=False 不等同于 exact 保证。

rho_j=u_j*N0_j；rho_sum 从第一个组分开始顺序相加。Wt_total 同样顺序加权。photo_creation_j=Wt_j*clip(N0_j-rho_j,0,N0_j)。net derivative 保留 beta_rec*rho_j**2 的复合项，photoionization_energy_rate 只积累正生成并在最后 maximum(...,0)。不得混淆两者。Ui 的原默认/回退语义由参考适配器保留。

稳定诊断 min/max/count 使用块内互斥缓冲，外层固定顺序合并；不以新的舍入容差替代结果比较。原函数 return_species_terms、return_species_densities、diagnose_integrator_stability 的组合及返回 arity 必须保持。

## 原六项 NaN 白名单

仅在 diagnostics 中，shape/dtype 一致、两侧全 NaN 且 canonical hash 相等时允许：energy_step_start_J、energy_after_linear_half1_J、energy_after_raman_pre_J、energy_after_nonraman_J、energy_after_raman_post_J、energy_after_linear_half2_J。本轮不新增 NaN 例外。内核输出必须有限。

## exclusive 性能依据

J1 NumPy 分母 497.352990 s：FFT 27.8646%；Raman stage/Heun 非 FFT 23.2152%；IIR 3.2199%；电离/组分/LUT 15.0740%（自身11.3795%、LUT3.6946%）；未细分12.2409%；线性非FFT5.5926%；其它12.7928%。不能将这些份额当成 BASE_CPUFFT 418.698781 s 服务的剩余占比，也不能保证全部可并行。

REF_NUMPY：传播485.424632 s、服务509.841480 s，n=1。BASE_CPUFFT：传播中位395.139603 s、服务418.698781 s，n=2；time48/spatial24，A/B 是同实现重复。后续必须报告两个分母。
