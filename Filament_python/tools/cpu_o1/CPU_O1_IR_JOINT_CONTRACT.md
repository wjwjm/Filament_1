# CPU-O1 电离与 Raman 联合本地合同

LOCAL_DIAGNOSTIC_ONLY / NOT_PRODUCTION_QUALIFIED。最终完成状态以独立审阅包为准。

## 身份与接口

科学参考 c8f3adaf41aa46c104feda6d34de402dd501d436；science tree 821fec5bdacd832b0347e1f2e84285cb54172e57；CPU-O1已接受检查点 dba727f72ed1289d21d53e6049b607e566be2b2b。新分支 codex/cpu-o1-ir-joint-local-20261014 的未提交实现以overlay清单标识。

B=BASE_CPUFFT，I=B+电离，R=B+Raman，IR=B+两候选；四者FFT时间/空间48/24。REF_NUMPY独立保留。联合入口只绑定私有传播clone，不修改原科学模块、xp或np.fft。Heun第二stage连接同一候选/分派；电离和Raman仍按原物理顺序，FFT/JIT分时，无外层进程池叠乘。

## 算子与生命周期

电离保留W1/W2/W4、四级斜率、组分顺序、最终u_next裁剪。原W_runtime/LUT和参数解释在外层，总密度裁剪在调用者。真实传播三项返回及必要species terms/densities不变；未请求稳定诊断时才可跳过，单物种别名保留。

Raman限complex128/full_isaacs_eq27；强度abs(E)**2、原exact-PWL IIR、FFT、全场max/sum/norm语义保留。Heun预测场重算动态响应，不跨版本复用。保持预计算stage1、Strang、独立电子Kerr、沉积及能量闭合。稳定能量按chunk_t=32、差积表达式、实部后虚部分别sum(axis=0,dtype=float64)并逐次累计；不改成大能量相减或块内scale。fastmath=False。

工作区实例独占，仅复用未逃逸临时量。E、组分、rho和stage响应在调用者引用期间不可覆盖；shape/dtype/layout与场版本不匹配则失效。无明确释放边界的输出保持独立分配。

## 数值回归与测量

既有固定环境/夹具exact是局部回归门；科学标量全部比较，整数精确、浮点保留符号零；特殊值按冻结合同。计时排除且单列。数组记录两侧shape/dtype、canonical/bitwise、max_abs、归一化Linf、relative L2和零值信息。新Raman有限非零差异标NUMERIC_DIAGNOSTIC_ONLY / ACCEPTANCE_TOLERANCE_NOT_AUTHORIZED，不自动授予科学/HPC资格。

大型比较按<=64MiB块处理，压缩零掩码和最多128个差异位置，禁止全量逐元素JSON。误差统计不改变求解器归约。哈希复用既有实现，额外内存/时间单列。

复用Python3.12.8/NumPy2.2.5/SciPy1.15.2/Numba0.61.2/llvmlite0.44.0，generic/workqueue单调用者；仓库backend->sanity->targeted和语法检查。tiny<=384x5x7，块1/7/32、线程1/2/4，不推断真实96核收益。本轮无HPC/真实大场/远端安装。

最终内存元数据复核：`traced_current_delta_bytes` 的实现为 max(0, current-baseline)，含义是非负保留增量，不是有符号净变化；RSS delta 为真实端点差，仍不是峰值。该说明限定字段语义，不更改原始测量记录。tracemalloc峰值只在本函数拥有tracer时测量；已有调用方tracer不重置。独立只读复核未发现阻断缺陷，64×5×7源通过正常builder生成，无裁切；最终数组shape另见独立验证回执。
