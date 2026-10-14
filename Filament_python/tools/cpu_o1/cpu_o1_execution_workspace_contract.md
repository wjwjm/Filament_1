# CPU-O1 执行、工作区与全局归约合同

状态：LOCAL_DESIGN_ONLY / NOT_PRODUCTION_QUALIFIED。

## 执行身份

实验执行描述携带 FFT 时间/空间线程策略、kernel_threads、block_size、layout、workspace ownership、reduction policy、实现标识。既有 BASE_CPUFFT 组件显式复用 time48/spatial24 私有命名空间分派；不全局替换 np.fft/xp。REF_NUMPY 使用未修改参考函数。tiny 本地测试不以48线程开展真实性能测量。

默认外部 [Nt,Ny,Nx]，float64 rates / complex128 Raman。内部 NtB 为默认，BNt 为唯一对照布局；不穷举尺寸。本地布局/block identity 与 future Linux/HPC identity 分开。

## 所有权与调度

单调用者持有工作区；无共享 Python dict/list 并行写。空间块 [start,stop) 覆盖完整像素全集，尾块明确，无重叠。每块保留完整 Nt，并以原物种顺序形成输出。块元数据预留 global offset、block ID、field version；多节点仅合同，未实现 MPI/通信。

每一时刻只有一个主要并行层：FFT stage 的 SciPy workers，或 typed kernel 的线程。BLAS/OMP/MKL/OPENBLAS 默认1；禁止外层传播池。workqueue 仅单调用者非嵌套；实现不宣称支持跨线程并发调用。线程总数不得超过本机可用核数；HPC 将来按真实 affinity 重新检查。

工作区在 rates 装入后只读，输出由一个块拥有。复用需匹配 field version、shape、dtype、layout；Heun predictor 必须新版本。E、intensity、I_R、product 的最后读点与重用点明确分开，禁止输入输出别名。正式大数组池尚未实现。

## Raman 接缝（未部署）

1. 全场构造原强度、原 IIR response；取得统一 intensity_scale、频谱倍率及固定标量。
2. 空间块内只做原逐点表达式，完整时间 FFT / IIR 链不截断；不调用完整 public stage 多次来假装全场。
3. 原全场 norm/max、时间求和和能量闭合在参考路径完成。要改 reduction tree 必须视为新数值实现，不能标为原 exact。
4. Heun predictor 产生后重复上述计算；不能跨场版本缓存动态响应。
5. 保留稳定分量差分能量路径；不能把 abs(E)**2 换成其它表达式并宣称位级相同。

## 测量和比较

冷导入、编译/缓存加载、LUT、打包/转置、kernel、回写、汇总、校验与服务分别记录，嵌套计时不相加。局部 tiny 时间只检查实现可运行，不能作为 HPC 速度。

指标包括 bitwise/canonical equality、max_abs、参考全场幅值归一化 L∞、相对 L2、逐元素相对误差及零值 mask。参考全零时不以任意 epsilon 隐藏误差；精确相等记0，否则归一化指标未定义并给原因。保留符号零 bitwise 差异。科学容差未授权。

## 依赖身份与退出

目标 Python3.12.8/NumPy2.2.5/SciPy1.15.2/Numba0.61.2/llvmlite0.44.0，fastmath=False、NUMBA_CPU_NAME=generic，独立 cache。workqueue 为本地指定线程层，导入前配置，运行后回读。Windows cache 不复制到 Linux 作为已验证对象。

输入/后端漂移、非法数值、公式/clip/return 改变立即停止候选。依赖不适配保留参考与日志，仅提出 Cython/OpenMP 同一数值内核备选，不编译 LLVM、不扩成多框架。

版本资料：https://numba.readthedocs.io/en/0.61.2/user/installing.html ；https://numba.readthedocs.io/en/0.61.2/user/threading-layer.html ；https://numba.readthedocs.io/en/0.61.2/reference/envvars.html 。实际安装及动态库见环境回执，不以 wheel 存在代替安装成功。
