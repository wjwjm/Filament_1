# 实现语义审计

本次 CPU 固定已接受 L1（全尺寸 c2644d2、小尺寸 1687944 及其必要依赖），测量工具依赖已修复 S45B 4a5f335。未合入 S45C/BAL14、S45D；KHz_filament 科学子树仍为 821fec5bdacd832b0347e1f2e84285cb54172e57。

GPU 参考选取已接受 S4R reduced d4bd96c 的实际光学核心。propagate.py、linear.py、raman.py、runner.py、config_normalize.py、device.py、longitudinal.py 的 blob 与 CPU 核心相同；整个历史子树不同主要在流式/恢复/调度文件。因此本次绑定共同核心和原 G2 PRE/POST/正常输出适配器，而不冒称运行了整个 S4R 生产流程或赋予匹配 c128 配置生产资格。

CPU 保留 G3 受控 fresh Heun 响应、lifetime identity、原校验和 L1 线性因子缓存；GPU 保留原 CuPy 算子及其原 exp/FFT 组织，不迁移 CPU 候选。两者额外执行成本不人为抹平。坐标获取只由 make_axes 改为读取同一份已封存轴；AST 去插桩证明和微空间原 prepare 对照保留。

两端保存同一逻辑终态、沉积、必要诊断和 POST，原公式、精度、八区间日程不变。GPU 必要设备传输进入应用时间；研究资格读回单列。CPU 112字段首边界保持原 108科学/4计时例外，GPU 原生 layout/alias 只与 GPU 当前环境参考比较。

输入 raw 身份、schedule/分支和 dtype/shape 是跨后端硬门。CPU/GPU 全元素差异只作描述，没有既有跨后端容差时固定 CROSS_BACKEND_NUMERICAL_EQUIVALENCE_NOT_QUALIFIED。局部测试 PASS、工作量匹配或性能差均不授予科学等价。

GPU 当前环境缺 psutil/Nsight Systems，使用 Linux procfs 仅替代 guardian/主机线程采样，不修改共享环境。不具备 kernel activity/硬件带宽计数的范围固定 UNKNOWN；CUDA event 跨度不等于设备忙时。

DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。
