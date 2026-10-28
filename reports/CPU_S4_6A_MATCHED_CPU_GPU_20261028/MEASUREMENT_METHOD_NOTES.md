# 计时方法依据与本轮证据范围

GPU异步提交需要完成同步或设备事件建立可解释的边界；首次context及编译缓存成本应与后续调用区分。方法依据为[CuPy Performance Best Practices](https://docs.cupy.dev/en/stable/user_guide/performance.html)及[NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。这些资料支持测量设计，不提供本项目任何秒数或速度比。

实际GPU环境保持Python3.13.7、CuPy13.6.0；官网当前版本可能较新，本轮没有升级依赖。GPU真实传播没有运行，CUDA events、kernel活动、实际传输量和显存峰值均未获得。

旧GPU历史原perf_counter缺少明确完成证据，因此历史0.7562 screen/s只按原公式复算。CPU/GPU共同传播因果顺序不构成跨系统速度差原因。process CPU/wall仅称CPU占用当量；不据此推断GIL、DRAM饱和、NUMA损失或可实现加速。

本地6577000工程检查证明的是原CPU参考/OFF/LIGHT的数值透明与整窗shell可解析；HPC实际尝试仍为fad2df6且矩阵未完成。两者身份分列，不能相互替代。
