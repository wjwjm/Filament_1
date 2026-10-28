# S46A 时钟合同（性能产生前封存）

- `process_service_s`：父进程从 Popen 前整数 ns 到 wait 返回，包含导入、研究收尾及至多约 1 秒 guardian 采样退出延迟；后者单列，不当作应用工作。
- `application_service_s`：entry.main 的整数 ns 到原正常 writer 返回。含源码/输入身份、参数/工厂、必要 H2D/D2H、空因子表首建、传播及原正常磁盘输出；不扣除某后端启动开销。未单独分辨的 CUDA context/JIT/FFT 计划成本标 UNKNOWN。
- `window_completed_s`：科学输入已就绪，进入整条传播调用至最终场完成；原调用内的诊断、沉积、POST 和原输出仍在内。GPU 开始前和函数返回完成同步。
- `first_interval_s`：完整 interval0 body。`later_body_s`：interval1 起点至 interval7 终点的同一主机墙钟跨度，包含原循环间工作。CPU 另存 body 逐项和；不以 CUDA event 总和替代。
- 原 `POST_COMMITTED` 来自 host POST payload 已物化并提交后的 perf_counter。它位于第二线性半步前；保留首 POST、7个相邻 POST 和最终场完成边界，不混作同一事件。

OFF 仅整窗、interval1/final 完成锚点和原计时。GPU REFERENCE 不加循环计时锚点，QUAL_OFF 对其做同后端 exact；与 REFERENCE 的宏观干扰另列。LIGHT 有限私有 scope 使用主机 wall/process ns 和同一 current stream 的延迟读取 CUDA events；不逐算子 device-wide 同步。AUX_SERIALIZED 显式同步 scope，结果仅为独立阶段诊断，记录对 P1_OFF 的应用干扰比。

应用窗口在正常输出后结束。同后端终态快照、逐元素比较、文件 raw index 等研究工作在该端点后；资格首边界捕获和 GPU 输入读回若处于端点内，原样计入 QUAL，不将 QUAL 当正式 P1–P3 OFF。

逐后端/尺寸三对 OFF/LIGHT，对 application/window/later 分别筛选：每对 [0.90,1.10]、中位 [0.95,1.05]。不删异常样本。父子 scope 只在本次调用同一计时域内求独占；host 与 device 不相减，不把事件跨度称 kernel busy time。原生 builtin 同步的范围可作为 completed wall，其他异步细项分别呈现。无法闭合的余量保留 UNKNOWN。

DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。
