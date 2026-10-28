# 历史“约27倍”口径复原

结论：两项历史数值可复算，但覆盖窗口不同，不能组成同工况硬件倍数。

GPU 原记录为 e7e900e94830405988626f7825408ebf6e3bec45，r1_telemetry，选中 screen7998–8045。原 CSV 48条 POST：首 6189003.518780609、末 6189066.994853880，跨度 63.47607327159494 s。原代码 `KHz_filament/hr4e5s_s4.py` 的公式是 **48 / 跨度 = 0.7561904435175519 screen/s**。若按47个间隔则为0.7404364759442695，后者不是原报告所用公式。原 closeout 为消费者容量采用两次观察 0.75619044/0.70111212 的较大值，不是本次科学传播性能选择规则。

同一次 optical_active 从6183224.583553778到6194114.947249961，共10890.363696183078 s。到选中首 POST 前5778.935226830654 s，选中首末跨度63.47607327159494 s，其后5047.952396080829 s。它们覆盖约15000区间长前缀执行与48个保存 screen 的不同部分，数小时与选中一分钟并不矛盾。选中窗口内还含6189036.887680640至6189047.675145036的背压10.787464395165443 s（capacity16、ordinal32、screen8030）；不通过扣除它创造“无背压传播速率”。原 perf_counter 记录没有明确 GPU completion 同步，硬件/CuPy/CUDA元数据不足。

CPU 历史 S44B 的 c2644d2、job8852224、96CPU，应用三次约331.612/331.264/332.611 s，传播约302.117/301.301/302.994 s。相邻 POST约36.11 s发生在第二线性半步之前；不是完整八区间末端或应用时间/8。该作业终态 COMPLETED 0:0、3512 s，93.653333 core-hours，为历史证据，不计入本任务资源账。

用36.11 × 0.7562会得到约27.3，但一个是 CPU 入口八区间中的 POST 递交节奏，另一个是长前缀内筛选 screen 的计数/跨度。两端输入、网格/精度、完整保存/工作量与完成边界都不能据此配对。本次不复跑历史长前缀，也不预设要得到27倍。

原始依据：

- `D:/Filament_1/Filament_python/results/hr4e5s_s4_e7e900e/remote_evidence_by_case/r1_telemetry/hr4e5s_s4_runtime_events.csv`
- 同结果目录的原 closeout JSON；`Filament_python/tools/run_hr4e5s_s4_closeout.py`。
- `D:/Filament_1/Filament_python/results/hr4e5s_s4r_reduced_closeout_d4bd96c/HR4E5S_S4R_REDUCED_CLOSEOUT.md`（reduced accepted，不代表全距离/跨后端资格）。
- S44B 原报告/CASE_RECEIPT及独立 REVIEW_FINDINGS，仅作历史锚点。

DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。
