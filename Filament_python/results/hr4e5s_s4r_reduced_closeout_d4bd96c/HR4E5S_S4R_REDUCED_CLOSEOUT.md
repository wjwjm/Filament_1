# HR-4E-5S-S4R 缩减矩阵收尾报告

## 裁决

`HR-4E-5S-S4R = CLOSED / REDUCED-SCOPE / SCIENTIFICALLY_QUALIFIED / PERFORMANCE_OPTIMIZATION_DEFERRED`，采用用户授权的缩减证据范围：6-GPU hydro replay、1 optical + 2 hydro 与 1 optical + 4 hydro Streaming。

这是 reduced-scope closeout，不是原定义的完整 `1+2 / 1+4 / 1+6` formal Streaming matrix closeout。`1+6` 为 `NOT TESTED / RESOURCE_UNAVAILABLE`：作业 `237234` 在启动前被用户要求取消，scheduler provenance 为 `CANCELLED by 1812 / 0:0`，运行时间为零；因此没有 1+6 直接性能、GPU-hours 或 topology 比较证据，不能由 6-GPU replay 外推替代，也不得标记为 PASS 或 FAIL。

## 已验收的科学证据

| 证据 | 作业 | 终态 | exact 比较 | 生命周期 |
|---|---:|---|---:|---|
| 6-GPU hydro replay，384 screens | 236541 | COMPLETED / 0:0 | 1,152 / 1,152，mismatch=0 | barrier PASS；384 screens 完成后 atomic promotion 至 NEXT |
| Streaming 1+2，48 screens | 237236 | COMPLETED / 0:0 | 432 / 432，mismatch=0 | deposition (ion/IB/Raman)、POST/NEXT 三场、optical、ledger、normalized manifest、ownership、barrier/promotion 全部 PASS |
| Streaming 1+4，48 screens | 237235 | COMPLETED / 0:0 | 432 / 432，mismatch=0 | deposition (ion/IB/Raman)、POST/NEXT 三场、optical、ledger、normalized manifest、ownership、barrier/promotion 全部 PASS |

所有 exact 判断均要求 shape、dtype、canonical hash 与 `array_equal` 同时成立；没有使用容差替代。

## 性能结果

冻结 optical reference 为 `R_opt = 0.756190444 screens/s`，原 production-performance target 对应 `R_hydro = 0.831809488 screens/s`（capacity ratio 1.10）。

| topology | total GPU | hydro rate (screens/s) | capacity vs frozen R_opt | application walltime | scheduler elapsed | 结论 |
|---|---:|---:|---:|---:|---:|---|
| replay 6 hydro | 6 | 0.794324 | 1.050428 | hydro service 483.43 s | 12 m 50 s | 科学合格；距旧目标 4.51% |
| Streaming 1+2 | 3 | 0.320888 | 0.424348 | 3 h 43 m 51 s | 3 h 44 m 29 s | 科学合格；性能延期 |
| Streaming 1+4 | 5 | 0.575570 | 0.761144 | 2 h 58 m 41 s | 2 h 59 m 28 s | 科学合格；性能延期 |

6-GPU replay 相对 1/2/4 hydro 的 speedup 分别为 4.333/2.142/1.186，6-GPU parallel efficiency 为 0.722，4→6 增益为 18.63%。六个 worker 的累计测得时间介于 476.75–485.84 s，负载均衡良好。主要附加开销是 manifest lock wait（28.53%）与 lock hold（12.14%）；HR-4 solver envelope 为 57.02%，且未为 telemetry 增加 device synchronization。

在已测 Streaming 拓扑中，1+4 相比 1+2 将 hydro rate 提高 1.794 倍、将 application walltime 缩短 20.18%（45 m 11 s）。1+4 无 producer backpressure；1+2 出现 1 次 producer block（23.68 s，0.176%）。两者 pipeline tail 均为零，但 optical-active path 主导总 walltime，实际 optical/hydro overlap 分数只有 0.778%（1+4）和 1.114%（1+2）。

## 推荐与边界

- Preferred tested production topology（当前已测试方案中的首选）：**1 optical + 4 hydro GPUs**，因为它是完成并验收的 tested topology 中 walltime 最低者。
- Validated low-resource fallback（已验证的低资源回退方案）：**1 optical + 2 hydro GPUs**，科学结果完全一致但 walltime 更长。
- Untested：**1 optical + 6 hydro GPUs**（`NOT TESTED / RESOURCE_UNAVAILABLE`）；不对其作性能或 topology 推荐。
- `capacity_ratio >= 1.10` 是 production-performance target，**not a scientific hard gate**。该 target 未满足，记录为 `PERFORMANCE_OPTIMIZATION_DEFERRED`；不构成对已完成科学 exact/lifecycle 验证的否决。
- 不以 48-screen 数据宣称 full-z endurance 或正式 E5 可执行性。任何 15,000-screen估算必须单独标注为 projection，并在后续授权阶段审查。
- S4R scientific qualification no longer blocks S5。1+6 formal Streaming 缺失是 deferred performance-matrix point，**not an S5-1 prerequisite**；性能优化与可选的 1+6 重测延期至 HR-5，`HR-5 = NOT STARTED`。本 closeout 未启动 formal HR-4E-5、HR-4F 或 HR-5，且未 push、未 merge。

## 溯源

- replay execution SHA: `725dbfb0c2689c5012f98c519439f8518a5b40ef`
- replay 384-screen manifest SHA256: `5ee80fd1463a97e8be1dcdf52ee41b5e9c6c7a78eae04e78c81974901e23f5c8`
- replay provenance receipt SHA256: `86ca2db316665d836d1bee41dc6d40877de86ec7e77e273d5b41661950debaaf`
- Streaming execution SHA: `d4bd96ca21c5815b57f1a3c2ee8f8cec54dfb220`
- Streaming 48-screen input manifest SHA256: `1be727ecebcf7a635953d7c22393001d1d33254175a51a9eb984409cf63270b7`
- Streaming preflight SHA256: `e1bea044437908aa3489b6bc23780db973f304b3192833137435513d2b25e5df`

本地 `remote_readback/` 仅保存从不可变远端结果同步的派生 JSON/CSV，未复制 raw state，也未改写任何远端结果。
