# CPU研究进度同步：2026-11-06

本目录保存S4-6J/6K已经完成的轻量报告与验收回执，供GitHub查阅。
复制的报告/JSON保留原文；其中的本地或HPC路径是历史证据位置，不要求GitHub具备这些目录。
它不是完整离线审阅包，也不能据此重新对未上传的大数组逐元素比较。
本次同步不新增HPC运行、不修改科学实现或验收门。

## 当前研究参考

L1＋INTENSITY_P24＋EQ10_P48＋PHASE_ABSORB_P24＋POWER_VOLUME_CLEANUP_P24＋ION_ALPHA_CLEANUP_P24。
用户选择alpha纳入后续研究基线；原报告的性能判定和首POST审阅记录保持原文。
正式测量采用 `[384,351,301]`、complex128、冻结入口/PRE、连续八区间和每窗空1.5 GiB线性缓存。

| 阶段 | 冻结执行SHA | 结果 |
|---|---|---|
| S4-6J alpha正式14窗 | `b907561223167b9996559bc0ee4996c53a626441` | 全部exact；body配对中位改善4.0483%；完整应用配对中位净省10.336953秒；首POST保留人工审阅记录 |
| S4-6K五项基线成本六窗 | `414d8caa2fdf3f79e54b92a84fcb2a8ede998717` | exact、计时干扰、计时闭合和离线复算通过；测量结束后停止 |

S4-6J的工厂COST6干扰门失败，因此没有选择或实施工厂修复，不能称作工厂修复负收益。
S4-6K的相位/吸收组装在剩余合格逐点链中排名第一：后七区间6.487543秒；
该排名不代表全传播最大瓶颈，也不等于未来可节省秒数。

最新K两次OFF测量：完整八区间传播中位241.213424秒、应用中位392.984387秒；
后七区间body中位203.865922秒，除以7为29.123703秒/屏。
这些K成本入口时钟不与J正式14窗样本合并，不把跨轮差值解释为优化收益。

## 轻量证据

- [S4-6J总结](S4-6J/S4_6J_REVIEW_SUMMARY.md)、[逐对正式结果](S4-6J/ALPHA_FORMAL_RESULTS.md)、[正式验收](S4-6J/ALPHA_FORMAL_ACCEPTANCE.json)、[审阅包验收](S4-6J/REVIEW_PACKAGE_ACCEPTANCE.json)。
- [S4-6K总结](S4-6K/S4_6K_REVIEW_SUMMARY.md)、[逐窗成本与排序](S4-6K/COST_RESULTS.md)、[测量验收](S4-6K/COST_ACCEPTANCE.json)、[exact索引](S4-6K/EXACT_INDEX.json)、[排序明细](S4-6K/RANKING_DETAIL.json)、[资源账](S4-6K/RESOURCE_LEDGER.json)、[审阅包验收](S4-6K/REVIEW_PACKAGE_ACCEPTANCE.json)。

完整审阅ZIP、原始计时、比较器和失败记录仍保留在原研究报告目录，科学大数组保留HPC。
J原目录：`D:/Filament_1/.worktrees/cpu_alpha_factory_20261103/reports/CPU_S4_6J_ALPHA_FACTORY_20261103`。
K原目录：`D:/Filament_1/.worktrees/cpu_five_chain_cost_20261104/reports/CPU_S4_6K_FIVE_CHAIN_COST_20261104`。

K审阅包：`CPU_S4_6K_REVIEW_20261104_001.zip`，19,580,261 bytes，3130 members；
其原始二进制文件SHA256为 `c1c9a3b50c49b5b5fc09ae773ac3e1c32aa8d63412aad368861489ab5692ebb8`，
已解包运行离线审阅并通过。该ZIP不纳入本次Git同步。

## 停止状态与后续方向

S4-6K当前状态为 `COMPLETE_STOPPED / PASS_MEASUREMENT_RANKING_STOP`，无在途作业。
后续讨论转向持续每屏成本：Raman IIR、LUT/rate packing、RK4后处理，以及诊断/沉积/归约。
已编译并行的是RK4和部分Raman/Heun计算；exact-PWL IIR仍保留逐时间点NumPy表达式。
这些方向尚未实施或性能验证，不代表新作业已经获准启动。

科学子树 `821fec5bdacd832b0347e1f2e84285cb54172e57` 未变。
保持 `DEFAULT=I0`、`PRODUCTION_QUALIFIED=false`；研究参考的选择不等于通用生产资格。
