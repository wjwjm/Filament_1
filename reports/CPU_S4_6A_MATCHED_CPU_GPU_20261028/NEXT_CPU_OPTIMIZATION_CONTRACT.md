# 唯一后续合同：先补匹配窗口证据

类型：MINIMUM_EVIDENCE_GAP_CONTRACT；当前不选择、不实施新的CPU数值优化。

缺口是同工况 CPU/GPU 的合法 P1–P3 OFF 与有效 LIGHT 数据。现有一个CPU资格窗不能排序跨后端绝对差，不能支持对Raman、FFT、数组读写或并行策略的唯一实施选择。

本地汇总修复已具体到 `entry.completed_harness_clock`、`records.cpu_completed_metrics` 和 `matrix.launch`：沿用原传播调用，只补一个OFF/LIGHT共同整窗标量shell，解析真实merged.raw_context，保持整数ns、无重叠和包含关系。不得用较宽的旧 optical_compute_window_s（包含finish/身份校验）代替完整传播调用。

后续前门：审阅本轮停止与本地修复；确认6577000真实微空间2/8区间和Nt384检查、同后端原exact、计时负例通过；重新冻结实际执行SHA/计划/资源。当前CPU一次技术修复已经用完，补证需要新的明确HPC授权，不能自动借用剩余核时或其他任务余额。

补证范围只恢复原固定small矩阵及单GPU当前环境参考/资格。只有工作量、原exact和时钟均有效且资源另获授权，才推进原尺寸；保留每窗fresh process/source/empty cache、传播预热0、原线程/精度/物理/步数/输出。未知比较器或新容差不得加入。

拿到数据后才按绝对秒数选择一个CPU实施方向。当前可改变成本份额 f 和新增成本 Δ 均为 UNKNOWN。成本模型为 T_new = T(1-f+f/r)+Δ；局部2倍/4倍情景分别是 T(1-f/2)+Δ 和 T(1-3f/4)+Δ。若仅作理想上限令Δ=0，整体速度上限为1/(1-f/2)和1/(1-3f/4)；当前没有合法f，故不给数值倍数。GPU阶段时间不作为CPU可实现下界。

DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。
