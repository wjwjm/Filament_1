# 追加GPU执行前的具体决策

状态：CPU小网格已通过；GPU原exact门失败，未覆盖、未豁免。未提交CPU原尺寸。既有文件和运行目录保留。

已核验事实：
- CPU 8854552 COMPLETED/0:0，8-case矩阵及LIGHT门通过；3个OFF传播窗中位48.732874894秒，应用中位81.246124952秒。
- GPU 290147 FAILED/1:0，47秒；传播前因cgroup峰值文件不可用停止。
- GPU 290163 FAILED/1:0，206秒；节点门通过，REFERENCE和QUAL_OFF均完成8区间，随后原比较器拒绝gpu_reserved_step_bytes。
- 两份OUTPUT的TREE.json和ALIASES.json原字节相同。232个数组指纹中只有4个已冻结计时向量和1个显存池保留量向量不同，其余227项相同。这不是原逐元素比较器PASS。
- allocated向量相同；reserved最大值2596857344和2689588736 bytes，差92731392 bytes，等于一个384x117x129 complex128场。原资源差异完整保留。
- diagnostics.py将gpu_allocated_step_bytes和gpu_reserved_step_bytes定义为CuPy内存池使用/保留字节，用于GPU memory gate。
- 实际RTX5090可见总显存33668988928 bytes，约31.36GiB；原full_required_bytes=34359738368（32GiB），原前门会阻断full，未下调。

需要用户明确批准的后续方案：
1. GPU累计JobID上限2到3，追加一次最长90分钟作业；累计2 GPU-hours及16附属CPU core-hours不变。已耗253秒，追加90分钟后的最坏总预留1.570278 GPU-hours及12.562222附属CPU核时。
2. 终态比较把gpu_allocated_step_bytes和gpu_reserved_step_bytes单列资源遥测，保留原数组、全指纹、layout、alias、差异和显存守门，不要求不同运行资源字节值逐位相等。其余科学字段、NaN处理、原4计时例外及112首边界合同不变；旧比较器和旧结果不修改，采用新版本明确比较视图。
3. 重新冻结GPU显存预算：优先满足原可用显存要求；若只有现有32GB卡，在独立资源探测进程核验原尺寸FFT工作区及源级活动数组寿命。正式窗口仍fresh process/new factory/传播预热0。预算不足停止full，不能降精度、删输出或改物理公式。
4. 新GPU小网格通过后继续GPU full；对应CPU full使用现有第4个CPU JobID额度，可独立并行。不把资格窗充当三次正式样本。

约束来源：用户原任务第4.1节同后端不放宽原exact、第7节小网格前门、第10节GPU最多2个JobID及一次30分钟修复。以上第1和2项改变原约束，需要明确批准；剩余GPU小时和局部指纹一致不自动放行。

无额外授权时只做只读诊断与证据整理，不提交GPU或CPU full，不发布匹配速度比。DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。