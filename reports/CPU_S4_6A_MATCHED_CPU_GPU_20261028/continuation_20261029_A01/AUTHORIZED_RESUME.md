# CPU-S4-6A 追加执行授权

用户于2026-10-29明确授权：追加一次CPU小网格技术重试，并相应增加作业额度；通过后继续GPU和原尺寸同工况对比。复用现有工具和冻结输入，物理模型不变，完成有效耗时比和主要差距归因。

累计CPU JobID上限3→4，技术修复次数1→2；CPU累计240核时、GPU最多2个JobID/2 GPU时/16附属CPU核时及原资源上限保持。两次旧CPU作业13.173333核时继续计费。此目录保存追加执行记录，旧计划、失败证据和审阅ZIP保持原字节。

串行顺序：追加CPU small → GPU small及条件full → CPU full。冻结科学子树821fec5bdacd832b0347e1f2e84285cb54172e57；复用一次封存的canonical，不重生成。仅工具预算上限变化，collector沿用6577000本地已通过实现。每次提交前核验真实环境、入口、源码、输入和最坏预算；前门失败停止扩展。

DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。