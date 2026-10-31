# A1 失败与 A2 技术重试

8858616 在 cb1106 终态 FAILED，主allocation为93秒、96核，计2.48核时；batch/extern不重复计费。节点资源、源码、库版本门通过，worker只运行约5秒，尚未进入STAGES payload核验或任何性能样本。

直接故障：新runner访问 `RamanConfig.method`，而原配置加载器会舍弃该JSON扩展键。原传播代码使用 `str(getattr(raman_conf, "method", "iir")).lower()` 及 chunk_pixels 的65536默认值。本地先前测试手工给出了kwargs，未覆盖实际配置加载后的对象，因此未提前捕获此错误。

修复仅使stage_kwargs沿用原传播入口的取值规则。测试现改为读取实际冻结配置、调用原load_all、构造微空间stage，并通过原完整exact比较器及原TREE/ARRAYS.zip读回环。22项测试、backend、sanity和语法检查通过。A1全部记录保留于attempt_A1，无性能样本可以复用或混入A2。

A2使用新的干净checkout/run，增量基线为A1执行提交7c0ac61；原科学子树和原输入不变。提交前独立核算2.48已用+48最坏预留=50.48核时，低于96核时。A2是第二个、也是本轮最后一个JobID；不存在第三次技术重试权限。

A1实测perf允许用户态cycles/instructions/cache-misses。A2在正式矩阵结束之后，只对pre_field的A/B与REF/P24/P_LOCAL24补六个固定计数器回放，每项五次，记录实际线程ID；不将该回放混入性能样本，不宣称DRAM bytes或带宽饱和。正式计划无有效数据后改动，也不扫描参数。

第二个作业名额已用于修复；如机制结果选出赢家而未在当前allocation完成经过本地资格的单一小网格接入，必须明确交付“小网格全路径未验证”，不得追加JobID或授予性能晋级。
