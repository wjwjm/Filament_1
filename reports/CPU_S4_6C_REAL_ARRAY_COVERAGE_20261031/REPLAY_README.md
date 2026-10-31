# S46C 只读审阅包

先读 MANUAL_REVIEW_REPORT.md。主状态：MECHANISM_VALID_SMALLGRID_NOT_RUN / MANUAL_REVIEW_PENDING；默认I0，PRODUCTION_QUALIFIED=false。

包内 raw 为8858624原始元数据，execution_sources 为实际执行源码副本，frozen_inputs 为旧真实输入的原身份文件。科学大数组仍在 REAL_INPUT_INDEX.json 指定的t0目录。A1_failure保留8858616失败证据；历史S46A/6B整包没有重复放入。

使用Python 3.10或更新版本，在解压目录运行（无需NumPy、SciPy、GPU或网络）：

```text
python -I -S -B replay.py --root .
python -I -S -B replay.py --root . --full
```

快速和完整模式都检查同一批底层证据；完整模式额外打印全部分输入矩阵。它们只读文件，不导入传播程序、不运行数组、不更新状态、不申请作业。`-S`明确禁止site-packages，`-B`禁止生成字节码。

重放内容：发布包与原导出逐文件原始字节哈希，执行源码canonical-LF绑定，原输入指纹，新stage原exact分类及布局/alias，48项候选输入与旧输出寿命，108条数值记录，784条完整计时的顺序/次数/时钟/输出身份/真实worker，P_LOCAL48准入和冻结候选排序。任何缺失不以顶层PASS替代。

原numeric_gate.py保留原字节。其唯一NumPy依赖为dtype.kind；离线适配器只解析本数据集的float64/complex128/int64元数据，其余dtype拒绝，不进行数值计算。Windows实际原NumPy分类与适配器对108原记录及540负向变体结果相同；见validation/DTYPE_ADAPTER_DIFFERENTIAL.json。初版Linux因无NumPy失败的记录仍保留，未安装依赖或改原科学比较器。

summary文件不替代raw；MATRIX.csv每输入/配置10正式样本，AGGREGATE_MATRIX.csv每配置40正式样本。SEPARATED_FACTORS.csv用同输入/区组中位区分P1→Pn和同轮同线程P→P_LOCAL，负值完整保留。汇总脚本summarize.py可在另存副本中重新生成表格；它会写文件，因此不属于只读重放命令。

metadata重放只能复核保存的数值比较记录，不能在没有原数组时重新逐元素比较；实际逐元素比较在8858624计算节点完成，原数组与来源索引可追溯。首次输入与本次完整八区间传播、CPU/GPU科学等价、生产资格是不同状态。

执行工具提交420ed7e82f09180da9a1b546f4ac69daeb795f59；原CPU科学基线ed8754041625549a80959db7a8abd804d5d219ae。报告发布tip见包外PUBLICATION_RECEIPT.json，不代替执行身份。
