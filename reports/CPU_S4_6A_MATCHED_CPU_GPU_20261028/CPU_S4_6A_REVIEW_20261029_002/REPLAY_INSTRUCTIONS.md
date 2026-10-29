# 离线复算

解压到新目录后，使用任意已安装的Python标准库解释器执行：

```text
python -s -B replay_readonly.py --root .
```

本命令只读，不读科学数组，不连接SSH、不提交作业、不需要NumPy/CuPy。JSON会忠实返回INCOMPLETE及GPU full LIGHT失败；它同时给出有效OFF比值、原始时钟闭合、资源与源码复算。该INCOMPLETE由预先声明的细分干扰失败及未带入包的大数组字节构成，不可改为科学PASS。

`validate_final_delivery_A01.py --root . --output <包外新文件.json>`进一步检查完整的有限终态：两端两尺寸各三次OFF、CPU/GPU各四个JobID预算、双执行源包、三叶零残差、三项LIGHT通过和原尺寸GPU LIGHT失败。预期为PASS_EXPECTED_LIMITED_REPLAY。输出目标必须是一个不存在的新文件；不要写回原始证据。

`REPLAY_SCOPE.json`是当前白名单：原formal CASE_RECEIPT和RAW计时重算样本，REFERENCE、QUAL、AUX及失败记录不能补正式P1-P3。两份约58MB的原CPU矩阵在包内保留但超出8MiB解析界限；实际复算使用原CASE_RECEIPT/RAW而不增加解析上限。双源包原manifest按独立CPU/GPU提交验证。旧GPU状态记录另有byte-identical SCHEDULER命名副本，使既有资源解析器能识别；同一主JobID只收费一次。

`PACKAGE_METADATA_MANIFEST.json`只定义本次交付文件raw bytes，原PV2 tracked文本身份不变。`ZIP_FINAL_RECEIPT.json`在ZIP外记录压缩包hash、大小、CRC和复算结果，避免自引用。

源码：source_roles/CPU、source_roles/GPU。原科学数据位置/size/raw SHA索引：各case OUTPUT_RAW_INDEX与SCIENTIFIC_ARRAY_RETENTION.json；离线没有重新检查远端数组字节。

最终验收日志在local_validation/。Windows使用仓库run_local_tests.ps1定向入口；Linux使用既有独立Linux解释器、原工具和复制到Linux原生文件系统的同一元数据，不以Windows模拟Linux。早期/mnt/c运行被终止的记录保留，不计PASS。
