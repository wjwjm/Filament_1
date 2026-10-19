# S5 收口验证记录（2026-09-23）

本地 clean final audit：`local_final_audit_r2.json = PASS`，验证 48/48、144 个
CURRENT/POST/NEXT 产物、canonical hashes、final optical、9 项 ledger、6 项
deposition archive、空 queue/backlog、barrier/promotion、authoritative NEXT 和
无 orphan/temp。最初一次本地审计 `local_final_audit.json` 为 FAIL：把原始
promotion 与历史 snapshot 的去运行时字段版本直接比较；修正为比对同一
`normalized_lifecycle` 后 r2 PASS。该调试报告保留，不覆盖原 `252545` 现场。

回归与检查：backend PASS；sanity 1 passed；S5-FINAL targeted 53 passed、
334 deselected；`compileall`、helper/test `py_compile`、`bash -n` 和真实
clean batch-entry audit 均 PASS。新定向回归覆盖不存在的 API、array/raw NPY
哈希口径、光学正例与单元素篡改、48-screen 合法终态、8 类必需产物缺失，以及
exact summary 的 PASS/FAIL 分类。

沿用既有补充审计 r2 的 34/34 PASS、新 clean reference 178/178 原始字节哈希、
`249485` candidate 339/339 复核，以及 strict exact comparison 432/432、
`mismatch_count=0`。详细原始路径和 SHA-256 在
`docs/physics_decisions/S5_EVIDENCE_INDEX.json`，完整结论在
`docs/physics_decisions/S5_FINAL_CLOSEOUT.md`。

没有 GPU 重跑，没有改变科学算子、lifecycle 生产规则、exact comparator、冻结输入，
没有 push/merge。`252545` 历史终态继续为 `FAILED/1:0`；S5 科学资格与
自动化告警分层收口，formal E5 为 `READY_FOR_DESIGN / NOT_STARTED`。
