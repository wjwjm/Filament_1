# S5 工程经验与 formal E5 前门检查表

S5 科学收口不等于 formal E5 执行授权。后续设计或提交前逐项记录证据；任一项不满足即停止，不以重跑 GPU 或放宽 exact 合同替代排查。

1. 在本地执行真实 child shell `env`/`argv` 启动回归，确认 runner 路径与身份参数会传入 `srun` 子进程。
2. GPU 启动前确认 clean/reference `streaming_manifest.json` 存在、可解析，且与输入、生命周期和必要产物一致；不能只检查目录存在。
3. 科学输入身份排除 `build_timestamp` 等 packaging-only metadata，但保留 source、LUT、参数和数组内容的严格身份。
4. 将 bootstrap receipt 的不可变快照与运行中的可变 queue/backlog 分开；不得用 live queue 的字面值追溯验证旧快照。
5. recovery optical 内部 bootstrap 校验完成并写入一次性 `bootstrap_ready` 后，hydro 才允许 claim；重构必须先于 live claim。
6. `sha256_array` 校验 dtype、shape、C-order 数组内容；raw `.npy` SHA-256 校验文件容器。两者不得跨口径比较。
7. final audit 只调用真实存在且受支持的公开 API；对历史产物执行只读审计，并保留失败作业原终态。
8. 昂贵 GPU 运行前验证所有 final comparison 输入的完整性、哈希、可读性与保留期，包括 reference manifest、optical、ledger、POST/NEXT 和 provenance。
9. 终态审计与科学 exact 结论分层记录；Slurm `FAILED` 不得改写为 `COMPLETED`，科学 exact PASS 也不得自动变成 formal E5 准入。
