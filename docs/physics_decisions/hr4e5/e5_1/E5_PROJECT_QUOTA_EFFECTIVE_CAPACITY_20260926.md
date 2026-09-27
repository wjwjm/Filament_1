# E5-1 project path effective capacity evidence

Status (2026-09-26): `FILESYSTEM_CAPACITY = PASS`; `PROJECT_QUOTA_EFFECTIVE_CAPACITY = PASS`; `PROJECT_QUOTA_EVIDENCE` blocker cleared for storage admission. This is a path capacity observation, not a direct read of the underlying JuiceFS quota object. It does not authorize K=8048, GPU preflight, Slurm submission, or E5-1C.

Evidence: `/data/run01/scvi806/E5_PROJECT_QUOTA_EVIDENCE_20260926_215729.txt`, SHA256 `82e29907377ae57af534f46c152ec837eb38fbc9d423963ce63c18d12d7e3c7b`. The file was retrieved read-only from `scvi-hpc` as `scvi806` on 2026-09-27; the local copy matched that SHA256.

| Path | JuiceFS `df -B1` total | Used | Available | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `/data` | 549,755,813,888,000 B (500 TiB) | 106,733,266,432,000 B | 443,022,547,456,000 B | Overall filesystem capacity |
| `/data/run01` | 549,755,813,888,000 B (500 TiB) | 106,733,272,907,776 B | 443,022,540,980,224 B | Overall filesystem capacity |
| `/data/run01/scvi806` | 858,993,459,200 B (800 GiB) | 147,324,698,624 B | 711,668,760,576 B (about 663 GiB) | Effective capacity at the formal project path |
| `/data/home/scvi806` | 1,073,741,824 B (1 GiB) | 1,073,741,824 B | 0 B | Separate restricted home path on the same mount |

The different capacities reported for paths on the same JuiceFS mount support treating 800 GiB as the currently effective limit of `/data/run01/scvi806`. The E5-1 formal campaign rolling peak estimate of about 287.849 GiB is below the measured 663 GiB available at that path. The capacity observation is time-specific; the formal entry should still perform its normal live free-space and storage-budget checks.

This update changes only the project-storage evidence gate. `E5_1B_WP2 = IN_PROGRESS`; recovery qualification, physical-GPU binding preflight, and human admission review remain separate gates. `E5-1C = NOT_AUTHORIZED`.
