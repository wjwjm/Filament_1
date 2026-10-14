# 隔离依赖与本地复现

固定解释器：D:\Filament_1\.local_envs\cpu_o1_py312\python.exe。
原 filament-local-test 环境保持未改动。准备脚本下载的文件仅进入 .local_envs/cpu_o1_artifacts，ZIP不含环境/wheel/cache。

1. conda create 的网络下载因 repo.anaconda.com TLS SSLEOF/HTTP000 失败，日志保留；未关闭证书校验。
2. 系统 HTTPS 下载同一官方 Python3.12.8 conda 包成功。精确包离线安装后发现显式包路径不会自动补依赖，_ssl 缺DLL。
3. offline solver 尝试选未缓存新版依赖失败；改为已经核对依赖闭包的16个精确缓存包，显式离线安装成功。没有重建LLVM或升级共享环境。
4. 11个固定wheel经PyPI发布JSON中的SHA256核验，然后pip --no-index --require-hashes离线安装成功。

ENVIRONMENT.json记录已加载DLL/PYD、NumPy/SciPy BLAS线程池、Numba CPU target及线程层配置。实际并行运行线程层见测试回执。该环境只验证Windows；LinuxABI、HPC动态库、远端cache/线程层均未验证。

离线重建：先用 conda-explicit-offline.txt 中的精确本地包建立一个新prefix，再使用 requirements-hashed.txt 和保留的wheel目录执行目标解释器的 pip install --no-index --find-links <wheels> --require-hashes -r <requirements>。文件里的绝对路径是本次实测资产位置；搬运后需独立路径映射及逐包摘要核验。

测试经 run_cpu_o1_local.ps1 包装现有仓库runner；传入全新的 EvidenceRoot。它设置单层线程、generic目标、独立cache，并恢复调用前环境变量。backend/sanity失败即停止。tiny夹具不构成真实焦区输入，不输出HPC加速结论。
