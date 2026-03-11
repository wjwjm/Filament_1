# khzfil_baseline (modular)

A modular rewrite of the minimal kHz filamentation baseline:
线性衍射：采用分步角谱（傍轴）算法。- Split-step angular spectrum (paraxial) for linear diffraction.
局域时域非线性步：包含 Kerr 相位、电离（速率方程）、等离子体折射率与吸收。
- Local time-domain nonlinear step: Kerr phase, ionization (rate eq.), plasma index and absorption.
脉冲间慢时演化：密度空洞/热透镜的二维扩散。- Slow-time (between pulses) 2D diffusion of density-hole/thermal lens.






## Dependencies

Install minimal dependencies before running:

```bash
pip install -r Filament_python/requirements.txt
```


### NPZ to MATLAB conversion

After simulation, you can convert output `.npz` to `.mat` and optionally delete source `.npz`:

```bash
python npz2mat.py --npz khzfil_out.npz --mat matlab_output/khzfil_out.mat --remove-npz
```

For HPC `sub.sh`, NPZ->MAT conversion is enabled by default (`CONVERT_TO_MAT=1`),
with default output directory `matlab保存数据/` and default `REMOVE_NPZ=1`.

Custom example:

```bash
sbatch --gpus=1 --export=MAT_DIR=matlab_output,MAT_NAME=run1.mat,REMOVE_NPZ=1 ./sub.sh
```

Disable conversion when needed:

```bash
sbatch --gpus=1 --export=CONVERT_TO_MAT=0 ./sub.sh
```

## Usage

```bash
# CPU default (NumPy)
python -m khzfil.cli

# GPU (CuPy)
UPPE_USE_GPU=1 python -m khzfil.cli


提交作业：sbatch  --gpus=卡数     ./run.sh
查看作业情况：squeue
结束作业：scancel  作业号（作业号执行squeue即可查看到）
实时查看输出文件：tail -f   文件名
可使用vim编辑脚本中最后一行修改为您的代码文件或完整命令即可

排除节点#SBATCH -x g0601,g0602,g0605
指定节点#SBATCH -w g0601,g0602,g0605
查看全部节点状态sinfo -N -l
idle  ：完全空闲
alloc ：全部核心被占
mix   ：部分占用，仍可跑作业
down  ：宕机或离线
drain ：管理员标记为维护，不再接收新作业
drng  ：drain + 仍有作业在跑
```

Outputs `/mnt/data/khzfil_out.npz` with axes and basic diagnostics.

## Modules

- `khzfil/constants.py`: Physical constants and helpers.物理常数与辅助函数。
- `khzfil/device.py`: Backend selector (`xp`) for NumPy/CuPy.后端选择器（xp），支持 NumPy/CuPy。
- `khzfil/config.py`: Dataclasses for grid, beam, propagation, ionization, heat, run.用数据类封装网格、光束、传播、电离、热累积、运行参数。
- `khzfil/grids.py`: Mesh/axes and spectral operators.网格/坐标轴及谱域算子。
- `khzfil/linear.py`: Angular spectrum propagator.角谱传播器。
- `khzfil/ionization.py`: Intensity, power-law W(I), rho(t) evolution.计算光强、幂律 W(I)、电子密度 ρ(t) 演化。
- `khzfil/nonlinear.py`: Kerr phase, plasma phase, IB absorption, NL application.Kerr 相位、等离子体相位、逆轫致吸收，并应用非线性效应。
- `khzfil/heat.py`: Heat accumulation and slow-time diffusion.热累积与慢时扩散。
- `khzfil/propagate.py`: Strang split per-pulse propagation and heat accumulation.每脉冲的 Strang 分裂传播及热累积循环。
- `khzfil/diagnostics.py`: Peak intensity, energy, and saving helpers.峰值光强、能量监测与存盘辅助。
- `khzfil/utils.py`: Gaussian sources and utilities.高斯源与其他工具函数。
- `khzfil/cli.py`: Demo runner and file output.演示程序与文件输出入口。

KHz_filament 仿真参数 README

diag = {
  "z_axis":               1D np.ndarray [Nz],                 # 采样的 z（米）
  "U_z":                  1D np.ndarray [Nz],                 # 每步脉冲能量 (J)
  "I_max_z":              1D np.ndarray [Nz],                 # 全场峰值强度 (W/m^2)
  "I_onaxis_max_z":       1D np.ndarray [Nz],                 # on-axis 像素（Ny/2,Nx/2）处对 t 取 max 的强度
  "I_center_t0_z":        1D np.ndarray [Nz],                 # on-axis 像素处 t≈0 的强度
  "w_mom_z":              1D np.ndarray [Nz],                 # 二阶矩光斑半径 (m)
  "rho_onaxis_max_z":     1D np.ndarray [Nz],                 # on-axis 电子密度的时间最大值 (1/m^3)
  "I_peak_q99_z":         1D np.ndarray [Nz],                 # 稳峰：I 的 top-0.1% 均值
  "rho_peak_q99_z":       1D np.ndarray [Nz],                 # 稳峰：max_t rho 的 top-0.1% 均值
  "E_dep_z":              1D np.ndarray [Nz],                 # 每步沉积能量 (J)，? Qslice dx dy
  "fwhm_plasma_z":        1D np.ndarray [Nz],                 # 等离子体通道等效 FWHM 直径 (m)
  "fwhm_fluence_z":       1D np.ndarray [Nz],                 # 激光能量密度(通量)等效 FWHM 直径 (m)
  # 可选：
  "rho_onaxis_t_z":       2D np.ndarray [Nz, Nt]              # 每步 on-axis 的 rho(t)
}

本项目支持两种线性传播模型（UPPE 与 抛物近似/paraxial）以及多种非线性与等离子体物理（Kerr、自陡峭、拉曼、ADK/PPT 电离、等离子体吸收与复合、气体热扩散）。本文档说明仿真摘要中各参数的含义、常见取值范围与调参建议。

1) Backend / dtype

Backend：numpy 或 cupy（GPU）。

GPU 模式由 UPPE_USE_GPU=1 或 CLI --gpu 控制；设备通过 CUDA_VISIBLE_DEVICES 选择。

dtype：fp32（推荐，快且够用）或 fp64（更稳，但慢/占内存大）。

建议

大多数成丝仿真用 fp32 即可；极端高动态范围（>10?）可尝试 fp64。

先在小网格验证，再上大网格/GPU。

2) Grid (XYZT)

Nx, Ny：横向网格点数。

Lx, Ly：横向窗口大小（米）。

Nt：时域采样点数。

Twin：时间窗宽度（秒）。

选型建议

视野要容纳 ≥ 3~4 × w? 的半径（即光斑在边界衰减至噪声），常用：Lx≈Ly≈(4~8)×w?。

横向采样：Δx=Lx/Nx，要满足角谱传播的带宽 Nyquist；通常 Nx, Ny = 512~1024 够用。

时间窗：Twin ≥ 6~8 × τ_FWHM，以防频谱泄漏与边界折返。

Nt=64~256：短脉冲（30–100 fs）一般足够；需要自陡峭/拉曼细节时可以略增。

3) Steps (z) 与自适应阈值

z_max：总传播距离（米）。

dz：名义步长（米）。

AutoSubstep：是否按阈值自适应子步。

dz_min, grow×：最小子步与接受后放大因子。

阈值：

lin_phase≤…：线性相位增量上限（|kz|max·dz）。

alpha·dz≤…：吸收 α dz 上限。

kerr_phase≤…：Kerr 相位上限 k0 n2 Imax dz。

Imax_growth≤…：步末强度相对增长的容差（>0 回退）。

调参建议

线性仿真：dz ~ (0.25~1) mm；UPPE 较严格时取更小。

含非线性：以 kerr_phase≤0.2~0.3 rad 为基准；接近焦点适当减小 dz。

开启自适应：可避免焦点/强吸收区“跳步”。

4) Linear

model："uppe" 或 "paraxial"。

factorize：UPPE 线性算子是否按 (t × k⊥) 因子化（省内存/更快）。

chunk_t：UPPE 三维 FFT 的时间分块（显存不够时调小）。

选择

UPPE：宽带（>5~10%）或需要色散/群速差的场景。

paraxial：窄带、弱色散/短程近轴场景，速度快。

对比两者线性基准（关闭一切非线性与电离），检查焦点/能量守恒。

5) Beam

λ0, n0, w0, τ_FWHM：入射脉冲与介质参数。

f (focal_length)：薄透镜焦距，初始化时注入 相位 exp[-i k0 (x2+y2)/(2f)]。

E0_peak / energy_J / P0_peak：支持三种输入方式。

- `energy_J` 与 `P0_peak` 必须二选一（不能同时给）。
- 若给 `E0_peak`，则直接使用该电场峰值。
- 若给 `energy_J`，程序先反推 `E0_peak`，再做能量归一化。
- 若给 `P0_peak`（峰值功率，W，定义为 t=0 时横截面积分功率），程序先反推 `E0_peak`。

常见范围

λ0=800 nm，n0≈1.00027；w0=0.1–2 mm；τ_FWHM=30–100 fs；f=0.3–2 m。

能量 10??–10?3 J 量级（单脉冲）。

临界功率：P_cr ≈ 0.148 λ?2/(n? n?)；比如 n?≈3.2e-23 m2/W 时，@800 nm 得 ~3 GW。

6) Energy（归一化）

config 与 actual：目标能量与归一化后的实际能量（差异<1e-3 为佳）。

E0_peak：归一化后对应的峰值电场（V/m）。

提示

若设了 energy_J，会自动按 (U_target/U_now)^0.5 缩放初场。

若设了 P0_peak（且未给 energy_J），则不会做能量目标归一化。

能量哨兵会在传播中定期检查能量是否异常增长。

7) Repetition / Run

f_rep：重复频率（Hz），用于热学模块。

pulses：脉冲数；kHz 高频下用于“脉间累积”的热/密度扰动。

8) Kerr

ON/OFF：开启/关闭 Kerr 非线性。

n2：Kerr 系数（m2/W），空气常用 3.2e-23。

P_cr：临界自聚焦功率，用于定性判断是否易成丝。

建议

宽带脉冲 + UPPE 时，Kerr 会伴随自相位调制引起光谱展宽；dz 需适当减小。

9) Ionization（ADK / PPT / Powerlaw / OFF）

PPT 参数：Ip_eV, Z, l, m，安全裁剪：W≤..., I≤...。

β_rec：三体复合系数（m3/s）。

σ_ib：反常布里渊（碰撞）吸收截面（m2）。

I_cap/W_cap：强度与电离率上限（防溢出/稳定性）。

建议

先以 ADK/PPT=OFF 做线性与 Kerr-only 基线；再逐步开启 PPT。

对极端强度（>101? W/m2）务必加严格裁剪，避免电子密度爆炸导致数值失稳。

10) Self-steepening（自陡峭）

ON/OFF 与 method：tdiff（默认，时域微分）或 fft。

需要较好的时间分辨率与足够宽的 Twin；dz 过大时容易数值振铃。

11) Raman（分子转动拉曼）

ON/OFF；f_R（延迟占比）；model（例如 rot_sinexp）；T2、T_R。

method：iir（时域因果滤波，省内存）或 fft（频域卷积）。

会引入慢时间折射率槽（特别是高重频），以及脉内延迟响应。

12) EnergyGuard（能量哨兵）

every：每 N 个 z 步检查一次；skip-first：忽略最初 K 次检查；

blowup×：能量相对初始放大的容忍倍数（超过则报警/可回退）。

13) 诊断与输出（diag 字段）

基础

z_axis：记录的 z 位置（与保存间隔一致）。

rho_onaxis_max_z：轴上最大电子密度 vs z。

rho_onaxis_t_z（可选）：每个保存点的轴上 ρ(t) 曲线。

扩展（已集成）

U_z：脉冲能量 vs z（时间与横向积分）。

I_max_z：全时空最大强度 vs z（稳健的“聚焦指标”）。

I_center_t0_z：固定 t=0、轴上强度 vs z（易受走时影响，仅参考）。

w_mom_z：二阶矩半径 vs z（在焦点处最小）；用于估计焦点位置。

t_peak_z：轴上强度随 t 的峰值时间 vs z（可看群延迟变化）。

焦点判定

推荐以 w_mom_z 的最小值对应的 z 为焦点；再辅以 I_max_z 的峰值验证。

若需要亚步长精度，可对最小值附近 5–7 点做二次拟合求极小值。

14) 常见场景配方
A. 线性验证（推荐第一步）

prop.linear_model="uppe" 或 "paraxial"；全部非线性、PPT、拉曼、自陡峭 OFF。

dz = 0.25~0.5 mm；Twin ≥ 8×τ_FWHM；Nx=Ny=512~768。

检查：能量守恒（U_z≈常数）、焦点位置与薄透镜预测是否一致。

B. 宽带脉冲（~80 nm 带宽）

linear_model="uppe"；保留色散；dz 略小（0.25–0.5 mm）。

开 Kerr、自相位调制；若出现时域振铃，适当加时间窗与频域掩膜。

自陡峭 ON (tdiff)，Twin 足够宽；Raman 可先 OFF 再逐步 ON。

PPT：待 Kerr-only 稳定后再打开，并调紧 I_cap/W_cap。

C. 单脉冲成丝

能量/聚焦使峰值功率 超 P_cr 多倍；Kerr ON，PPT ON（或先 OFF 验证自聚焦）。

dz 在焦点附近可开启自适应，限制 kerr_phase≤0.2~0.3。

观察：I_max_z 是否出现平台；rho_onaxis_max_z 是否达 1023–102? m?3 量级；
Δn_gas 为负值（慢时间槽）。

D. 高频热效应（kHz–MHz）

run.Npulses > 1，heat 模块启用，设置 f_rep、D_gas、gamma_heat。

关注 Δn_gas 的脉间累积（对准直传播/多脉冲优化时尤为关键）。

15) 排查与小贴士

焦点错位（UPPE vs paraxial）
统一薄透镜相位实现；做线性基准；确保 dz 足够小、FFT 掩膜合理（避免 alias）。

I_center_t0_z 的震荡
改看 I_onaxis_max_t_z 与 t_peak_z；固定 t=0 容易受走时影响。

能量“泄露”
窗口太小、频域泄漏、dz 过大、谱掩膜过硬都会引起；先做线性验证再开非线性。

GPU 显存
减少 Nx,Ny,Nt；调小 chunk_t；关闭不必要的诊断；使用 free_diag_to_cpu 与显存清池。

16) 配置键位对照（简表）

grid：Nx, Ny, Lx, Ly, Nt, Twin

beam：lam0, n0, w0, tau_fwhm, E0_peak | (energy_J xor P0_peak), focal_length

propagation：

基本：z_max, dz, linear_model("uppe"|"paraxial"), strang

自适应：auto_substep, dz_min, grow_factor, max_linear_phase, max_alpha_dz, max_kerr_phase, imax_growth_limit

线性实现：full_linear_factorize(False/True), linear_chunk_t(=8)

其它：use_self_steepening(True/False), progress_every_z, show_eta, free_diag_to_cpu(True), gpu_hard_gc_every

ionization：model("ppt"|"adk"|"powerlaw"|"off"), Ip_eV, Z, l, m, beta_rec, sigma_ib, I_cap, W_cap

raman：enabled, f_R, model, T2, T_R, method("iir"|"fft")

heat：D_gas, gamma_heat, f_rep

run：Npulses

快速检查清单（每次开新参数前）

线性基准通过（能量≈常数、z_of_focus 正确）。

dz / 阈值 合理（焦点处不过步、无明显震荡）。

窗口/Twin 充足、频域掩膜平滑。

逐项开启物理（Kerr → 自陡峭 → 拉曼 → PPT），每加一项都看 U_z 与 I_max_z 是否合理。

诊断优先看：w_mom_z（焦点）、I_max_z（平台/峰值）、t_peak_z（走时）、rho_onaxis_max_z（等离子体量级）。
