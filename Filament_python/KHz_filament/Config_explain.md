# `Filament_python/KHz_filament/config.py` 参数说明

> `config.py` 是默认值来源；`config_schema.py` 定义字段和 rate 别名；`config_normalize.py` 负责兼容、校验和派生量。本页解释稳定的公共配置含义，不替代这三个源码文件。

## 配置文件与加载规则

当前仓库提供三份可运行 JSON 配置：

| 文件 | 定位 |
| --- | --- |
| `Filament_python/config_ref.json` | 小网格、短距离的 smoke test/reference 基准 |
| `Filament_python/khz_config.json` | 常规高分辨率传播 |
| `Filament_python/khz_config_lut.json` | Talebpour LUT 缓存和焦区窗口 |

`confio.py` 支持 JSON/YAML/TOML。规范化时，`energy_J` 与 `P0_peak` 互斥；二者的场幅归一化会在运行入口建立离散时空场后执行，避免把高斯解析面积错误套用于其他横向轮廓。`Twin` 缺省时按 `8 * tau_fwhm` 补齐，`species.fraction` 会归一化。

---

## GridConfig（数值网格/窗口）

- **Nx (int)**：横向 x 方向采样点数（空间网格）。
- **Ny (int)**：横向 y 方向采样点数（空间网格）。
- **Nt (int)**：时间采样点数（时域网格）。
- **Lx (float, m)**：x 方向物理窗口大小（模拟区域宽度）。
- **Ly (float, m)**：y 方向物理窗口大小（模拟区域宽度）。
- **Twin (float, s)**：时间窗口长度（脉冲在时域的仿真窗口）。

> 通常：`dx=Lx/Nx`，`dy=Ly/Ny`，`dt=Twin/Nt`（是否严格这样取决于内部网格实现）。

---

## BeamConfig（入射束/脉冲参数）

- **lam0 (float, m)**：中心波长（真空波长）。
- **n0 (float)**：背景折射率（线性介质/空气平均折射率）。
- **w0 (float, m)**：束腰半径（1/e **电场**半径，高斯束常用定义）。
- **tau_fwhm (float, s)**：脉冲强度包络的 FWHM（半高全宽）。
- **E0_peak (float)**：电场幅值的峰值（注释强调“不是峰值强度”）。当你直接指定电场初始幅值时使用。
- **energy_J (Optional[float], J)**：单脉冲能量。与 `P0_peak` 互斥；当 `E0_peak=0` 时用于反推幅值。
- **P0_peak (Optional[float], W)**：脉冲中心的峰值功率。与 `energy_J` 互斥；在离散横向轮廓建立后归一化，保证实际网格上的峰值功率匹配目标。
- **transverse_profile (Optional[dict])**：横向强度轮廓。省略时沿用 `w0` 为 `1/e` 场半径的 Gaussian；可显式写为 `{ "type": "gaussian", "radius_m": ... }` 或 `{ "type": "flat_top_cosine", "radius_m": ..., "edge_start_fraction": 0.9 }`。平顶轮廓定义的是强度，电场幅度自动取其平方根。
- **focal_length (float, m)**：透镜焦距（用于加入聚焦相位/等效曲率）。
- **n2_air (float, (m²/W) 或等效单位)**：空气 Kerr 非线性系数 n₂（用于 `Δn = n2 * I`）。

---

## PropagationConfig（z 向传播与线性/数值控制）

### 基本传播范围与分裂
- **z_max (float, m)**：最大传播距离。
- **dz (float, m)**：名义 z 步长（若启用自适应子步，会在此基础上缩放）。
- **strang (bool)**：是否使用 Strang splitting（典型：线性半步 → 非线性整步 → 线性半步）。

### 线性模型 / UPPE 相关
- **linear_model (str)**：线性传播模型选择：
  - `"uppe"`：UPPE（更全波/更宽角度频谱意义下的线性传播）
  - `"paraxial"`：傍轴近似线性传播
  - `"bk_nee"`：Brabec–Krausz NEE 相关线性项形式
- **full_linear_factorize (bool)**：线性算子/相位是否按 ω 切片分解处理以省内存：
  - True：逐频率切片做 2D FFT/相位（更省内存、更稳，可能慢一点）
  - False：可能构建 3D（Nt×Ny×Nx）级别相位/算子（更快但占内存）
- **use_self_steepening (bool)**：是否开启自陡峭（非线性项中加入时间导数/频域因子修正，导致前沿变陡、频谱展宽等）。

### 独立非线性反馈开关（Phase 2）

以下字段均位于 `propagation`。省略时保留旧配置行为；显式新字段优先于旧的 `raman.enabled` / `raman.absorption`，并会在启动摘要和 NPZ 元数据中记录最终有效值。

- **use_electronic_kerr (bool | null)**：是否将已计算的 `n2*I` 加入传播相位；关闭后仍保存原始电子 Kerr 诊断。
- **use_raman_phase (bool | null)**：是否将已计算的 `n_R I_R` 加入传播相位；关闭后仍保留 Raman convolution 与原始 Raman 诊断。
- **use_raman_full_operator (bool | null)**：仅用于显式启用 `full_isaacs_eq27` 复场算子；默认/省略为关闭，且不能与 `use_raman_phase=true` 或 legacy Raman absorption 同时使用。
- **raman.nonlinear_split_order**：完整 Raman 算子与其余非线性项的组合顺序，可选 `after_other`（兼容默认）、`before_other`、`strang`。严格生产候选使用预飞收敛测试选择的顺序。
- **use_plasma_phase (bool | null)**：是否将由电子密度得到的 plasma phase 加入传播相位；关闭后仍求解并保存 `rho` 与原始 plasma phase。
- **use_ionization_loss (bool | null)**：是否将已计算的 `alpha_ion` 加入 `alpha_total`；关闭后仍保存原始电离率、电子密度和潜在沉积。
- **use_raman_absorption (bool | null)**：是否将 Raman absorption 加入 `alpha_total`；可与 Raman phase 独立设置。
- **use_ionization_solver (bool | null)**：是否执行电离率和电子密度求解。纯线性或纯 Kerr 基准可显式设为 false，避免无用电离计算。

`null` 只表示“未显式设置”，通常不应写入 JSON；旧 JSON 未包含上述字段时，等效关系为：电子 Kerr / plasma phase / ionization loss 为 ON，Raman phase 跟随 `raman.enabled`，Raman absorption 跟随 `raman.enabled and raman.absorption`，电离求解跟随是否存在 `species`。

### 空气色散模型（线性折射率/色散）
- **air_model (str)**：空气折射率/色散模型名称（此处为 `"ciddor_simple"` 简化 Ciddor）。
- **air_T (float, K)**：空气温度（用于折射率模型）。
- **air_P (float, Pa)**：空气压强（用于折射率模型）。
- **air_CO2 (float, 体积分数)**：CO₂ 含量（简化模型里可能是占位或弱依赖）。

### BK-NEE 线性项参数（当 `linear_model="bk_nee"`）
- **nee_beta2 (float, s²/m)**：二阶群速度色散系数（k''，GVD）。
- **nee_denom_floor (float)**：分母下限（避免如 `1 + Omega/omega0` 在 `Omega ≈ -omega0` 附近奇异或数值爆炸）。

### 进度输出/日志
- **progress_every_z (int)**：每多少个 z 步打印一次进度（0=不打印）。
- **show_eta (bool)**：是否打印 ETA（预计剩余时间）。

### 自适应子步（adaptive substep）
- **auto_substep (bool)**：是否启用基于稳定性/相位增量等条件的子步长控制。
- **dz_min (float, m)**：允许的最小步长（防止无限缩小）。
- **grow_factor (float)**：当条件满足/“接受”后，下一步尝试放大步长的倍率。
- **shrink_factor (float)**：当条件不满足时，缩小步长的倍率。
- **max_linear_phase (float, rad)**：线性算子导致的最大相位增量阈值（约束 `|Δφ_linear|max`）。
- **max_alpha_dz (float)**：吸收系数 α 与步长 dz 的乘积上限（约束每步吸收不要过强）。
- **max_kerr_phase (float, rad)**：Kerr 非线性相位增量阈值（约束 `|Δφ_Kerr|max`，更保守通常更稳）。
- **imax_growth_limit (float)**：单步最大峰值强度增长限制（用于检测/抑制数值“爆冲式”增长）。
- **safety_mode (str)**：安全模式开关/策略名称（如 `"on"`；具体含义需看实现处如何处理）。
- **energy_probe_every (float)**：能量探针检查频率（具体是“步数”还是“距离”取决于实现）。
- **energy_probe_tol (float)**：能量守恒/漂移容忍阈值（如相对误差 0.03 = 3%）。
- **diag_extra (bool)**：是否导出更多诊断信息（“大量信息”的开关）。

### 焦点附近的步长/窗口控制
- **focus_window_step (bool)**：是否在焦点附近启用特殊步进（通常更小步长）。
- **focus_center_m (float, m)**：焦点中心位置（z 坐标，常取透镜焦距附近）。
- **focus_halfwidth_m (float, m)**：焦点区域半宽（在 `[center-halfwidth, center+halfwidth]` 范围内启用特殊策略）。
- **dz_focus (float, m)**：焦点区域内使用的更小步长。
- **limit_focus_window (bool)**：是否限制“焦点窗口”之外的传播窗口/策略（具体行为看实现）。
- **window_halfwidth_m (float, m)**：窗口半宽（常围绕 `focus_center_m` 定义范围）。

---

## TimeMode（电离“时间近似”模式类型别名）

`TimeMode = Literal["full", "qs_peak", "qs_mean", "qs_mean_esq"]`

- **"full"**：全时域计算（随时间演化积分/求解）。
- **"qs_peak"**：准稳态近似，用峰值强度代表（不积分全波形）。
- **"qs_mean"**：准稳态近似，用某种平均强度/平均量代表。
- **"qs_mean_esq"**：准稳态近似，用电场平方的平均（或等效强度平均）代表。

> `mean/mean_esq` 的精确定义取决于电离模块实现，但总体是：用简化统计量替代全时域积分以加速。

---

## Integrator（全时域电离积分器类型别名）

`Integrator = Literal["rk4", "euler"]`

- **"rk4"**：四阶 Runge–Kutta。
- **"euler"**：显式欧拉。

---

## IonizationConfig（电离/等离子体相关配置）

### 物种与通道组合
- **species (Optional[List[dict]])**：电离物种列表。每个 dict 描述一个组分/通道，按 `fraction` 加权叠加电离率 `W`；规范化阶段会将非零 fraction 归一化。常见键：
  - `name`：物种名
  - `rate`：速率模型
  - `fraction`：该组分体积分数
  - 不同模型需要的参数（当前保留分支）：
    - `ppt_talebpour_i_lut` / `ppt_talebpour_i_full_reference` / `ppt_talebpour_i_legacy`：`Ip_eV, Ip_eV_eff, Zeff, l, m` 等
    - `popruzhenko_atom_i_lut` / `popruzhenko_atom_i_full_reference` / `popruzhenko_atom_i_legacy`：`Ip_eV, Z, l, m` 等
    - `mpa_fact`：`ell, I_mp`
  - 兼容别名：`ppt_talebpour_i -> ppt_talebpour_i_lut`，`ppt_talebpour_i_full -> ppt_talebpour_i_full_reference`，`popruzhenko_atom_i -> popruzhenko_atom_i_lut`，`popruzhenko_atom_i_full -> popruzhenko_atom_i_full_reference`

  已移除 `ppt_e`、`ppt_i`、`adk_e`、`powerlaw`、`mpa` 等旧名称；多光子近似使用 `mpa_fact`，禁用电离使用 `off`。

### 时间处理与积分方式
- **time_mode (TimeMode)**：电离计算时间近似模式（见上）。
- **integrator (Integrator)**：当 `time_mode="full"` 时生效的时间积分器。

### 周期平均/采样与尾部裁剪
- **cycle_avg_samples (int)**：对需要“光学周期平均”的 *_i 类模型，周期内采样点数（越大越准但更慢）。
- **mean_clip_frac (float)**：准稳态 mean 类近似的弱尾裁剪比例（如 1e-3 表示裁剪掉低于峰值 0.1% 的尾部）。

### 电离与等离子体方程参数（与旧实现同名）
- **beta_rec (float)**：复合系数（电子-离子复合项，控制等离子体密度衰减）。
- **sigma_ib (float)**：逆制动辐射（inverse bremsstrahlung）相关吸收截面/等效系数（用于等离子体吸收）。
- **nu_ei_const (Optional[float])**：电子-离子碰撞频率常数（给定则用常数近似；None 可能表示由模型计算）。
- **I_cap (float)**：强度上限裁剪（防止极端强度导致电离率/非线性数值爆炸）。
- **W_cap (float)**：电离率上限裁剪（限制 `W` 最大值以增强稳定性）。

---

## HeatConfig（脉冲间热/折射率慢效应）

- **D_gas (float, m²/s)**：气体热/密度扩散系数（控制慢时扩散平滑速度）。
- **gamma_heat (float, Δn per J/m³)**：热沉积能量密度到折射率变化的耦合系数（负值通常表示加热导致折射率降低）。
- **f_rep (float, Hz)**：重复频率（kHz 条件下用于脉冲间累积/慢时演化）。

---

## RunConfig（运行批次/脉冲数）

- **Npulses (int)**：要模拟的脉冲个数（用于重复脉冲累积效应、多脉冲传播等）。

---

## RamanConfig（拉曼延迟响应与吸收）

- **enabled (bool)**：是否启用拉曼延迟非线性（Raman delayed response）。
- **f_R (float)**：拉曼分量占 Kerr 总响应的比例（空气常用 0.1–0.15）。
- **operator_mode (str)**：Raman 相位算子的实现口径：
  - `"legacy_split"`（默认）：显式双系数 `Δn = n2_elec·I + n_R·I_R`，`f_R` 不参与相位通道。
  - `"split_energy_closed"`：split 相位 + closed-form 吸收口径。
  - `"full_isaacs_eq27"`：完整 Isaacs Eq. (27) 复场算子（显式 opt-in）。
  - `"historical_fr_mixture"`：pre-April 历史 f_R mixture 相位算子（仅相位通道）。
- **model (str)**：拉曼响应核模型：
  - `"rot_sinexp"`：旋转拉曼的“指数衰减 × 正弦”核（带 Heaviside `u(t)`）
  - `"exp"`：单指数核（常用于光纤振动拉曼的简化占位）
- **T2 (float, s)**：去相干时间（旋转拉曼核的指数衰减时间常数）。
- **T_R (Optional[float], s)**：旋转响应特征周期（若给定可确定振荡频率；若为 None 可能改用 `Omega_R`）。
- **Omega_R (Optional[float], rad/s)**：旋转拉曼振荡角频率；若给定则忽略 `T_R`。
- **tau2 (float, s)**：单指数核时间常数（仅 `model="exp"` 使用）。
- **diagnose (bool)**：是否输出/记录拉曼相关诊断量（具体看实现）。
- **absorption_model (str)**：吸收模型名称（此处为 `"closed_form"`）。
- **absorption (bool)**：是否启用与拉曼/分子相关的吸收（或等效吸收项）。
- **omega_R (float)**：拉曼相关频率参数（是否为角频率需对照实现处单位）。
- **Gamma_R (float)**：拉曼线宽/阻尼参数（单位通常为 1/s 或 rad/s，取决于实现）。
- **tau_fwhm (float, s)**：拉曼模型内部使用的脉宽 FWHM 参数（常用于某些闭式近似/参数化）。
- **n_rot_frac (float)**：旋转（rotational）贡献占比（或参与分子数比例）参数。
- **R0_mode (str)**：R0（某个特征半径/响应尺度）取法模式（如 `"mom"` 可能表示由矩/二阶矩估计）。
- **R0_fixed_m (float, m)**：当 `R0_mode` 需要固定值时使用的 R0（固定半径）。

> **historical_fr_mixture（诊断用途）**：`I_R = h_R*I`，`I_nl = (1-f_R)·I + f_R·I_R`，`Δn_NL = n2·I_nl`，`Δφ = k0·Δn_NL·Δz`。相位核由 `T2`/`T_R` 决定（`ω_R=2π/T_R`、`Γ_R=1/T2`，`method=iir`、`iir_sampling=legacy_right_hold`），显式忽略 `omega_R`/`Gamma_R`（二者仍保留给吸收路径）。吸收与 `legacy_split` 对照完全一致（conv_deriv、`n_R`、`omega_R`、`Gamma_R`），不恢复历史 conv_deriv 的 `n0/c0` 写法。该模式仅用于判断“只替换 Raman 相位算子”是否改变 120 fs 成丝起涨，不作为默认生产实现。

---
