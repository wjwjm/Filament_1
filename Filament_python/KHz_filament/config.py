from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Literal

@dataclass
class GridConfig:
    Nx: int = 384
    Ny: int = 384
    Nt: int = 128
    Lx: float = 6.0e-3      # m
    Ly: float = 6.0e-3      # m
    Twin: float = 640e-15   # s (time window)

@dataclass
class BeamConfig:
    lam0: float = 800e-9    # m
    n0: float = 1.00027
    w0: float = 1.979e-3      # m (1/e field radius)  束腰半径
    tau_fwhm: float = 40e-15 # s
    E0_peak: float = 0.0            # 仍然支持直接给 E0   电场幅值，不是峰值强度
    energy_J: Optional[float] = 0.68e-3  # 新增：也可只给单脉冲能量
    P0_peak: Optional[float] = None  # 新增：峰值功率（W，脉冲中心时刻的横截面积分功率）
    focal_length: float = 0.95     #透镜焦距 m
    n2_air: float = 7.8e-24
@dataclass
class PropagationConfig:
    z_max: float = 1.2      # m
    dz: float = 1e-3      # m
    strang: bool = True     # use Strang splitting if True
#---- # ==== B 档：UPPE 相关 ====
    linear_model: str = "uppe"  # "uppe" | "paraxial" | "bk_nee"
    full_linear_factorize: bool = False  # True: ω-切片逐片做2D FFT，省内存（更稳）
    use_self_steepening: bool = True  # 建议开启
    #若显存/内存充裕且想更快，把 full_linear_factorize=False（会创建 [Nt,Ny,Nx] 级别的 3D相位，内存压力大）。
    # 空气色散模型参数（简化 Ciddor）
    air_model: str = "ciddor_simple"
    air_T: float = 293.15  # K
    air_P: float = 101325.0  # Pa
    air_CO2: float = 400e-6  # 体积分数（占位，简化模型里不显式使用）
    # Brabec–Krausz NEE 线性项参数（linear_model="bk_nee" 时使用）
    nee_beta2: float = 0.2e-28         # s^2/m, GVD 系数 k''
    nee_denom_floor: float = 1e-4  # 防止 1+Omega/omega0 在 -omega0 邻域奇异
    # === 进度输出 ===
    progress_every_z: int = 100    # 每多少个 z 步打印一次（0=不打印）
    show_eta: bool = True          # 打印 ETA 估计
    # ---- new: adaptive substep ----
    auto_substep: bool = True
    dz_min: float = 2.5e-5  # 25 μm
    grow_factor: float = 1.5  # 接受后尝试放大
    shrink_factor: float = 0.5  # 条件不满足时缩小
    max_linear_phase: float = 0.8  # |Δφ_linear|max ≤ 0.8 rad
    max_alpha_dz: float = 0.2  # α_max·dz ≤ 0.2
    max_kerr_phase: float = 0.3  # |Δφ_Kerr|max ≤ 0.3 rad（保守）
    imax_growth_limit: float = 0.5
    safety_mode: str = "on"
    energy_probe_every: float = 25
    energy_probe_tol: float = 0.03
    diag_extra: bool= True #是否导出大量信息


    focus_window_step: bool= True
    focus_center_m: float =0.95
    focus_halfwidth_m: float =0.10
    dz_focus: float =0.1e-3
    limit_focus_window: bool= True
    window_halfwidth_m: float =0.3
TimeMode = Literal["full", "qs_peak", "qs_mean", "qs_mean_esq"]
Integrator = Literal["rk4", "euler"]
@dataclass
class IonizationConfig:
    """
    以 species 为核心的电离配置（简化版）。
    - species: 列表，每个元素是 dict，常用键：
        name: str
        rate: "ppt_talebpour_i_legacy" | "ppt_talebpour_i_full" |
              "popruzhenko_atom_i_full" | "mpa_fact" | "off"
              （兼容别名：ppt_talebpour_i -> *_full，popruzhenko_atom_i -> *_full）
        fraction: float   # 该组分体积分数；会在 __post_init__ 中自动归一化
        # Talebpour molecule: Ip_eV, Ip_eV_eff(optional), Zeff, l, m
        # Popruzhenko atom: Ip_eV, Z, l, m, max_terms(optional), sum_rel_tol(optional)
        # mpa_fact: ell, I_mp
        # 可选：W_cap（覆盖全局 W_cap）
    其他字段与电离模块的新实现一一对应。
    """
    # --- 必填：物种列表（按 fraction 线性叠加） ---
    species: Optional[List[dict]] = None  # 若提供，则按 fraction 线性叠加各通道的 W
    # --- LUT 运行时加速与缓存 ---
    rate_table: Optional[dict] = None

    # --- 时间近似与全时域积分器（只作为开关传递到电离模块） ---
    time_mode: TimeMode = "full"        # "full" | "qs_peak" | "qs_mean" | "qs_mean_esq"
    integrator: Integrator = "rk4"      # 仅在 time_mode="full" 时生效

    # 周期平均速率相位采样数（对 *_i 模型有效）
    cycle_avg_samples: int = 64

    # 准稳态 mean 的弱尾裁剪比例（0~1），如 1e-3 表示剪掉 <0.1% 峰值的尾部
    mean_clip_frac: float = 1e-3

    # 物理/数值参数（与原来保持同名）
    beta_rec: float = 0.0
    sigma_ib: float = 0.0
    nu_ei_const: Optional[float] = None
    I_cap: float = 1e19
    W_cap: float = 1e16

    # def __post_init__(self):
    #     # 将 fraction 归一化到和为 1（仅当总和>0时）
    #     if not self.species:
    #         return
    #     total = 0.0
    #     for sp in self.species:
    #         frac = float(sp.get("fraction", 1.0))
    #         if frac < 0.0:
    #             frac = 0.0
    #         sp["fraction"] = frac
    #         total += frac
    #     if total > 0.0:
    #         for sp in self.species:
    #             sp["fraction"] = float(sp["fraction"] / total)
@dataclass
class HeatConfig:
    D_gas: float = 2.0e-5        # m^2/s
    gamma_heat: float = -1.0e-23 # Δn per J/m^3
    f_rep: float = 1.0e3         # Hz (1 kHz default)

@dataclass
class RunConfig:
    Npulses: int = 1

@dataclass
class RamanConfig:
    enabled: bool = True     # 开关：是否启用拉曼延迟
    f_R: float = 0.15         # 兼容字段；rot_sinexp 下相位/吸收不再用该系数缩放
    n_R: float = 2.3e-23      # rotational Raman coefficient (m^2/W)
    model: str = "rot_sinexp" # "rot_sinexp" | "exp"
    # —— 旋转拉曼（空气）简化核：e^{-t/T2} * sin(Ω_R t) * u(t)
    T2: float = 80e-12        # 去相干时间（ps 量级）
    T_R: Optional[float] = 8.4e-12  # 旋转响应特征周期（~8.4 ps；可 None）
    Omega_R: Optional[float] = None # 若给定则忽略 T_R；单位 rad/s
    # —— 单指数核：e^{-t/τ2} / τ2 * u(t)，常用于光纤振动拉曼的“占位式”
    tau2: float = 32e-15      # 仅当 model="exp" 使用
    diagnose: bool = False
    absorption_model: str = "closed_form"
    absorption: bool = True
    omega_R: float =7.5e11
    Gamma_R: float =1.25e10
    tau_fwhm: float =120e-15
    n_rot_frac: float =0.99
    R0_mode: str ="mom"
    R0_fixed_m: float =2.0e-4
