from __future__ import annotations
from .device import xp
from .constants import c0

# n - 1 = 1e-6 * [ A/(238.0185 - sigma^2) + B/(57.362 - sigma^2) ], sigma=1/λ(μm)
# 近可见红外有效；P,T 做线性密度缩放
def n_air_ciddor_simple(lambda_um, P=101325.0, T=293.15):
    sigma2 = (1.0 / xp.maximum(lambda_um, 1e-9))**2
    d = 1e-6 * (0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2))
    # 密度缩放（~理想气体）：与 (P/T) 比例
    scale = (P / 101325.0) * (273.15 / T)
    return 1.0 + d * scale

def n_of_omega(omega, P=101325.0, T=293.15):
    # ω→λ： λ = 2πc/ω；取 |ω| 并避免 ω=0
    omega = xp.asarray(omega)
    w = xp.maximum(xp.abs(omega), 1e-6)  # rad/s
    lambda_m = 2*xp.pi*c0 / w
    lambda_um = lambda_m * 1e6
    return n_air_ciddor_simple(lambda_um, P=P, T=T)
