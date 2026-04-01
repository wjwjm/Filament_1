from __future__ import annotations
from .device import xp
from dataclasses import dataclass
#make_axes(...) 生成 x,y,t 与 kx,ky,KX,KY,kperp2 以及 dx,dy,dt。

#所有算子依赖的频域量在此集中计算，保持一致性。
@dataclass
class Axes:
    x: object; y: object; t: object
    kx: object; ky: object; KX: object; KY: object; kperp2: object
    dx: float; dy: float; dt: float
    Omega: object              # <== 新增：时间角频率轴

def make_axes(Nx: int, Ny: int, Nt: int, Lx: float, Ly: float, Twin: float):
    dx, dy = Lx / Nx, Ly / Ny
    dt = Twin / Nt
    x = (xp.arange(Nx) - Nx // 2) * dx
    y = (xp.arange(Ny) - Ny // 2) * dy
    t = (xp.arange(Nt) - Nt // 2) * dt
    kx = 2 * xp.pi * xp.fft.fftfreq(Nx, d=dx)
    ky = 2 * xp.pi * xp.fft.fftfreq(Ny, d=dy)
    KX, KY = xp.meshgrid(kx, ky, indexing='xy')
    kperp2 = KX**2 + KY**2
    Omega = 2 * xp.pi * xp.fft.fftfreq(Nt, d=dt)  # <== 新增
    return Axes(
        x=x, y=y, t=t, kx=kx, ky=ky, KX=KX, KY=KY,
        kperp2=kperp2, dx=dx, dy=dy, dt=dt, Omega=Omega
    )


def make_edge_apodizer(x, y, frac=0.05):
    # frac: 边缘过渡宽度占总宽度的比例
    def tukey(n, alpha):
        # 简化版 1D tukey
        i = xp.arange(n)
        w = xp.ones(n, dtype=xp.float32)
        m = int(alpha*(n-1)/2)
        if m>0:
            w[:m] = 0.5 * (1 + xp.cos(xp.pi*(2*i[:m]/(alpha*(n-1)) - 1)))
            w[-m:] = w[:m][::-1]
        return w
    wx = tukey(x.size, frac)
    wy = tukey(y.size, frac)
    W = wy[:, None] * wx[None, :]
    return W