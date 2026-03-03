from __future__ import annotations
from .device import xp, to_cpu
from .constants import eps0, c0
import math,numpy as _np

def intensity(E, n0: float):
    """Compute intensity (W/m^2) from complex field E."""
    return 0.5 * eps0 * c0 * n0 * xp.abs(E)**2

def peak_intensity(I):
    """Return scalar peak intensity."""
    return float(xp.max(I))

def pulse_energy(I, dt: float, dx: float, dy: float):
    """
    Energy in Joules from intensity:
      E = ∭ I(x,y,t) dt dx dy
    """
    return float(xp.sum(I) * dt * dx * dy)

def save_npz(path: str, **arrays):
    """Save arrays to .npz on CPU."""

    cpu_arrays = {k: to_cpu(v) for k, v in arrays.items()}
    xp.savez(path, **cpu_arrays)

"""
Itxy: [Nt,Ny,Nx] 强度
算法：
  1) 对时间积分 -> I2D
  2) 去掉极小背景 (rel_floor)
  3) 按能量累积分位 (frac_keep) 做阈值截断
  4) 仅用被保留区域计算二阶矩半径: w = sqrt( 2 * <r^2> )
"""
def second_moment_radius(I3D, x, y, *, dt,
                         frac_keep: float = 0.999,
                         rel_floor: float = 1e-8) -> float:
    """
    由三维强度 I3D[t,y,x] 计算二阶矩等效半径 w。
    约定：理想高斯 I ∝ exp(-2 r^2 / w^2) 时，本函数返回的 w 即该式中的 w。

    参数
    ----
    I3D : [Nt, Ny, Nx]  强度（非负）
    x   : [Nx]          x 坐标（均匀步进）
    y   : [Ny]          y 坐标（均匀步进）
    dt  : float         时间步长
    frac_keep : (0,1]   仅用累计能量前 frac_keep 的主能量区做二阶矩（抗噪）
    rel_floor : float   按峰值的相对阈值，小于该阈值的像素清零（进一步抗噪）
    """
    # ---- 1) 沿 t 积分得到 fluence F2D[y,x] ----
    F = xp.trapz(xp.asarray(I3D), dx=float(dt), axis=0)  # [Ny, Nx]
    F = xp.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F = xp.maximum(F, 0.0)

    # ---- 2) 基础过滤：相对门限 ----
    fmax = float(F.max())
    if not math.isfinite(fmax) or fmax <= 0.0:
        return 0.0
    if rel_floor > 0.0:
        F = xp.where(F >= rel_floor * fmax, F, 0.0)

    Ny, Nx = F.shape
    x = xp.asarray(x)
    y = xp.asarray(y)
    dx = float(x[1] - x[0]) if Nx > 1 else 1.0
    dy = float(y[1] - y[0]) if Ny > 1 else 1.0

    # ---- 3) 主能量区选择（避免 CuPy 的 searchsorted 限制）----
    # 扁平化并按强度降序
    if 0.0 < frac_keep < 1.0:
        flat = F.ravel()
        order = xp.argsort(flat)[::-1]
        flat_sorted = flat[order]
        csum = xp.cumsum(flat_sorted, dtype=xp.float64)
        total = float(csum[-1])
        if total <= 0.0:
            return 0.0
        target = frac_keep * total
        # CuPy 对标量 searchsorted 有兼容问题；用计数实现等价功能
        k = int(xp.count_nonzero(csum < target)) + 1  # 第一个使累计≥target的位置（1基）
        k = max(1, min(k, flat_sorted.size))
        thr = float(flat_sorted[k - 1])
        F = xp.where(F >= thr, F, 0.0)

    # ---- 4) 二阶矩计算（矢量化，避免大 meshgrid）----
    S = float(xp.sum(F, dtype=xp.float64) * dx * dy)
    if S <= 0.0 or not math.isfinite(S):
        return 0.0

    x2 = x * x                  # [Nx]
    y2 = y * y                  # [Ny]
    Fx_sum_y = xp.sum(F, axis=0, dtype=xp.float64)  # [Nx]
    Fy_sum_x = xp.sum(F, axis=1, dtype=xp.float64)  # [Ny]
    mom_x2 = float(xp.sum(x2 * Fx_sum_y) * dx * dy)
    mom_y2 = float(xp.sum(y2 * Fy_sum_x) * dx * dy)

    mean_x2 = mom_x2 / S
    mean_y2 = mom_y2 / S

    # 高斯关系：<x^2>=w^2/4, <y^2>=w^2/4 => w = sqrt(2*(<x^2>+<y^2>))
    w_sq = 2.0 * (mean_x2 + mean_y2)
    if w_sq <= 0.0 or not math.isfinite(w_sq):
        return 0.0
    return math.sqrt(w_sq)


def second_moment_radius_from_2d(F2D, x, y, *,
                                 dx=None, dy=None,
                                 frac_keep: float = 0.999,
                                 rel_floor: float = 1e-8) -> float:
    """
    由 2D 面密度(如 fluence) 计算二阶矩等效半径 w。
    约定：理想高斯 I ∝ exp(-2 r^2 / w^2) 时，本函数返回的 w 即该式中的 w。

    参数
    ----
    F2D : [Ny, Nx]  非负实数数组（例如对 I(t,x,y) 沿 t 积分后的 fluence）
    x   : [Nx]      x 坐标
    y   : [Ny]      y 坐标
    dx,dy : float   网格间距（可省略；若省略则以 x、y 的差分首个间距近似）
    frac_keep : (0,1]  仅使用累计能量占比前 frac_keep 的主能量区进行统计
    rel_floor : 相对峰值的强度阈值，小于阈值的像素置零

    返回
    ----
    w : float  二阶矩等效半径（米）
    """
    F = xp.asarray(F2D)
    # 基本清理：非负、去 NaN/Inf、小于阈值置零
    F = xp.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F = xp.maximum(F, 0.0)
    fmax = float(F.max())
    if not math.isfinite(fmax) or fmax <= 0.0:
        return 0.0
    if rel_floor > 0.0:
        F = xp.where(F >= rel_floor * fmax, F, 0.0)

    Ny, Nx = F.shape
    x = xp.asarray(x)
    y = xp.asarray(y)
    if dx is None:
        dx = float(x[1] - x[0]) if Nx > 1 else 1.0
    if dy is None:
        dy = float(y[1] - y[0]) if Ny > 1 else 1.0

    # 如果启用 frac_keep < 1：仅保留累计能量占比前 frac_keep 的像素
    if 0.0 < frac_keep < 1.0:
        flat = F.ravel()
        # 按强度降序排序，取前 K 使得累计和 >= frac_keep * 总和
        order = xp.argsort(flat)[::-1]
        flat_sorted = flat[order]
        csum = xp.cumsum(flat_sorted, dtype=xp.float64)
        total = float(csum[-1])
        if total <= 0.0:
            return 0.0
        target = frac_keep * total
        # 找到第一个累计超过目标的位置
        k = int(xp.searchsorted(csum, target, side="left"))
        k = max(1, min(k, flat_sorted.size))
        thr = float(flat_sorted[k-1])  # 阈值=第 k 个值
        # 保留 >= 阈值 的像素（注意：可能略多于 frac_keep，对结果影响很小）
        F = xp.where(F >= thr, F, 0.0)

    # 二阶矩：<x^2> 与 <y^2>（对 F 权重）
    # 先构造网格上的 x^2, y^2（矢量外积避免大额中间数组）
    # S = ∬ F dx dy
    S = float(xp.sum(F, dtype=xp.float64) * dx * dy)
    if S <= 0.0 or not math.isfinite(S):
        return 0.0

    x2 = x * x           # [Nx]
    y2 = y * y           # [Ny]
    # ∬ F x^2 dx dy  = (∑_x x^2 ∑_y F) dx dy
    Fx_sum_y = xp.sum(F, axis=0, dtype=xp.float64)             # [Nx]
    Fy_sum_x = xp.sum(F, axis=1, dtype=xp.float64)             # [Ny]
    mom_x2 = float(xp.sum(x2 * Fx_sum_y) * dx * dy)            # ∬ F x^2 dx dy
    mom_y2 = float(xp.sum(y2 * Fy_sum_x) * dx * dy)            # ∬ F y^2 dx dy

    mean_x2 = mom_x2 / S
    mean_y2 = mom_y2 / S
    # 高斯关系：<x^2>=w^2/4, <y^2>=w^2/4 => w = sqrt( 2*(<x^2>+<y^2>) )
    w_sq = 2.0 * (mean_x2 + mean_y2)
    if w_sq <= 0.0 or not math.isfinite(w_sq):
        return 0.0
    return math.sqrt(w_sq)
def parabola_peak(y_minus, y0v, y_plus):
    denom = (y_minus - 2 * y0v + y_plus) + 1e-30
    x_peak = 0.5 * (y_minus - y_plus) / denom
    return y0v - 0.5 * (y_minus - y_plus) * x_peak

def _fwhm_1d_centerline(v, x, i0):
    v = xp.asarray(v)
    vmax = float(v[i0])
    if vmax <= 0.0 or not _np.isfinite(vmax):
        return 0.0
    thr = 0.5 * vmax
    L, R = i0, i0
    n = v.size
    while L > 0 and v[L] >= thr:
        L -= 1
    while R < n - 1 and v[R] >= thr:
        R += 1
    xL = float(x[L]); xR = float(x[R])
    if L < i0 and v[L] < thr:
        w = (thr - float(v[L])) / (float(v[L+1]) - float(v[L]) + 1e-30)
        xL = float(x[L] + w * (x[L+1] - x[L]))
    if R > i0 and v[R] < thr:
        w = (thr - float(v[R])) / (float(v[R-1]) - float(v[R]) + 1e-30)
        xR = float(x[R] + w * (x[R-1] - x[R]))
    return max(0.0, xR - xL)

def _fwhm_diameter_xy_center(r2d, axes, x0i, y0i):
    row = r2d[y0i, :]
    col = r2d[:, x0i]
    fwhm_x = _fwhm_1d_centerline(row, axes.x, x0i)
    fwhm_y = _fwhm_1d_centerline(col, axes.y, y0i)
    return 0.5 * (fwhm_x + fwhm_y)

def _fwhm_time_1d(vt, dt):
    vt = xp.asarray(vt)
    vmax = float(vt.max())
    if vmax <= 0.0 or not _np.isfinite(vmax):
        return 1e-15
    thr = 0.5 * vmax
    idx = xp.where(vt >= thr)[0]
    if idx.size < 2:
        return 1e-15
    tL = float((idx[0]) * dt)
    tR = float((idx[-1]) * dt)
    return max(1e-15, tR - tL)