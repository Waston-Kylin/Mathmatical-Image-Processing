import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import gaussian_filter
from Reinitial2D import Reinitial2D

def fspecial(shape, sigma):
    """生成高斯模糊核"""
    s = shape[0]
    center = (s + 1.0) / 2.0
    y, x = np.mgrid[1:s+1, 1:s+1]
    h = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma * sigma))
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def psf2otf(psf, shape):
    """空间核转频域传递函数"""
    psf_pad = np.zeros(shape)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    psf_pad = np.roll(psf_pad, -(psf.shape[0] // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(psf.shape[1] // 2), axis=1)
    return fft2(psf_pad)

def neumann_shifts(u):
    """返回四邻域及四个对角邻域平移 (Neumann 零通量边界)"""
    u_xp = np.vstack([u[1:, :], u[-1:, :]])    # i+1, j
    u_xm = np.vstack([u[:1, :], u[:-1, :]])    # i-1, j
    u_yp = np.hstack([u[:, 1:], u[:, -1:]])    # i, j+1
    u_ym = np.hstack([u[:, :1], u[:, :-1]])    # i, j-1

    u_xp_yp = np.hstack([u_xp[:, 1:], u_xp[:, -1:]])
    u_xp_ym = np.hstack([u_xp[:, :1], u_xp[:, :-1]])
    u_xm_yp = np.hstack([u_xm[:, 1:], u_xm[:, -1:]])
    u_xm_ym = np.hstack([u_xm[:, :1], u_xm[:, :-1]])
    return u_xp, u_xm, u_yp, u_ym, u_xp_yp, u_xp_ym, u_xm_yp, u_xm_ym

def compute_curvature(u, eps=1e-8):
    """
    水平集函数曲率 K = div(nabla u / |nabla u|)
    半步中心差分散度格式保证数值稳定性
    """
    u_xp, u_xm, u_yp, u_ym, u_xp_yp, u_xp_ym, u_xm_yp, u_xm_ym = neumann_shifts(u)

    # x 方向前向差分与 y 方向中心差分
    dx_p = u_xp - u
    dx_m = u - u_xm
    dy_c = 0.5 * (u_yp - u_ym)
    dy_xm = 0.5 * (u_xm_yp - u_xm_ym)

    # y 方向前向差分与 x 方向中心差分
    dy_p = u_yp - u
    dy_m = u - u_ym
    dx_c = 0.5 * (u_xp - u_xm)
    dx_ym = 0.5 * (u_xp_ym - u_xm_ym)

    n_xp = dx_p / np.sqrt(dx_p**2 + dy_c**2 + eps)
    n_xm = dx_m / np.sqrt(dx_m**2 + dy_xm**2 + eps)
    n_yp = dy_p / np.sqrt(dx_c**2 + dy_p**2 + eps)
    n_ym = dy_m / np.sqrt(dx_ym**2 + dy_m**2 + eps)

    curvature = (n_xp - n_xm) + (n_yp - n_ym)
    return curvature

def init_level_set(shape, init_type="circle", margin=15):
    """
    初始水平集函数 phi_0 (约定: 轮廓内部 phi > 0 或 < 0 均可，
    统一约定: 初始轮廓外部为负/正与课件对齐:
    对 GAC 和 CV，设轮廓内部 phi > 0，外部 phi < 0，零水平集为边界)
    """
    M, N = shape
    Y, X = np.mgrid[0:M, 0:N]

    if init_type == "box":
        # 距离四周边界 margin 像素的矩形框 (内正外负)
        d_top = Y - margin
        d_bot = (M - 1 - margin) - Y
        d_left = X - margin
        d_right = (N - 1 - margin) - X
        phi0 = np.minimum(np.minimum(d_top, d_bot), np.minimum(d_left, d_right)).astype(np.float64)
    elif init_type == "multi_circles":
        # 多圆网格初始化 (非凸 CV 捕获孔洞)
        phi0 = (-5.0) * np.ones((M, N), dtype=np.float64)
        r = min(M, N) / 10.0
        for cy in np.linspace(M * 0.2, M * 0.8, 3):
            for cx in np.linspace(N * 0.2, N * 0.8, 3):
                dist = r - np.sqrt((Y - cy)**2 + (X - cx)**2)
                phi0 = np.maximum(phi0, dist)
    elif init_type == "corner_circle":
        # 偏左上角孤立小圆 (半径 35，圆心位于 (60, 60))
        cy, cx = M * 0.25, N * 0.25
        r = min(M, N) * 0.15
        phi0 = r - np.sqrt((Y - cy)**2 + (X - cx)**2)
    else:
        # 默认大圆包络 (内正外负)
        cy, cx = M / 2.0, N / 2.0
        r = min(M, N) * 0.42
        phi0 = r - np.sqrt((Y - cy)**2 + (X - cx)**2)

    return Reinitial2D(phi0, steps=10)

# GAC
def solve_gac(img, phi0=None, max_iter=400, dt=0.25, alpha=-1.0, edge_k=0.05, sigma_g=1.5, reinit_int=5):
    """
    Geodesic Active Contours (GAC) 水平集演化:
    partial v / partial t = g |nabla v| div(nabla v / |nabla v|)
                            + alpha g |nabla v| + nabla g cdot nabla v
    params:
        alpha: 气球力系数 (初始轮廓包裹物体向内收缩时，若内部 phi > 0，则取 alpha < 0 使正区收缩)
        edge_k: 边缘停止函数 g(|∇I|) = 1 / (1 + (|∇I|/edge_k)^2) 的梯度灵敏度阈值
        sigma_g: 计算图像梯度前的高斯预平滑标准差
    """
    u = phi0.copy() if phi0 is not None else init_level_set(img.shape, "box", margin=12)

    # 计算平滑图像的边缘停止函数 g(|∇I|) 及梯度 gx, gy
    img_smooth = gaussian_filter(img, sigma=sigma_g)
    ix_p, ix_m, iy_p, iy_m, _, _, _, _ = neumann_shifts(img_smooth)
    Ix = 0.5 * (ix_p - ix_m)
    Iy = 0.5 * (iy_p - iy_m)
    grad_I_sq = Ix**2 + Iy**2

    g = 1.0 / (1.0 + grad_I_sq / (edge_k**2 + 1e-10))
    gx_p, gx_m, gy_p, gy_m, _, _, _, _ = neumann_shifts(g)
    gx = 0.5 * (gx_p - gx_m)
    gy = 0.5 * (gy_p - gy_m)

    c_speed = alpha * g  # 气球力标量速度场

    for step in range(1, max_iter + 1):
        u_xp, u_xm, u_yp, u_ym, _, _, _, _ = neumann_shifts(u)

        # 单侧前向与后向差分 \delta^+, \delta^-
        dx_plus = u_xp - u
        dx_minus = u - u_xm
        dy_plus = u_yp - u
        dy_minus = u - u_ym

        # 中心差分 \delta_x u, \delta_y u (曲率项模长)
        dx_cen = 0.5 * (u_xp - u_xm)
        dy_cen = 0.5 * (u_yp - u_ym)
        grad_cen_mag = np.sqrt(dx_cen**2 + dy_cen**2 + 1e-10)

        # 曲率扩散项: g * K * |\nabla u|
        K = compute_curvature(u)
        term_curv = g * K * grad_cen_mag

        # 标量速度迎风项 Non-oscillatory upwind
        nabla_plus = np.sqrt(
            np.maximum(dx_minus, 0.0)**2 + np.minimum(dx_plus, 0.0)**2 +
            np.maximum(dy_minus, 0.0)**2 + np.minimum(dy_plus, 0.0)**2
        )
        nabla_minus = np.sqrt(
            np.minimum(dx_minus, 0.0)**2 + np.maximum(dx_plus, 0.0)**2 +
            np.minimum(dy_minus, 0.0)**2 + np.maximum(dy_plus, 0.0)**2
        )
        term_balloon = np.maximum(c_speed, 0.0) * nabla_plus + np.minimum(c_speed, 0.0) * nabla_minus

        # 对流项 nabla g \cdot nabla u (One-side upwind)
        term_adv = (
            np.maximum(gx, 0.0) * dx_minus + np.minimum(gx, 0.0) * dx_plus +
            np.maximum(gy, 0.0) * dy_minus + np.minimum(gy, 0.0) * dy_plus
        )

        # 显式时间步更新
        u = u + dt * (term_curv + term_balloon + term_adv)

        # 每隔 reinit_int 步执行水平集重初始化
        if reinit_int > 0 and (step % reinit_int == 0):
            u = Reinitial2D(u, steps=8)

    seg_mask = (u >= 0).astype(np.float64)
    return u, seg_mask


# Chan-Vese 模型
def solve_cv(img, phi0=None, max_iter=250, dt=0.5, mu=0.2, nu=0.0, lam1=1.0, lam2=1.0, eps_h=1.0, reinit_int=5):
    """
    Original Chan-Vese (CV) 模型:
    partial phi / partial t = delta_varepsilon(phi) [ mu div(nabla phi / |nabla phi|)
                            - nu - lambda_1 (u_0 - c_1)^2 + lambda_2 (u_0 - c_2)^2 ]
    """
    phi = phi0.copy() if phi0 is not None else init_level_set(img.shape, "multi_circles")

    for step in range(1, max_iter + 1):
        # 全局支集正则化 Heaviside H_{2, varepsilon} 与 Dirac delta_varepsilon
        H_phi = 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / eps_h))
        delta_phi = (1.0 / np.pi) * (eps_h / (eps_h**2 + phi**2))

        # 更新内部均值 c1 与外部均值 c2
        c1 = np.sum(img * H_phi) / (np.sum(H_phi) + 1e-10)
        c2 = np.sum(img * (1.0 - H_phi)) / (np.sum(1.0 - H_phi) + 1e-10)

        # 计算曲率项与区域保真项
        K = compute_curvature(phi)
        fidelity = -nu - lam1 * ((img - c1)**2) + lam2 * ((img - c2)**2)

        # 演化更新水平集函数
        phi = phi + dt * delta_phi * (mu * K + fidelity)

        # 周期重初始化
        if reinit_int > 0 and (step % reinit_int == 0):
            phi = Reinitial2D(phi, steps=5)

    seg_mask = (phi >= 0).astype(np.float64)
    return phi, seg_mask


# 凸化 CV 模型 (Convexified CV) 
def solve_convex_cv(img, max_iter=150, mu=2.0, lam_admm=1.0, alpha_g=10.0, use_edge_g=True):
    """
    Convexified Binary Mumford-Shah (Convex CV) 模型:
    min_{0 <= u <= 1} |nabla u|_g + mu <u, r>,  r = (f - c_1)^2 - (f - c_2)^2
    用 Split Bregman / ADMM 交替优化 u 与 vec{d} = nabla u
    """
    H, W = img.shape

    # 权重函数 g(x) = 1 或 1 / (1 + alpha * |\nabla f|^2)
    if use_edge_g and alpha_g > 0:
        img_s = gaussian_filter(img, sigma=1.0)
        ix_p, ix_m, iy_p, iy_m, _, _, _, _ = neumann_shifts(img_s)
        grad_sq = (0.5 * (ix_p - ix_m))**2 + (0.5 * (iy_p - iy_m))**2
        g_weight = 1.0 / (1.0 + alpha_g * grad_sq)
    else:
        g_weight = np.ones_like(img)

    # 频域差分算子用于 FFT 快速求解 u-子问题: (I - \Delta) u = rhs
    dx_k = np.zeros((H, W)); dx_k[0, 0] = -1.0; dx_k[0, 1] = 1.0
    dy_k = np.zeros((H, W)); dy_k[0, 0] = -1.0; dy_k[1, 0] = 1.0
    otf_Dx = fft2(dx_k)
    otf_Dy = fft2(dy_k)
    lap_denom = 1.0 + lam_admm * (np.abs(otf_Dx)**2 + np.abs(otf_Dy)**2)  # (I - λΔ)

    # 初始化 u \in [0, 1] (图像均值阈值初始化)
    u = (img >= np.mean(img)).astype(np.float64)
    dx = np.zeros((H, W), dtype=np.float64)
    dy = np.zeros((H, W), dtype=np.float64)
    bx = np.zeros((H, W), dtype=np.float64)
    by = np.zeros((H, W), dtype=np.float64)

    for _ in range(max_iter):
        # 更新区域均值常数 c1, c2（连续 u 软加权，符合凸化模型的交替优化）
        sum_u = np.sum(u) + 1e-10
        sum_1mu = np.sum(1.0 - u) + 1e-10
        c1 = np.sum(img * u) / sum_u
        c2 = np.sum(img * (1.0 - u)) / sum_1mu

        r = (img - c1)**2 - (img - c2)**2

        # 更新 u-子问题: 求解 \lambda (-\Delta u) + \mu r = \lambda div(d - b) 并投影到 [0, 1]
        div_db = lam_admm * (np.conj(otf_Dx) * fft2(dx - bx) + np.conj(otf_Dy) * fft2(dy - by))
        rhs = u - mu * r
        u_unconstrained = np.real(ifft2((fft2(rhs) + div_db) / lap_denom))
        u = np.clip(u_unconstrained, 0.0, 1.0)

        # 更新 d-子问题 (各向同性软阈值收缩 Shrinkage)
        nabla_x = np.roll(u, -1, axis=1) - u
        nabla_y = np.roll(u, -1, axis=0) - u
        vx = nabla_x + bx
        vy = nabla_y + by
        V = np.sqrt(vx**2 + vy**2 + 1e-12)

        thresh = g_weight / lam_admm
        shrink = np.maximum(V - thresh, 0.0) / V
        dx = shrink * vx
        dy = shrink * vy

        # 更新 Bregman 对偶变量 b
        bx = bx + nabla_x - dx
        by = by + nabla_y - dy

    # Co-area 定理，取阈值 u >= 0.5 得全局最优分割集，令水平集函数为 u - 0.5
    phi_equiv = u - 0.5
    seg_mask = (u >= 0.5).astype(np.float64)
    return phi_equiv, seg_mask


# 三联结果可视化
def save_segmentation_result(u_clean, f_obs, phi, seg_mask, output_image, title_model="Model"):
    """
    三联对比图:
      左图: 退化/输入观测图像 + 初始/最终分割轮廓红线 (Zero-level set)
      中图: 分割二值掩码 (Binary Mask)
      右图: 分片常数重建图 c1 * Mask + c2 * (1 - Mask)
    """
    c1 = np.sum(f_obs * seg_mask) / (np.sum(seg_mask) + 1e-10)
    c2 = np.sum(f_obs * (1.0 - seg_mask)) / (np.sum(1.0 - seg_mask) + 1e-10)
    recon = c1 * seg_mask + c2 * (1.0 - seg_mask)

    fg_ratio = np.mean(seg_mask) * 100.0
    log_text = (
        f"模型: [{title_model}]\n"
        f"内部均值 c1 = {c1:.4f}, 外部均值 c2 = {c2:.4f}\n"
        f"分割前景面积占比 = {fg_ratio:.2f}%"
    )
    print(log_text + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # 观测图像 + 红色分割轮廓线
    axes[0].imshow(f_obs, cmap='gray', vmin=0, vmax=1)
    axes[0].contour(phi, levels=[0.0], colors='red', linewidths=2.0)
    axes[0].set_title("Observed Image + Contour", fontsize=13)

    # 二值分割区域掩码
    axes[1].imshow(seg_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"{title_model} Binary Mask", fontsize=13)

    # Piecewise Constant Approximation
    axes[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Piecewise Constant (c1={c1:.2f}, c2={c2:.2f})", fontsize=13)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if output_image:
        os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else '.', exist_ok=True)
        plt.savefig(output_image, dpi=300, bbox_inches='tight')

    return fig, log_text
