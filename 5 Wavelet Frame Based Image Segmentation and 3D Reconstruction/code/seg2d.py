import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter, convolve1d

# 2D B-样条小波紧框架变换
def get_framelet_filters(frame=1):
    """生成 1D 紧框架滤波器组 D & R"""
    if frame == 0:  # Haar
        h0 = np.array([0.5, 0.5])
        h1 = np.array([0.5, -0.5])
        filters = [h0, h1]
    elif frame == 1: # Piecewise Linear
        h0 = np.array([0.25, 0.5, 0.25])
        h1 = np.array([np.sqrt(2.0) / 4.0, 0.0, -np.sqrt(2.0) / 4.0])
        h2 = np.array([-0.25, 0.5, -0.25])
        filters = [h0, h1, h2]
    elif frame == 3: # Piecewise Cubic
        h0 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        h1 = np.array([1.0, 2.0, 0.0, -2.0, -1.0]) / 8.0
        h2 = np.array([-1.0, 0.0, 2.0, 0.0, -1.0]) * np.sqrt(6.0) / 16.0
        h3 = np.array([-1.0, 2.0, 0.0, -2.0, 1.0]) / 8.0
        h4 = np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / 16.0
        filters = [h0, h1, h2, h3, h4]
    else:
        raise ValueError("frame must be 0, 1, or 3")
    return filters

def frame_dec_2d(u, frame=1):
    """单层 2D 紧框架分解 W(u)，返回 (nD, nD, H, W) 系数张量"""
    filters = get_framelet_filters(frame)
    nD = len(filters)
    H, W = u.shape
    coeffs = np.zeros((nD, nD, H, W), dtype=np.float64)
    for i, fi in enumerate(filters):
        temp = convolve1d(u, fi, axis=0, mode='reflect')
        for j, fj in enumerate(filters):
            coeffs[i, j] = convolve1d(temp, fj, axis=1, mode='reflect')
    return coeffs

def frame_rec_2d(coeffs, frame=1):
    """单层 2D 紧框架重构 W^T(d)，W^T W = I"""
    filters = get_framelet_filters(frame)
    nD = len(filters)
    _, _, H, W = coeffs.shape
    u_rec = np.zeros((H, W), dtype=np.float64)
    for i, fi in enumerate(filters):
        temp = np.zeros((H, W), dtype=np.float64)
        for j, fj in enumerate(filters):
            # 伴随算子对应翻转卷积核
            temp += convolve1d(coeffs[i, j], fj[::-1], axis=1, mode='reflect')
        u_rec += convolve1d(temp, fi[::-1], axis=0, mode='reflect')
    return u_rec

# Algorithm 2: 小波框架图像分割
def solve_frame_segmentation(f, alpha=0.01, sigma=None, mu=1.0, gamma=0.5,
                             frame=1, max_iter=300, tol=1e-4):
    if sigma is None:
        sigma = 50.0

    Wf = frame_dec_2d(f, frame=frame)
    nD = Wf.shape[0]

    # 高频子带梯度能量
    grad_energy = np.zeros_like(f)
    for i in range(nD):
        for j in range(nD):
            if i != 0 or j != 0:
                grad_energy += Wf[i, j] ** 2

    # 空间自适应阈值场 lambda(x)
    lam_map = alpha / (1.0 + sigma * grad_energy)
    tau_map = lam_map / mu

    # 初始化 (图像平均灰度轻微偏移)
    f_mean = float(np.mean(f))
    c1 = min(f_mean + 0.15, 1.0)
    c2 = max(f_mean - 0.15, 0.0)
    
    # 初始 u 设为粗略两相指示
    u = (f > f_mean).astype(np.float64)
    Wu0 = frame_dec_2d(u, frame=frame)
    d = Wu0.copy()
    v = np.zeros_like(Wu0)
    norm_f = np.linalg.norm(f) + 1e-12

    for k in range(1, max_iter + 1):
        # 更新 u
        WT_dv = frame_rec_2d(d - v, frame=frame)
        data_term = (f - c1) ** 2 - (f - c2) ** 2
        # 当 f 接近 c1 时 data_term < 0，u 趋向 1
        u_half = WT_dv - data_term / mu
        u = np.clip(u_half, 0.0, 1.0)

        # 更新各向同性软阈值 d
        Wu = frame_dec_2d(u, frame=frame)
        z = Wu + v
        d = np.zeros_like(z)
        d[0, 0] = z[0, 0] # 低频不收缩

        R = np.zeros_like(f)
        for i in range(nD):
            for j in range(nD):
                if i != 0 or j != 0:
                    R += z[i, j] ** 2
        R = np.sqrt(R + 1e-20)
        shrink = np.maximum(R - tau_map, 0.0) / R

        for i in range(nD):
            for j in range(nD):
                if i != 0 or j != 0:
                    d[i, j] = shrink * z[i, j]

        # 更新 c1, c2
        mask = u > gamma
        if np.any(mask):
            c1 = float(np.mean(f[mask]))
        if np.any(~mask):
            c2 = float(np.mean(f[~mask]))

        # 更新 v
        diff = Wu - d
        v = v + diff

        # 停机准则: 至少 25 步
        rel_err = np.linalg.norm(diff) / norm_f
        if rel_err < tol and k >= 25:
            break

    seg_mask = (u > gamma).astype(np.float64)
    return u, seg_mask, c1, c2, k

# GAC 与 Chan-Vese
def reinitial_2d(phi, steps=5):
    """Godunov 迎风格式水平集重初始化"""
    phi = phi.copy()
    M, N = phi.shape
    h = 1.0 / (max(N, M) - 1.0)
    dt = 0.5 * h
    dx, dy = 1.0 / (M - 1.0), 1.0 / (N - 1.0)
    for _ in range(steps):
        xp = (np.vstack([phi[1:, :], phi[-1:, :]]) - phi) / dx
        xn = (phi - np.vstack([phi[:1, :], phi[:-1, :]])) / dx
        yp = (np.hstack([phi[:, 1:], phi[:, -1:]]) - phi) / dy
        yn = (phi - np.hstack([phi[:, :1], phi[:, :-1]])) / dy
        phi_p = (phi >= 0).astype(np.float64)
        phi_n = 1.0 - phi_p
        Gp = np.sqrt(np.maximum(np.maximum(xn, 0.0)**2, np.minimum(xp, 0.0)**2) +
                     np.maximum(np.maximum(yn, 0.0)**2, np.minimum(yp, 0.0)**2))
        Gn = np.sqrt(np.maximum(np.minimum(xn, 0.0)**2, np.maximum(xp, 0.0)**2) +
                     np.maximum(np.minimum(yn, 0.0)**2, np.maximum(yp, 0.0)**2))
        phi -= (dt * phi_p * (Gp - 1.0) * phi / np.sqrt(phi**2 + Gp**2 * dx * dy + 1e-9) +
                dt * phi_n * (Gn - 1.0) * phi / np.sqrt(phi**2 + Gn**2 * dx * dy + 1e-9))
    return phi

def init_level_set(shape, init_type="box"):
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    if init_type == "box":
        margin = int(0.1 * min(H, W))
        phi = -np.ones((H, W), dtype=np.float64)
        phi[margin:H-margin, margin:W-margin] = 1.0
    else:  # circle
        cy, cx, r = H / 2.0, W / 2.0, 0.35 * min(H, W)
        phi = r - np.sqrt((x - cx)**2 + (y - cy)**2)
    return reinitial_2d(phi, steps=10)

def compute_curvature(phi):
    px = 0.5 * (np.pad(phi[:, 1:], ((0,0),(0,1)), 'edge') - np.pad(phi[:, :-1], ((0,0),(1,0)), 'edge'))
    py = 0.5 * (np.pad(phi[1:, :], ((0,1),(0,0)), 'edge') - np.pad(phi[:-1, :], ((1,0),(0,0)), 'edge'))
    norm = np.sqrt(px**2 + py**2 + 1e-10)
    nx, ny = px / norm, py / norm
    nxx = 0.5 * (np.pad(nx[:, 1:], ((0,0),(0,1)), 'edge') - np.pad(nx[:, :-1], ((0,0),(1,0)), 'edge'))
    nyy = 0.5 * (np.pad(ny[1:, :], ((0,1),(0,0)), 'edge') - np.pad(ny[:-1, :], ((1,0),(0,0)), 'edge'))
    return nxx + nyy, norm

def solve_gac(f, init_type="box", max_iter=800, dt=0.4, alpha=-0.2, edge_k=0.05):
    """主动轮廓模型 GAC"""
    f_s = gaussian_filter(f, sigma=1.0)
    gx = 0.5 * (np.pad(f_s[:, 1:], ((0,0),(0,1)), 'edge') - np.pad(f_s[:, :-1], ((0,0),(1,0)), 'edge'))
    gy = 0.5 * (np.pad(f_s[1:, :], ((0,1),(0,0)), 'edge') - np.pad(f_s[:-1, :], ((1,0),(0,0)), 'edge'))
    g = 1.0 / (1.0 + (gx**2 + gy**2) / (edge_k**2))
    dgx = 0.5 * (np.pad(g[:, 1:], ((0,0),(0,1)), 'edge') - np.pad(g[:, :-1], ((0,0),(1,0)), 'edge'))
    dgy = 0.5 * (np.pad(g[1:, :], ((0,1),(0,0)), 'edge') - np.pad(g[:-1, :], ((1,0),(0,0)), 'edge'))

    phi = init_level_set(f.shape, init_type)
    for k in range(1, max_iter + 1):
        kap, grad_norm = compute_curvature(phi)
        dxp = np.pad(phi[:, 1:], ((0,0),(0,1)), 'edge') - phi
        dxm = phi - np.pad(phi[:, :-1], ((0,0),(1,0)), 'edge')
        dyp = np.pad(phi[1:, :], ((0,1),(0,0)), 'edge') - phi
        dym = phi - np.pad(phi[:-1, :], ((1,0),(0,0)), 'edge')

        if alpha >= 0:
            upwind_grad = np.sqrt(np.maximum(dxm, 0)**2 + np.minimum(dxp, 0)**2 +
                                  np.maximum(dym, 0)**2 + np.minimum(dyp, 0)**2)
        else:
            upwind_grad = np.sqrt(np.minimum(dxm, 0)**2 + np.maximum(dxp, 0)**2 +
                                  np.minimum(dym, 0)**2 + np.maximum(dyp, 0)**2)

        adv = (np.maximum(dgx, 0)*dxm + np.minimum(dgx, 0)*dxp +
               np.maximum(dgy, 0)*dym + np.minimum(dgy, 0)*dyp)
        phi += dt * (g * kap * grad_norm + alpha * g * upwind_grad + adv)
        if k % 5 == 0:
            phi = reinitial_2d(phi, steps=5)

    mask = (phi >= 0).astype(np.float64)
    c1 = float(np.mean(f[mask > 0.5])) if np.any(mask > 0.5) else 1.0
    c2 = float(np.mean(f[mask <= 0.5])) if np.any(mask <= 0.5) else 0.0
    return phi, mask, c1, c2

def solve_chan_vese(f, init_type="box", max_iter=1500, dt=0.4, mu=0.2):
    """Chan-Vese 模型"""
    phi = init_level_set(f.shape, init_type)
    eps = 1.0
    for k in range(1, max_iter + 1):
        H_phi = 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / eps))
        delta_phi = (1.0 / np.pi) * (eps / (eps**2 + phi**2))
        c1 = np.sum(f * H_phi) / (np.sum(H_phi) + 1e-10)
        c2 = np.sum(f * (1.0 - H_phi)) / (np.sum(1.0 - H_phi) + 1e-10)
        kap, _ = compute_curvature(phi)
        phi += dt * delta_phi * (mu * kap - (f - c1)**2 + (f - c2)**2)
        if k % 5 == 0:
            phi = reinitial_2d(phi, steps=5)

    mask = (phi >= 0).astype(np.float64)
    return phi, mask, float(c1), float(c2)

# 测试图像与对比绘图
def load_or_generate_test_image(name, image_dir="image"):
    """image/ 下存在测试图则直接读取，否则自动生成"""
    for ext in [".png", ".jpg", ".bmp"]:
        p = os.path.join(image_dir, name + ext)
        if os.path.exists(p):
            img = mpimg.imread(p)
            if img.ndim == 3:
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            if img.max() > 1.0:
                img = img / 255.0
            return img.astype(np.float64)

    # shapes / weak_edge / kanizsa
    H, W = 200, 200
    y, x = np.mgrid[0:H, 0:W]
    img = np.ones((H, W), dtype=np.float64)
    if name == "shapes":
        # 圆环 + 三角形 + 拱门
        r_ring = np.sqrt((x - 55)**2 + (y - 55)**2)
        img[(r_ring <= 35) & (r_ring >= 15)] = 0.0
        img[(x >= 115) & (y >= 25) & ((x - 115) + (y - 25) <= 65)] = 0.0
        rect = (x >= 55) & (x <= 145) & (y >= 115) & (y <= 175)
        hole = np.sqrt((x - 100)**2 + (y - 160)**2) <= 28
        img[rect & (~hole)] = 0.0
    elif name == "weak_edge":
        img = np.exp(-((x - 100)**2 + (y - 100)**2) / (2.0 * 42.0**2))
    else: # kanizsa
        img[:] = 0.5
        for cy, cx in [(55, 55), (55, 145), (145, 55), (145, 145)]:
            img[np.sqrt((x - cx)**2 + (y - cy)**2) <= 26] = 0.0
        img[(x >= 55) & (x <= 145) & (y >= 55) & (y <= 145)] = 0.5
    return img

def run_2d_segmentation_comparison(image_names=("shapes", "weak_edge", "kanizsa", "coins"),
                                   noise_level=0.0, output_dir="../doc/segmentation_results"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"{'Image':<12} | {'GAC Area (c1, c2)':<20} | {'Chan-Vese Area (c1, c2)':<23} | {'Frame-Seg Area (c1, c2) [k*]':<28}")

    for name in image_names:
        f_clean = load_or_generate_test_image(name)
        np.random.seed(42)
        f = np.clip(f_clean + noise_level * np.random.randn(*f_clean.shape), 0.0, 1.0)

        init_tp = "circle" if name == "weak_edge" else "box"
        phi_gac, mask_gac, c1_g, c2_g = solve_gac(f, init_type=init_tp)
        phi_cv,  mask_cv,  c1_c, c2_c = solve_chan_vese(f, init_type=init_tp)
        u_fr,    mask_fr,  c1_f, c2_f, k_fr = solve_frame_segmentation(f, alpha=0.02, mu=5.0, frame=1)

        # 前景面积占比
        area_g = min(np.mean(mask_gac), 1.0 - np.mean(mask_gac))
        area_c = min(np.mean(mask_cv),  1.0 - np.mean(mask_cv))
        area_f = min(np.mean(mask_fr),  1.0 - np.mean(mask_fr))

        print(f"{name:<12} | {area_g:.4f} ({c1_g:.2f},{c2_g:.2f})   | "
              f"{area_c:.4f} ({c1_c:.2f},{c2_c:.2f})      | "
              f"{area_f:.4f} ({c1_f:.2f},{c2_f:.2f}) [k={k_fr}]")

        # Input | GAC | Chan-Vese | Wavelet Frame
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        axes[0].imshow(f, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"Input: {name} (noise={noise_level})", fontsize=11)

        for ax, m_contour, title in zip(
            axes[1:],
            [phi_gac, phi_cv, u_fr - 0.5],
            [f"GAC (Area={area_g:.3f})", f"Chan-Vese (Area={area_c:.3f})", f"Wavelet Frame (Area={area_f:.3f}, k={k_fr})"]
        ):
            ax.imshow(f, cmap='gray', vmin=0, vmax=1)
            ax.contour(m_contour, levels=[0.0], colors='red', linewidths=1.8)
            ax.set_title(title, fontsize=11)

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"seg2d_{name}_n{noise_level}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)