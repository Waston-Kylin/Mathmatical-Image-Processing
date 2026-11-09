import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def fspecial(shape, sigma):
    """生成高斯模糊核"""
    s = shape[0]
    center = (s + 1.0) / 2.0
    y, x = np.mgrid[1:s+1, 1:s+1]
    h = np.exp(-((x - center)**2 + (y - center)**2) / (sigma * sigma))
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def psf2otf(psf, shape):
    """空间核转频域传递函数 (周期边界)"""
    psf_pad = np.zeros(shape)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    psf_pad = np.roll(psf_pad, -(psf.shape[0] // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(psf.shape[1] // 2), axis=1)
    return fft2(psf_pad)

def neumann_shift(u):
    """
    Neumann 边界条件下的四邻域平移 (零通量边界 ∂u/∂N = 0)
    return: u_{i+1,j}, u_{i-1,j}, u_{i,j+1}, u_{i,j-1}
    """
    u_ip1 = np.pad(u[:, 1:], ((0, 0), (0, 1)), mode='edge')   # x+1 (右)
    u_im1 = np.pad(u[:, :-1], ((0, 0), (1, 0)), mode='edge')  # x-1 (左)
    u_jp1 = np.pad(u[1:, :], ((0, 1), (0, 0)), mode='edge')   # y+1 (下)
    u_jm1 = np.pad(u[:-1, :], ((1, 0), (0, 0)), mode='edge')  # y-1 (上)
    return u_ip1, u_im1, u_jp1, u_jm1

def minmod(a, b):
    """minmod 斜率限制器，用于抑制双曲方程数值振荡"""
    res = np.zeros_like(a)
    mask = (a * b) > 0
    res[mask] = np.sign(a[mask]) * np.minimum(np.abs(a[mask]), np.abs(b[mask]))
    return res

# Heat Equation
def solve_heat(u0, T=2.0, dt=0.15, u_true=None):
    """
    各向同性扩散: ∂u/∂t = Δu
    去噪 Baseline
    """
    u = u0.copy()
    num_steps = int(np.round(T / dt))
    psnr_history = []
    max_psnr, max_ssim, best_t = -1.0, -1.0, 0.0

    for step in range(1, num_steps + 1):
        u_ip1, u_im1, u_jp1, u_jm1 = neumann_shift(u)
        lap = u_ip1 + u_im1 + u_jp1 + u_jm1 - 4.0 * u
        u = u + dt * lap

        if u_true is not None:
            u_curr = np.clip(u, 0, 1)
            p = compare_psnr(u_true, u_curr, data_range=1.0)
            psnr_history.append((step * dt, p))
            if p > max_psnr:
                max_psnr = p
                best_t = step * dt
                res_s = compare_ssim(u_true, u_curr, data_range=1.0, win_size=15)
                max_ssim = float(res_s[0]) if isinstance(res_s, tuple) else float(res_s)

    return np.clip(u, 0, 1), max_psnr, max_ssim, best_t, psnr_history

# Perona-Malik Equation
def solve_pm(u0, T=5.0, dt=0.15, K=0.05, scheme="symmetric", u_true=None):
    """
    各向异性扩散: ∂u/∂t = div( c(|∇u|^2) ∇u ), c(s) = 1 / (1 + s/K^2)
    scheme:
      - 'symmetric'  : 半步中心对称格式
      - 'asymmetric' : 标准单侧非对称格式
    """
    u = u0.copy()
    num_steps = int(np.round(T / dt))
    psnr_history = []
    max_psnr, max_ssim, best_t = -1.0, -1.0, 0.0
    K2 = K * K

    for step in range(1, num_steps + 1):
        u_ip1, u_im1, u_jp1, u_jm1 = neumann_shift(u)

        if scheme == "symmetric":
            # 四个半整数网格点计算方向梯度与电导率 b
            dx_p = u_ip1 - u
            dx_m = u - u_im1
            dy_p = u_jp1 - u
            dy_m = u - u_jm1

            b_p0 = 1.0 / (1.0 + (dx_p**2) / K2)
            b_m0 = 1.0 / (1.0 + (dx_m**2) / K2)
            b_0p = 1.0 / (1.0 + (dy_p**2) / K2)
            b_0m = 1.0 / (1.0 + (dy_m**2) / K2)

            div_term = b_p0 * dx_p - b_m0 * dx_m + b_0p * dy_p - b_0m * dy_m

        elif scheme == "asymmetric":
            # 非对称: b_{i+1,j}u_{i+1,j} + b_{i,j}u_{i-1,j} ...
            dx_m = u - u_im1
            dy_m = u - u_jm1
            grad_sq = dx_m**2 + dy_m**2
            b = 1.0 / (1.0 + grad_sq / K2)

            b_ip1, _, b_jp1, _ = neumann_shift(b)
            div_term = (b_ip1 * (u_ip1 - u) - b * (u - u_im1) +
                        b_jp1 * (u_jp1 - u) - b * (u - u_jm1))
        else:
            raise ValueError("scheme 必须为 'symmetric' 或 'asymmetric'")

        u = u + dt * div_term
        
        if u_true is not None:
            u_curr = np.clip(u, 0, 1)
            p = compare_psnr(u_true, u_curr, data_range=1.0)
            psnr_history.append((step * dt, p))
            if p > max_psnr:
                max_psnr = p
                best_t = step * dt
                res_s = compare_ssim(u_true, u_curr, data_range=1.0, win_size=15)
                max_ssim = float(res_s[0]) if isinstance(res_s, tuple) else float(res_s)

    return np.clip(u, 0, 1), max_psnr, max_ssim, best_t, psnr_history

# Shock Filter
def solve_shock(u0, T=3.0, dt=0.1, L_type="u_nn", u_true=None):
    """
    双曲型锐化方程: ∂u/∂t = - |∇u| F(L(u)), F(s) = sign(s)
    L_type:
      - 'laplacian' : L(u) = Δu = u_xx + u_yy
      - 'u_nn' : L(u) = u_NN = (u_x^2 u_xx + 2 u_x u_y u_xy + u_y^2 u_yy) / |∇u|^2
    """
    u = u0.copy()
    num_steps = int(np.round(T / dt))
    max_psnr, max_ssim, best_t = -1.0, -1.0, 0.0

    for step in range(1, num_steps + 1):
        u_ip1, u_im1, u_jp1, u_jm1 = neumann_shift(u)

        # minmod 算子计算迎风梯度模长 |∇u|
        dx_plus = u_ip1 - u
        dx_minus = u - u_im1
        dy_plus = u_jp1 - u
        dy_minus = u - u_jm1

        mx = minmod(dx_plus, dx_minus)
        my = minmod(dy_plus, dy_minus)
        grad_mag = np.sqrt(mx**2 + my**2)

        # 中心差分计算二阶导算子 L(u)
        u_xx = u_ip1 - 2.0 * u + u_im1
        u_yy = u_jp1 - 2.0 * u + u_jm1

        if L_type == "laplacian":
            L_val = u_xx + u_yy
        elif L_type == "u_nn":
            # 中心差分一阶导与交叉二阶导 u_xy
            u_x = 0.5 * (u_ip1 - u_im1)
            u_y = 0.5 * (u_jp1 - u_jm1)

            u_ip1_jp1 = np.pad(u[1:, 1:], ((0, 1), (0, 1)), mode='edge')
            u_ip1_jm1 = np.pad(u[:-1, 1:], ((1, 0), (0, 1)), mode='edge')
            u_im1_jp1 = np.pad(u[1:, :-1], ((0, 1), (1, 0)), mode='edge')
            u_im1_jm1 = np.pad(u[:-1, :-1], ((1, 0), (1, 0)), mode='edge')
            u_xy = 0.25 * (u_ip1_jp1 - u_ip1_jm1 - u_im1_jp1 + u_im1_jm1)

            denom = u_x**2 + u_y**2 + 1e-10
            L_val = (u_x**2 * u_xx + 2.0 * u_x * u_y * u_xy + u_y**2 * u_yy) / denom
        else:
            raise ValueError("L_type 必须为 'laplacian' 或 'u_nn'")

        # 演化更新
        u = u - dt * grad_mag * np.sign(L_val)

        if u_true is not None:
            u_curr = np.clip(u, 0, 1)
            p = compare_psnr(u_true, u_curr, data_range=1.0)
            if p > max_psnr:
                max_psnr = p
                best_t = step * dt
                res_s = compare_ssim(u_true, u_curr, data_range=1.0, win_size=15)
                max_ssim = float(res_s[0]) if isinstance(res_s, tuple) else float(res_s)

    return np.clip(u, 0, 1), max_psnr, max_ssim, best_t

# 打印与三联画
def save_comparison(u_true, f, u, max_psnr, max_ssim, best_t, output_image, title_u="u"):
    end_psnr = compare_psnr(u_true, u, data_range=1.0)
    end_ssim = compare_ssim(u_true, u, data_range=1.0, win_size=min(15, min(u_true.shape)-1))

    log_text = (
        f"恢复图 (End) -> PSNR: {end_psnr:.3f}, SSIM: {end_ssim:.4f}\n"
        f"恢复图 (Max) -> PSNR: {max_psnr:.3f}, SSIM: {max_ssim:.4f}, (最佳停机时间 T* = {best_t:.2f})"
    )
    print(log_text+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, img, title in zip(axes, [u_true, f, u], ["u_true", "f", title_u]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else '.', exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig, log_text