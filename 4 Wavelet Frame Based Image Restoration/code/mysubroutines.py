# File Name: code/mysubroutines.py
import os
import sys
sys.path.append('./2D')
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from GenerateFrameletFilter import GenerateFrameletFilter
from FraDecMultiLevel2D import FraDecMultiLevel2D
from FraRecMultiLevel2D import FraRecMultiLevel2D
from CoeffOper2D import CoeffOper2D
from cellnorm2D import cellnorm2D


# 模糊核与频域
def fspecial(shape, sigma):
    """生成二维高斯卷积核 (对齐 MATLAB fspecial('gaussian'))"""
    s = shape[0]
    center = (s + 1.0) / 2.0
    y, x = np.mgrid[1:s+1, 1:s+1]
    h = np.exp(-((x - center)**2 + (y - center)**2) / (sigma * sigma))
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def psf2otf(psf, shape):
    """将空间点扩散函数转换为频域光学传递函数"""
    psf_pad = np.zeros(shape, dtype=float)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    psf_pad = np.roll(psf_pad, -(psf.shape[0] // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(psf.shape[1] // 2), axis=1)
    return fft2(psf_pad)

def build_threshold_cell(ref_coeff, base_thresh):
    """
    构造各层小波系数的阈值结构体 lambda:
    低频 (0, 0) 置 0；高频频带采用尺度自适应 / 统一阈值
    """
    Level = len(ref_coeff)
    nD = len(ref_coeff[0])
    nD1 = len(ref_coeff[0][0])
    thr = CoeffOper2D('0', ref_coeff)
    for ki in range(Level):
        # 细尺度噪声集中在第 1 层，随层数加深可保持或微调
        level_factor = 1.0 / (2.0 ** (ki * 0.5))
        for ji in range(nD):
            for jj in range(nD1):
                if ji != 0 or jj != 0:
                    thr[ki][ji][jj] = np.ones_like(ref_coeff[ki][ji][jj]) * (base_thresh * level_factor)
    return thr

def _safe_ssim(u_true, u_est):
    res = compare_ssim(u_true, u_est, data_range=1.0, win_size=15)
    return float(res[0]) if isinstance(res, tuple) else float(res)


# Analysis Approach
def solve_analysis_admm(f, kernel, lambda_weight=0.001, mu=0.02, delta=1.0,
                        frame=1, Level=2, max_iter=50, tol=1e-4, thresh_type='s', u_true=None):
    """
    Homework 4 Algorithm 1: Analysis Based Approach via Split Bregman (ADMM)
    求解: min_u ||lambda cdot Wu||_1 + 1/2 ||Au - f||_2^2
    迭代公式 (Lecture 7 Page 18):
        u_{k+1} = (A^T A + mu I)^{-1} (A^T f + mu W^T(d_k - b_k))
        d_{k+1} = T_{lambda / mu}(W u_{k+1} + b_k)
        b_{k+1} = b_k + delta (W u_{k+1} - d_{k+1})
    """
    H, W_dim = f.shape
    otf_K = psf2otf(kernel, (H, W_dim))

    # 紧框架重构性质 W^T W = I，频域分母化简为 |otf_K|^2 + mu
    Denominator = np.abs(otf_K)**2 + mu
    AT_f = np.conj(otf_K) * fft2(f)
    norm_f = np.linalg.norm(f, 'fro') + 1e-12

    D, R = GenerateFrameletFilter(frame)
    W = lambda x: FraDecMultiLevel2D(x, D, Level)
    WT = lambda c: FraRecMultiLevel2D(c, R, Level)

    # 初始化
    Wf0 = W(f)
    d = CoeffOper2D('0', Wf0)
    b = CoeffOper2D('0', Wf0)
    tau_cell = build_threshold_cell(Wf0, lambda_weight / mu)

    u = f.copy()
    max_psnr, max_ssim, best_iter = -1.0, -1.0, 0
    history = []

    # 主循环
    for k in range(1, max_iter + 1):
        # 更新 u_{k+1}
        WT_db = WT(CoeffOper2D('-', d, b))
        u = np.real(ifft2((AT_f + mu * fft2(WT_db)) / Denominator))

        # 更新 d_{k+1} = T_{\lambda/\mu}(W u_{k+1} + b_k)
        Wu = W(u)
        Wu_plus_b = CoeffOper2D('+', Wu, b)
        d = CoeffOper2D(thresh_type, Wu_plus_b, tau_cell)

        # 更新 b_{k+1} = b_k + \delta (W u_{k+1} - d_{k+1})
        diff_Wud = CoeffOper2D('-', Wu, d)
        b = CoeffOper2D('+', b, CoeffOper2D('*', diff_Wud, delta))

        # PSNR / SSIM
        if u_true is not None:
            u_curr = np.clip(u, 0.0, 1.0)
            cur_p = float(compare_psnr(u_true, u_curr, data_range=1.0))
            history.append((k, cur_p))
            if cur_p > max_psnr:
                max_psnr = cur_p
                best_iter = k
                max_ssim = _safe_ssim(u_true, u_curr)

        # 停机准则: ||W u_{k+1} - d_{k+1}||_2 / ||f||_2 < tol
        rel_err = cellnorm2D(diff_Wud, 2) / norm_f
        if rel_err < tol:
            break

    return np.clip(u, 0.0, 1.0), max_psnr, max_ssim, best_iter, history


# Balanced Approach
def solve_balanced_pfbs(f, kernel, lambda_weight=0.001, kappa=1.0, frame=1, Level=2,
                        max_iter=50, tol=1e-4, thresh_type='s', use_apg=False, u_true=None):
    """
    Homework 4 Algorithm 2: Balanced Approach via Proximal Forward-Backward Splitting (PFBS)
    求解: min_alpha ||lambda cdot alpha||_1 + 1/2 ||A W^T alpha - f||_2^2 + kappa/2 ||(I - W W^T)alpha||_2^2
    迭代公式 (Lecture 7 Page 24 & Page 26):
        nabla F_2(alpha_k) = W A^T (A W^T alpha_k - f) + kappa (I - W W^T) alpha_k
        g_k = alpha_k - nabla F_2(alpha_k) / L,   其中 L = rho_{max}(A^T A) + kappa = 1 + kappa
        alpha_{k+1} = T_{lambda / L}(g_k)
    参数 use_apg: 若为 True，则启用 Lecture 7 Page 26 的 Nesterov 加速梯度法 (APG/FISTA)
    """
    H, W_dim = f.shape
    otf_K = psf2otf(kernel, (H, W_dim))
    fft_f = fft2(f)
    norm_f = np.linalg.norm(f, 'fro') + 1e-12

    # 归一化高斯核满足 \rho_{max}(A^T A) = 1.0，故 Lipschitz 常数 L = 1.0 + \kappa
    L_lip = float(np.max(np.abs(otf_K)**2)) + kappa

    D, R = GenerateFrameletFilter(frame)
    W = lambda x: FraDecMultiLevel2D(x, D, Level)
    WT = lambda c: FraRecMultiLevel2D(c, R, Level)

    # 初始化 \alpha_0 = W f
    alpha = W(f)
    alpha_prev = CoeffOper2D('=', alpha)
    t_k = 1.0
    t_prev = 0.0

    tau_cell = build_threshold_cell(alpha, lambda_weight / L_lip)

    u = f.copy()
    max_psnr, max_ssim, best_iter = -1.0, -1.0, 0
    history = []

    for k in range(1, max_iter + 1):
        # 开启 APG，先计算外推点 \beta_k；否则直接取 \beta_k = \alpha_k
        if use_apg and k > 1:
            momentum = (t_prev - 1.0) / t_k
            diff_a = CoeffOper2D('-', alpha, alpha_prev)
            beta_k = CoeffOper2D('+', alpha, CoeffOper2D('*', diff_a, momentum))
        else:
            beta_k = alpha

        # 重构图像 u_beta = W^T \beta_k
        u_beta = WT(beta_k)

        # 频域计算 A^T (A W^T \beta_k - f)
        residual_img = np.real(ifft2(np.conj(otf_K) * (otf_K * fft2(u_beta) - fft_f)))
        term1 = W(residual_img)  # W A^T (A W^T \beta_k - f)

        # 计算 \kappa (I - W W^T) \beta_k = \kappa (\beta_k - W(u_beta))
        W_ubeta = W(u_beta)
        term2 = CoeffOper2D('*', CoeffOper2D('-', beta_k, W_ubeta), kappa)

        # 梯度步: g_k = \beta_k - \nabla F_2(\beta_k) / L
        grad_F2 = CoeffOper2D('+', term1, term2)
        g_k = CoeffOper2D('-', beta_k, CoeffOper2D('*', grad_F2, 1.0 / L_lip))

        # 近端软阈值步: \alpha_{k+1} = T_{\lambda / L}(g_k)
        alpha_next = CoeffOper2D(thresh_type, g_k, tau_cell)

        # 更新变量并重构图像 u_{k+1} = W^T \alpha_{k+1}
        diff_alpha = CoeffOper2D('-', alpha_next, alpha)
        alpha_prev = alpha
        alpha = alpha_next
        t_prev = t_k
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * (t_k**2))) / 2.0

        u = WT(alpha)

        if u_true is not None:
            u_curr = np.clip(u, 0.0, 1.0)
            cur_p = float(compare_psnr(u_true, u_curr, data_range=1.0))
            history.append((k, cur_p))
            if cur_p > max_psnr:
                max_psnr = cur_p
                best_iter = k
                max_ssim = _safe_ssim(u_true, u_curr)

        if cellnorm2D(diff_alpha, 2) / norm_f < tol:
            break

    return np.clip(u, 0.0, 1.0), max_psnr, max_ssim, best_iter, history

# TV
def solve_tv(f, kernel, lambda_weight=0.001, mu=0.012, max_iter=50, u_true=None):
    """
    全变分 (TV) 模型求解器
    """
    H, W_dim = f.shape
    otf_K = psf2otf(kernel, (H, W_dim))

    dx_k = np.zeros((H, W_dim)); dx_k[0, 0] = -1.0; dx_k[0, 1] = 1.0
    dy_k = np.zeros((H, W_dim)); dy_k[0, 0] = -1.0; dy_k[1, 0] = 1.0
    otf_Dx, otf_Dy = fft2(dx_k), fft2(dy_k)

    Denominator = np.abs(otf_K)**2 + mu * (np.abs(otf_Dx)**2 + np.abs(otf_Dy)**2)
    AT_f = np.conj(otf_K) * fft2(f)

    u = f.copy()
    dx, dy = np.zeros((H, W_dim)), np.zeros((H, W_dim))
    bx, by = np.zeros((H, W_dim)), np.zeros((H, W_dim))
    tau = lambda_weight / mu

    max_psnr, max_ssim, best_iter = -1.0, -1.0, 0
    history = []

    for k in range(1, max_iter + 1):
        term_diff = np.conj(otf_Dx) * fft2(dx - bx) + np.conj(otf_Dy) * fft2(dy - by)
        u = np.real(ifft2((AT_f + mu * term_diff) / Denominator))

        nabla_x_u = np.roll(u, -1, axis=1) - u
        nabla_y_u = np.roll(u, -1, axis=0) - u
        vx, vy = nabla_x_u + bx, nabla_y_u + by
        V = np.sqrt(vx**2 + vy**2 + 1e-20)

        shrink = np.maximum(V - tau, 0.0) / V
        dx, dy = shrink * vx, shrink * vy
        bx += nabla_x_u - dx
        by += nabla_y_u - dy

        if u_true is not None:
            u_curr = np.clip(u, 0.0, 1.0)
            cur_p = float(compare_psnr(u_true, u_curr, data_range=1.0))
            history.append((k, cur_p))
            if cur_p > max_psnr:
                max_psnr = cur_p
                best_iter = k
                max_ssim = _safe_ssim(u_true, u_curr)

    return np.clip(u, 0.0, 1.0), max_psnr, max_ssim, best_iter, history

# 三联对比
def save_comparison(u_true, f, u, max_psnr, max_ssim, best_iter, output_image, title_u="Restored u"):
    orig_psnr = float(compare_psnr(u_true, f, data_range=1.0))
    orig_ssim = _safe_ssim(u_true, f)
    end_psnr = float(compare_psnr(u_true, u, data_range=1.0))
    end_ssim = _safe_ssim(u_true, u)

    log_text = (
        f"退化图 (Origin) -> PSNR: {orig_psnr:.3f} dB, SSIM: {orig_ssim:.4f}\n"
        f"恢复图 (End)    -> PSNR: {end_psnr:.3f} dB, SSIM: {end_ssim:.4f} (提升: {end_psnr - orig_psnr:+.3f} dB)\n"
        f"恢复图 (Max)    -> PSNR: {max_psnr:.3f} dB, SSIM: {max_ssim:.4f} (最佳迭代步 k* = {best_iter})"
    )
    print("=" * 68)
    print(log_text)
    print("=" * 68)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, img, title in zip(axes, [u_true, f, u], ["u_true", f"f (PSNR={orig_psnr:.2f})", f"{title_u} (PSNR={end_psnr:.2f})"]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if output_image:
        os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else '.', exist_ok=True)
        plt.savefig(output_image, dpi=300, bbox_inches='tight')

    return fig, log_text