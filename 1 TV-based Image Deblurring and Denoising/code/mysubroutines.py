import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def fspecial(shape, sigma):
    """
    Simulate fspecial('gaussian') to generate a convolution kernel
    """
    s = shape[0]
    center = (s + 1.0) / 2.0
    y, x = np.mgrid[1:s+1, 1:s+1]
    h = np.exp(-((x - center)**2 + (y - center)**2) / (sigma * sigma))
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def psf2otf(psf, shape):
    """
    Transform the spatial kernel into the frequency domain and handle periodic boundaries
    """
    psf_pad = np.zeros(shape)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    shift_r = -(psf.shape[0] // 2)
    shift_c = -(psf.shape[1] // 2)
    psf_pad = np.roll(psf_pad, shift_r, axis=0)
    psf_pad = np.roll(psf_pad, shift_c, axis=1)
    return fft2(psf_pad)

def mysolver(f, kernel, lambda_weight, u_true=None, mu=0.012, max_iter=50):
    """
    ADMM / Split Bregman TV Denoising and Deblurring Solver
    """
    H, W = f.shape
    otf_K = psf2otf(kernel, (H, W))

    dx_kernel = np.zeros((H, W))
    dx_kernel[0, 0] = -1.0
    dx_kernel[0, 1] = 1.0
    otf_Dx = fft2(dx_kernel)

    dy_kernel = np.zeros((H, W))
    dy_kernel[0, 0] = -1.0
    dy_kernel[1, 0] = 1.0
    otf_Dy = fft2(dy_kernel)

    Denominator = np.abs(otf_K)**2 + mu * (np.abs(otf_Dx)**2 + np.abs(otf_Dy)**2)
    AT_f = np.conj(otf_K) * fft2(f)

    u = f.copy()
    dx, dy = np.zeros((H, W)), np.zeros((H, W))
    bx, by = np.zeros((H, W)), np.zeros((H, W))
    tau = lambda_weight / mu

    max_psnr = -1.0
    max_ssim = -1.0

    for _ in range(max_iter):
        term_diff = np.conj(otf_Dx) * fft2(dx - bx) + np.conj(otf_Dy) * fft2(dy - by)
        u = np.real(ifft2((AT_f + mu * term_diff) / Denominator))

        nabla_x_u = np.roll(u, -1, axis=1) - u
        nabla_y_u = np.roll(u, -1, axis=0) - u
        vx, vy = nabla_x_u + bx, nabla_y_u + by
        V = np.sqrt(vx**2 + vy**2)
        V[V == 0] = 1e-10

        shrink = np.maximum(V - tau, 0) / V
        dx = shrink * vx
        dy = shrink * vy

        bx = bx + nabla_x_u - dx
        by = by + nabla_y_u - dy

        # track MAX value
        if u_true is not None:
            u_curr = np.clip(u, 0, 1)
            cur_p = compare_psnr(u_true, u_curr, data_range=1.0)
            cur_s = compare_ssim(u_true, u_curr, data_range=1.0, win_size=15)
            if cur_p > max_psnr:
                max_psnr = cur_p
            if cur_s > max_ssim:
                max_ssim = cur_s

    u_final = np.clip(u, 0, 1)
    return u_final, max_psnr, max_ssim

def save_comparison(u_true, f, u, max_psnr, max_ssim, output_image):
    end_psnr = compare_psnr(u_true, u, data_range=1.0)
    end_ssim = compare_ssim(u_true, u, data_range=1.0, win_size=15)

    print(f"恢复图 (End) -> PSNR: {end_psnr:.3f}, SSIM: {end_ssim:.4f}")
    print(f"恢复图 (Max) -> PSNR: {max_psnr:.3f}, SSIM: {max_ssim:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, img, title in zip(axes, [u_true, f, u], ["u_true", "f", "u"]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else '.', exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()