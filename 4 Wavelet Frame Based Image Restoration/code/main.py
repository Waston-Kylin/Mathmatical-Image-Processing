import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import fft2, ifft2
import mysubroutines as sub

def load_clean_image(image_path):
    u_true = mpimg.imread(image_path)
    if u_true.ndim == 3:
        u_true = np.dot(u_true[..., :3], [0.2989, 0.5870, 0.1140])
    if u_true.max() > 1.0:
        u_true = u_true / 255.0
    return u_true.astype(np.float64)

def degrade_image(u_true, kernel_size=15, gaussian_sigma=1.0, noise_level=0.01):
    np.random.seed(42)
    kernel = sub.fspecial([kernel_size, kernel_size], gaussian_sigma)
    otf_K = sub.psf2otf(kernel, u_true.shape)
    f_blur = np.real(ifft2(fft2(u_true) * otf_K))
    f = f_blur + noise_level * np.random.randn(*u_true.shape)
    return np.clip(f, 0.0, 1.0), kernel

def run_single(args):
    u_true = load_clean_image(args.image_path)
    f, kernel = degrade_image(u_true, args.kernel_size, args.gaussian_sigma, args.noise_level)

    if args.method == "analysis":
        u, max_p, max_s, best_k, _ = sub.solve_analysis_admm(
            f, kernel, lambda_weight=args.lambda_weight, mu=args.mu, delta=args.delta,
            frame=args.frame, Level=args.Level, max_iter=args.max_iter, tol=args.tol,
            thresh_type=args.thresh_type, u_true=u_true
        )
        title_u = f"Analysis-ADMM (F={args.frame}, L={args.Level})"

    elif args.method == "balanced":
        u, max_p, max_s, best_k, _ = sub.solve_balanced_pfbs(
            f, kernel, lambda_weight=args.lambda_weight, kappa=args.kappa,
            frame=args.frame, Level=args.Level, max_iter=args.max_iter, tol=args.tol,
            thresh_type=args.thresh_type, use_apg=args.use_apg, u_true=u_true
        )
        alg_tag = "Balanced-APG" if args.use_apg else "Balanced-PFBS"
        title_u = f"{alg_tag} (F={args.frame}, k={args.kappa})"

    elif args.method == "tv":
        u, max_p, max_s, best_k, _ = sub.solve_tv(
            f, kernel, lambda_weight=args.lambda_weight, mu=args.mu,
            max_iter=args.max_iter, u_true=u_true
        )
        title_u = "TV (HW1 Baseline)"

    fig, _ = sub.save_comparison(u_true, f, u, max_p, max_s, best_k, args.output_image, title_u=title_u)
    plt.close(fig)

def run_full_benchmark():
    """一键运行完整对比实验: TV vs Analysis (ADMM) vs Balanced (PFBS)"""
    images = ["aircraft.jpg", "peppers256.png", "barbara.png"]
    noises = [0.01, 0.02]
    header = f"{'Image':<12} | {'Noise':<6} | {'Method':<16} | {'Origin PSNR/SSIM':<18} | {'End PSNR/SSIM':<18} | {'Max PSNR/SSIM (k*)':<22}"
    print(header)
    print("-" * len(header))

    for name in images:
        img_path = f"image/{name}"
        if not os.path.exists(img_path):
            continue
        u_true = load_clean_image(img_path)
        for nl in noises:
            f, kernel = degrade_image(u_true, 15, 1.0, nl)
            orig_p = sub.compare_psnr(u_true, f, data_range=1.0)
            orig_s = sub._safe_ssim(u_true, f)

            # 不同噪声级别匹配正则化参数
            lam_tv = 0.001 if nl == 0.01 else 0.0025
            lam_ana = 0.0008 if nl == 0.01 else 0.0020
            lam_bal = 0.0005 if nl == 0.01 else 0.0012

            configs = [
                ("TV", lambda: sub.solve_tv(f, kernel, lambda_weight=lam_tv, mu=0.012, max_iter=50, u_true=u_true)),
                ("Analysis (ADMM)", lambda: sub.solve_analysis_admm(f, kernel, lambda_weight=lam_ana, mu=0.015, frame=1, Level=2, max_iter=50, u_true=u_true)),
                ("Balanced (PFBS)", lambda: sub.solve_balanced_pfbs(f, kernel, lambda_weight=lam_bal, kappa=1.0, frame=1, Level=2, max_iter=80, use_apg=True, u_true=u_true)),
            ]

            for m_name, runner in configs:
                u_rec, max_p, max_s, best_k, _ = runner()
                end_p = sub.compare_psnr(u_true, u_rec, data_range=1.0)
                end_s = sub._safe_ssim(u_true, u_rec)
                print(f"{name:<12} | {nl:<6.2f} | {m_name:<16} | {orig_p:6.3f} / {orig_s:.4f}   | {end_p:6.3f} / {end_s:.4f}   | {max_p:6.3f} / {max_s:.4f} ({best_k})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW4: Wavelet Frame Based Image Restoration")
    parser.add_argument("--image_path", default="aircraft.jpg", help="输入图片路径")
    parser.add_argument("--method", default="analysis", choices=["analysis", "balanced", "tv"], help="恢复算法")
    parser.add_argument("--lambda_weight", type=float, default=0.0008, help="L1 稀疏正则项权重 lambda")
    parser.add_argument("--output_image", default="../docs/image/aircraft_analysis_n0.01_f1_ts.jpg", help="输出对比图保存路径")
    parser.add_argument("--kernel_size", type=int, default=15, help="高斯模糊核尺寸")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0, help="高斯模糊核标准差")
    parser.add_argument("--noise_level", type=float, default=0.01, help="高斯噪声标准差 (0.01 或 0.02)")

    # 小波框架公共参数
    parser.add_argument("--frame", type=int, default=1, choices=[0, 1, 3], help="小波类型: 0=Haar, 1=Linear, 3=Cubic")
    parser.add_argument("--Level", type=int, default=2, help="小波分解层数 (通常 1~4)")
    parser.add_argument("--thresh_type", default="s", choices=["s", "vs"], help="软阈值类型: s=各向异性, vs=各向同性向量阈值")
    parser.add_argument("--max_iter", type=int, default=50, help="最大迭代步数")
    parser.add_argument("--tol", type=float, default=1e-4, help="收敛停机容差 tol")

    # Analysis ADMM
    parser.add_argument("--mu", type=float, default=0.015, help="ADMM 增广拉格朗日惩罚参数 mu")
    parser.add_argument("--delta", type=float, default=1.0, help="ADMM 对偶步长 delta (0 < delta <= 1)")

    # Balanced PFBS
    parser.add_argument("--kappa", type=float, default=1.0, help="Balanced 模型权重 kappa (>0)")
    parser.add_argument("--use_apg", action="store_true", help="是否启用 Nesterov 加速 (Algorithm 3 APG/FISTA)")

    # 基准测试
    parser.add_argument("--benchmark", action="store_true", help="一键运行所有测试图与噪声级别的对比实验")

    args = parser.parse_args()
    if args.benchmark:
        run_full_benchmark()
    else:
        run_single(args)