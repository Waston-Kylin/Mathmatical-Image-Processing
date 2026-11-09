import argparse
import numpy as np
import matplotlib.image as mpimg
from scipy.fftpack import fft2, ifft2
import mysubroutines as sub

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW2: PDE Image Restoration")
    parser.add_argument("--name", default="barbara", choices=["barbara", "peppers256"], help="测试图片")
    parser.add_argument("--method", default="pm", choices=["heat", "pm", "shock"], help="选择 PDE 方程")
    parser.add_argument("--T", type=float, default=0.2, help="停机时间 T")
    parser.add_argument("--dt", type=float, default=0.01, help="时间步长 dt (建议 <= 0.1)")
    parser.add_argument("--noise_level", type=float, default=0.02, help="高斯噪声标准差 (0 即无噪声)")
    parser.add_argument("--blur", action="store_true", help="是否开启高斯模糊")
    parser.add_argument("--K", type=float, default=0.04, help="P-M 方程边缘梯度阈值 K")
    parser.add_argument("--scheme", default="symmetric", choices=["symmetric", "asymmetric"], help="P-M 离散格式")
    parser.add_argument("--L_type", default="u_nn", choices=["laplacian", "u_nn"], help="Shock Filter 算子 L 类型")
    args = parser.parse_args()

    image_path = f"image/{args.name}.png"
    u_true = mpimg.imread(image_path)
    if u_true.ndim == 3:
        u_true = np.dot(u_true[..., :3], [0.2989, 0.5870, 0.1140])
    if u_true.max() > 1.0:
        u_true = u_true / 255.0

    # 退化图像 f
    np.random.seed(42)
    f = u_true.copy()
    if args.blur:
        kernel = sub.fspecial([15, 15], 1.0)
        otf_K = sub.psf2otf(kernel, u_true.shape)
        f = np.real(ifft2(fft2(u_true) * otf_K))

    if args.noise_level > 0:
        f = f + args.noise_level * np.random.randn(*u_true.shape)
    f = np.clip(f, 0, 1)

    u, max_psnr, max_ssim, best_t, output_image, title_u = None, -1.0, -1.0, 0.0, "", ""

    # 求解 PDE 方程
    if args.method == "heat":
        output_image = f"image/hw2_{args.name}_heat_T{args.T}.png"
        u, max_psnr, max_ssim, best_t, _ = sub.solve_heat(f, T=args.T, dt=args.dt, u_true=u_true)
        title_u = f"Heat (T={args.T})"

    elif args.method == "pm":
        output_image = f"image/hw2_{args.name}_pm_{args.scheme}_T{args.T}.png"
        u, max_psnr, max_ssim, best_t, _ = sub.solve_pm(
            f, T=args.T, dt=args.dt, K=args.K, scheme=args.scheme, u_true=u_true
        )
        title_u = f"PM-{args.scheme} (T={args.T})"

    elif args.method == "shock":
        tag = "blur_noise" if args.noise_level > 0 else "blur_clean"
        output_image = f"image/hw2_{args.name}_shock_{args.L_type}_{tag}_T{args.T}.png"
        u, max_psnr, max_ssim, best_t = sub.solve_shock(
            f, T=args.T, dt=args.dt, L_type=args.L_type, u_true=u_true
        )
        title_u = f"Shock-{args.L_type} (T={args.T})"

    sub.save_comparison(u_true, f, u, max_psnr, max_ssim, best_t, output_image, title_u=title_u)