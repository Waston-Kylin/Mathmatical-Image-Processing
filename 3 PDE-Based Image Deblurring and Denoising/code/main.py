import os
import argparse
import numpy as np
import matplotlib.image as mpimg
from scipy.fftpack import fft2, ifft2
import mysubroutines as sub

def load_image(name):
    image_path = f"image/{name}.png"
    img = mpimg.imread(image_path)
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    if img.max() > 1.0:
        img = img / 255.0
    return img.astype(np.float64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW3: Level Set Image Segmentation (GAC, CV, Convex-CV)")
    parser.add_argument("--name", default="shapes", help="图像名称")
    parser.add_argument("--method", default="cv", choices=["gac", "cv", "convex_cv"], help="分割模型")
    parser.add_argument("--init_type", default="box", choices=["box", "circle", "corner_circle", "multi_circles"], help="初始水平集轮廓")
    parser.add_argument("--noise_level", type=float, default=0.00, help="高斯白噪声标准差")
    parser.add_argument("--blur", action="store_true", help="是否添加高斯模糊退化")
    parser.add_argument("--blur_sigma", type=float, default=2.0, help="高斯模糊核标准差")

    # 迭代参数
    parser.add_argument("--max_iter", type=int, default=1000, help="最大迭代步数")
    parser.add_argument("--dt", type=float, default=0.4, help="PDE 时间步长")
    parser.add_argument("--reinit_int", type=int, default=5, help="重初始化间隔步数")

    # GAC 参数
    parser.add_argument("--alpha", type=float, default=-0.2, help="GAC 气球力系数 (收缩取负，扩张取正)")
    parser.add_argument("--edge_k", type=float, default=0.05, help="GAC 边缘停止阈值")

    # CV / Convex-CV 参数
    parser.add_argument("--mu", type=float, default=1.0, help="轮廓长度正则项权重 mu")
    parser.add_argument("--lam_admm", type=float, default=1.0, help="Convex-CV ADMM 惩罚参数")

    args = parser.parse_args()
    u_clean = load_image(args.name)

    # 2. 添加模糊与噪声 (固定随机种子保证实验可复现)
    np.random.seed(42)
    f = u_clean.copy()
    if args.blur:
        kernel = sub.fspecial([15, 15], args.blur_sigma)
        otf_K = sub.psf2otf(kernel, u_clean.shape)
        f = np.real(ifft2(fft2(u_clean) * otf_K))

    if args.noise_level > 0:
        f = f + args.noise_level * np.random.randn(*u_clean.shape)
    f = np.clip(f, 0.0, 1.0)

    # 初始水平集 / 分割器
    phi0 = sub.init_level_set(f.shape, init_type=args.init_type)
    phi, seg_mask, title_m = None, None, ""

    if args.method == "gac":
        phi, seg_mask = sub.solve_gac(
            f, phi0=phi0, max_iter=args.max_iter, dt=args.dt,
            alpha=args.alpha, edge_k=args.edge_k, reinit_int=args.reinit_int
        )
        title_m = f"GAC (iter={args.max_iter})"

    elif args.method == "cv":
        phi, seg_mask = sub.solve_cv(
            f, phi0=phi0, max_iter=args.max_iter, dt=args.dt,
            mu=args.mu, reinit_int=args.reinit_int
        )
        title_m = f"Original CV (mu={args.mu})"

    elif args.method == "convex_cv":
        phi, seg_mask = sub.solve_convex_cv(
            f, max_iter=args.max_iter, mu=args.mu, lam_admm=args.lam_admm
        )
        title_m = f"Convex CV (mu={args.mu})"

    if args.blur:
        output_image = f"../doc/result/{args.name}_{args.method}_{args.init_type}_{args.noise_level}_{args.blur}_{args.blur_sigma}.png"
    else:
        output_image = f"../doc/result/{args.name}_{args.method}_{args.init_type}_{args.noise_level}_{args.blur}.png"
    sub.save_segmentation_result(u_clean, f, phi, seg_mask, output_image, title_model=title_m)
