import argparse
from seg2d import run_2d_segmentation_comparison
from surf3d import reproduce_figure4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW5: Wavelet Frame Segmentation & 3D Surface Reconstruction")
    parser.add_argument("--task", default="all", choices=["all", "seg2d", "surf3d"],
                        help="任务: seg2d (2D 分割), surf3d (3D 表面重构), all (全部运行)")
    parser.add_argument("--noise_level", type=float, default=0.02,
                        help="2D 图像分割测试的高斯噪声标准差")
    parser.add_argument("--models", nargs="+", default=["cndragon", "lucy", "statuette"],
                        help="重构 3D 数据集")
    parser.add_argument("--alpha_3d", type=float, default=0.8, help="3D 重构参数 alpha")
    parser.add_argument("--mu_3d", type=float, default=1.0, help="3D 重构参数 mu")
    parser.add_argument("--max_iter_3d", type=int, default=1000, help="3D 重构最大迭代步数")
    parser.add_argument("--tol_3d", type=float, default=1e-4, help="3D 重构停机容差")
    parser.add_argument("--output_seg_dir", default="../doc/segmentation_results", help="图片保存目录")
    parser.add_argument("--output_surf_dir", default="../doc/reconstruction_results", help="图片数据保存目录")
    args = parser.parse_args()

    if args.task in ["all", "seg2d"]:
        # 无噪和含噪 args.noise_level 下 GAC vs CV vs Frame-based 对比
        run_2d_segmentation_comparison(noise_level=0.0, output_dir=args.output_seg_dir)
        if args.noise_level > 0:
            run_2d_segmentation_comparison(noise_level=args.noise_level, output_dir=args.output_seg_dir)

    if args.task in ["all", "surf3d"]:
        reproduce_figure4(
            data_dir="Data",
            output_dir=args.output_surf_dir,
            datasets=args.models,
            alpha=args.alpha_3d,
            mu=args.mu_3d,
            max_iter=args.max_iter_3d,
            tol=args.tol_3d,
        )