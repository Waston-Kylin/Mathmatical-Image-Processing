import argparse
import numpy as np
import matplotlib.image as mpimg
from scipy.fftpack import fft2, ifft2
import mysubroutines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="image/peppers256.png")
    parser.add_argument("--lambda_weight", type=float, default=0.001, help="TV正则项权重")
    parser.add_argument("--output_image", default="image/result_peppers256.png")
    parser.add_argument("--kernel_size", type=int, default=15, help="高斯核窗口边长")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0, help="高斯核方差参数")
    parser.add_argument("--noise_level", type=float, default=0.01, choices=[0.01, 0.02], help="高斯噪声级别")
    args = parser.parse_args()

    u_true = mpimg.imread(args.image_path)
    if u_true.ndim == 3:
        u_true = np.dot(u_true[...,:3], [0.2989, 0.5870, 0.1140])

    # processing f (add blur and noise)
    np.random.seed(42)
    kernel = mysubroutines.fspecial([args.kernel_size, args.kernel_size], args.gaussian_sigma)
    f_blur = np.real(ifft2(fft2(u_true) * mysubroutines.psf2otf(kernel, u_true.shape)))
    f = f_blur + args.noise_level * np.random.randn(*u_true.shape)
    f = np.clip(f, 0, 1)

    # solving u
    u, max_psnr, max_ssim = mysubroutines.mysolver(f, kernel, args.lambda_weight, u_true=u_true)

    mysubroutines.save_comparison(u_true, f, u, max_psnr, max_ssim, args.output_image)

if __name__ == "__main__":
    main()