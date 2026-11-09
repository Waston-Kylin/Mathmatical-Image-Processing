import sys
sys.path.append('./2D')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from GenerateFrameletFilter import GenerateFrameletFilter
from FraDecMultiLevel2D import FraDecMultiLevel2D
from FraRecMultiLevel2D import FraRecMultiLevel2D

if __name__ == '__main__':
    frame = 1  # 0: Haar; 1: Piecewise linear; 3: Piecewise cubic
    Level = 2
    
    D, R = GenerateFrameletFilter(frame)
    
    # 定义 W 和 WT
    def W(x): return FraDecMultiLevel2D(x, D, Level)
    def WT(x): return FraRecMultiLevel2D(x, R, Level)

    # 读取图像 (请确保当前目录下有 aircraft.jpg，或者替换为你的 peppers256.png)
    try:
        f = mpimg.imread('./image/aircraft.jpg')
    except FileNotFoundError:
        print("未找到 aircraft.jpg，请检查图像路径。")
        exit()

    if f.ndim == 3:
        f = np.dot(f[..., :3], [0.2989, 0.5870, 0.1140])

    if f.max() > 1.0:
        f = f / 255.0

    Cf = W(f)
    f_reconstruction = WT(Cf)

    nD = len(D) - 1 if isinstance(D[-1], (str, list, tuple)) else len(D)
    fig1 = plt.figure(1, figsize=(10, 8))
    kk = 1
    for kn in range(nD):
        for km in range(nD):
            if kn > 0 or km > 0:
                plt.subplot(3, 3, kk)
                plt.imshow(Cf[0][kn][km], cmap='gray')
                plt.title(f'Band ({km+1}, {kn+1})')
            kk += 1

    fig2 = plt.figure(2, figsize=(10, 8))
    kk = 1
    for kn in range(nD):
        for km in range(nD):
            if kn == 0 and km == 0:
                plt.subplot(3, 3, kk)
                plt.imshow(Cf[1][kn][km], cmap='gray')
                plt.title('Low Frequency Approx')
            elif kn > 0 or km > 0:
                plt.subplot(3, 3, kk)
                plt.imshow(Cf[1][kn][km], cmap='gray')
                plt.title(f'Band ({km+1}, {kn+1})')
            kk += 1

    plt.show()
    
    err = np.linalg.norm(f - f_reconstruction, 'fro')
    print(f"Reconstruction error = {err:.5e} (Perfect Reconstruction)")