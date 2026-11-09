import numpy as np
from scipy.signal import convolve2d

def ConvSymAsym2D(A, M, b, L):
    m, n = A.shape
    nM = len(M)
    step = 2**(L - 1)

    ker = np.zeros(step * (nM - 1) + 1)
    ker[0::step] = M
    
    # 将一维卷积核转为二维列向量，以便进行 2D 卷积
    ker_2d = ker[:, np.newaxis]

    if b == 'c':
        # 模拟 MATLAB 的 imfilter(A, ker, 'circular')
        # imfilter 默认执行相关性(correlation)，而 convolve2d 执行卷积(convolution)
        # 因此需要将核翻转 (flipud) 来保持等价
        C = convolve2d(A, np.flipud(ker_2d), mode='same', boundary='wrap')
    else:
        lker = len(ker) // 2
        # MATLAB padarray 'symmetric'
        Ae = np.pad(A, ((lker, lker), (0, 0)), mode='symmetric')
        if b == 'a':
            Ae[:lker, :] = -Ae[:lker, :]
            Ae[-lker:, :] = -Ae[-lker:, :]
        # valid 卷积，结果尺寸与原图一致
        C = convolve2d(Ae, ker_2d, mode='valid')
    
    return C