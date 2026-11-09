import numpy as np
from scipy.ndimage import convolve1d

def ConvSymAsym3D(A, M, b, L, axis=0):
    """
    沿特定轴 axis (0, 1, 2) 进行对称/反对称 1D 卷积
    """
    nM = len(M)
    step = 2**(L - 1)

    # 构造上采样后的卷积核
    ker = np.zeros(step * (nM - 1) + 1)
    ker[0::step] = M
    
    lker = len(ker) // 2
    
    # 边界延拓
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (lker, lker)
    
    if b == 'c': # 周期边界
        # 利用 numpy 的 roll 模拟 circular
        Ae = np.pad(A, pad_width, mode='wrap')
    elif b == 's': # 对称边界
        Ae = np.pad(A, pad_width, mode='symmetric')
    elif b == 'a': # 反对称边界
        Ae = np.pad(A, pad_width, mode='symmetric')
        # 翻转符号
        sl_start = [slice(None)] * 3
        sl_end = [slice(None)] * 3
        sl_start[axis] = slice(0, lker)
        sl_end[axis] = slice(-lker, None)
        Ae[tuple(sl_start)] = -Ae[tuple(sl_start)]
        Ae[tuple(sl_end)] = -Ae[tuple(sl_end)]
    
    # 在给定轴上进行 1D 卷积，mode='constant' 因为我们已经手动 pad 了
    C = convolve1d(Ae, ker, axis=axis, mode='constant')
    
    # 截取 valid 部分
    sl_out = [slice(None)] * 3
    sl_out[axis] = slice(lker, -lker)
    C = C[tuple(sl_out)]
    
    return C