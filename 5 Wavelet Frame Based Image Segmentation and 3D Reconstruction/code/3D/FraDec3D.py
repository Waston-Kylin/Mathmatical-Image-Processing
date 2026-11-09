import numpy as np
from ConvSymAsym3D import ConvSymAsym3D

def FraDec3D(A, D, L):
    nD = len(D)
    SorAS = D[-1]
    
    # 3D 结构是 3x3x3 的嵌套列表
    Dec = [[[None for _ in range(nD - 1)] for _ in range(nD - 1)] for _ in range(nD - 1)]
    
    for i in range(nD - 1):
        M1 = D[i]
        # 沿 axis 0 (z轴)
        tempi = ConvSymAsym3D(A, M1, SorAS[i], L, axis=0)
        for j in range(nD - 1):
            M2 = D[j]
            # 沿 axis 1 (y轴)
            tempj = ConvSymAsym3D(tempi, M2, SorAS[j], L, axis=1)
            for k in range(nD - 1):
                M3 = D[k]
                # 沿 axis 2 (x轴)
                tempk = ConvSymAsym3D(tempj, M3, SorAS[k], L, axis=2)
                Dec[i][j][k] = tempk
                
    return Dec