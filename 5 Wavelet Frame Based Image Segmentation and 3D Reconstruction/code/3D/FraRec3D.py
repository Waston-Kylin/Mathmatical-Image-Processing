import numpy as np
from ConvSymAsym3D import ConvSymAsym3D

def FraRec3D(C, R, L):
    nR = len(R)
    SorAS = R[-1]
    ImSize = C[0][0][0].shape
    Rec = np.zeros(ImSize, dtype=C[0][0][0].dtype)
    
    for i in range(nR - 1):
        M1 = R[i]
        tempi = np.zeros(ImSize, dtype=C[0][0][0].dtype)
        for j in range(nR - 1):
            M2 = R[j]
            tempj = np.zeros(ImSize, dtype=C[0][0][0].dtype)
            for k in range(nR - 1):
                M3 = R[k]
                # 对 C[i][j][k] 沿 x 轴重构
                tempj += ConvSymAsym3D(C[i][j][k], M3, SorAS[k], L, axis=2)
            # 对 tempj 沿 y 轴重构
            tempi += ConvSymAsym3D(tempj, M2, SorAS[j], L, axis=1)
        # 对 tempi 沿 z 轴重构
        Rec += ConvSymAsym3D(tempi, M1, SorAS[i], L, axis=0)
        
    return Rec