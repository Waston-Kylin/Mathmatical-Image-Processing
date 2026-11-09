import numpy as np
from ConvSymAsym2D import ConvSymAsym2D

def FraRec2D(C, R, L):
    nR = len(R)
    SorAS = R[-1]
    ImSize = C[0][0].shape
    Rec = np.zeros(ImSize)
    
    for i in range(nR - 1):
        temp = np.zeros(ImSize)
        for j in range(nR - 1):
            M2 = R[j]
            temp = temp + ConvSymAsym2D(C[i][j].T, M2, SorAS[j], L).T
        M1 = R[i]
        Rec = Rec + ConvSymAsym2D(temp, M1, SorAS[i], L)
        
    return Rec