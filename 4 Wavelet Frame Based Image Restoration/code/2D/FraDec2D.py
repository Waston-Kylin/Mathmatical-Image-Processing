import numpy as np
from ConvSymAsym2D import ConvSymAsym2D

def FraDec2D(A, D, L):
    nD = len(D)
    SorAS = D[-1]
    
    Dec = [[None for _ in range(nD - 1)] for _ in range(nD - 1)]
    
    for i in range(nD - 1):
        M1 = D[i]
        tempi = ConvSymAsym2D(A, M1, SorAS[i], L)
        for j in range(nD - 1):
            M2 = D[j]
            # MATLAB 中的 ' 表示共轭转置，对于实数等同于 .T
            tempj = ConvSymAsym2D(tempi.T, M2, SorAS[j], L)
            Dec[i][j] = tempj.T
            
    return Dec