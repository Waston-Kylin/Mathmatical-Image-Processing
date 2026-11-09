import copy
from FraRec2D import FraRec2D

def FraRecMultiLevel2D(C, R, L):
    C_list = [copy.copy(level) for level in C]
    for k in range(L - 1, 0, -1):
        C_list[k - 1] = [list(row) for row in C_list[k - 1]]
        C_list[k - 1][0][0] = FraRec2D(C_list[k], R, k + 1)
        
    Rec = FraRec2D(C_list[0], R, 1)
    return Rec