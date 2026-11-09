from FraDec2D import FraDec2D

def FraDecMultiLevel2D(A, D, L):
    Dec = []
    kDec = A
    for k in range(L):
        Dec_k = FraDec2D(kDec, D, k + 1)
        Dec.append(Dec_k)
        kDec = Dec_k[0][0]
        
    return Dec