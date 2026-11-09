import numpy as np

def cellnorm3D(x, type=2):
    if isinstance(x, list):
        Level = len(x)
        nD = len(x[0])
        nD1 = len(x[0][0])
        nD2 = len(x[0][0][0])
        cn = 0.0
        for L in range(Level):
            for ii in range(nD):
                for jj in range(nD1):
                    for kk in range(nD2):
                        if type == 2:
                            cn += np.linalg.norm(x[L][ii][jj][kk])**2
                        elif type == 1:
                            cn += np.sum(np.abs(x[L][ii][jj][kk]))
        if type == 2:
            cn = np.sqrt(cn)
    else:
        if type == 2:
            cn = np.linalg.norm(x)
        elif type == 1:
            cn = np.sum(np.abs(x))
    return cn
