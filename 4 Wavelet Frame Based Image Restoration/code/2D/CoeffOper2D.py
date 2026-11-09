import numpy as np

def wthresh(x, opt, t):
    if opt == 's':
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
    return x

def CoeffOper2D(op, alpha, beta=None):
    """
    小波框架系数算子: 实现 alpha op beta = gamma
    支持 op: '=', '0', '-', '+', '*', 's' (各向异性软阈值), 'vs' (各向同性耦合软阈值)
    """
    Level = len(alpha)
    nD = len(alpha[0])
    nD1 = len(alpha[0][0])
    gamma = []

    for ki in range(Level):
        if op == 'vs':
            vec = np.zeros_like(alpha[ki][0][0], dtype=float)
            for ji in range(nD):
                for jj in range(nD1):
                    if (ji != 0 or jj != 0) and np.sum(beta[ki][ji][jj]) != 0:
                        vec += alpha[ki][ji][jj]**2
            vec = np.sqrt(vec)

        gamma_ki = []
        for ji in range(nD):
            gamma_ji = []
            for jj in range(nD1):
                if op == '=':
                    val = alpha[ki][ji][jj].copy()
                elif op == '0':
                    val = np.zeros_like(alpha[ki][ji][jj], dtype=float)
                elif op == '-':
                    val = alpha[ki][ji][jj] - beta[ki][ji][jj]
                elif op == '+':
                    val = alpha[ki][ji][jj] + beta[ki][ji][jj]
                elif op == '*':
                    val = alpha[ki][ji][jj] * beta
                elif op == 's':
                    # (0, 0) 为低频近似分量，保持不收缩；仅对高频细节频带做软阈值
                    if ji == 0 and jj == 0:
                        val = alpha[ki][ji][jj].copy()
                    else:
                        t_val = beta[ki][ji][jj] if isinstance(beta, list) else beta
                        val = wthresh(alpha[ki][ji][jj], 's', t_val)
                elif op == 'vs':
                    if (ji != 0 or jj != 0) and np.sum(beta[ki][ji][jj]) != 0:
                        val = np.maximum(vec - beta[ki][ji][jj], 0.0) * (alpha[ki][ji][jj] / (vec + 1e-20))
                    else:
                        val = alpha[ki][ji][jj].copy()
                else:
                    raise ValueError(f"Unknown operation: {op}")
                gamma_ji.append(val)
            gamma_ki.append(gamma_ji)
        gamma.append(gamma_ki)

    return gamma