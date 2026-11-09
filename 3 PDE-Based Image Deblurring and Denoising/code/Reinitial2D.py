import numpy as np

def Reinitial2D(phi: np.ndarray, steps: int = 10, mode: str = "pixel") -> np.ndarray:
    """
    基于 Godunov 迎风格式的二维水平集重初始化函数 (Reinitialization)
    求解偏微分方程: phi_t + sign(phi_0)(|nabla phi| - 1) = 0
    将水平集函数修正为有符号距离函数 (SDF, |nabla phi| = 1)，同时保持零水平集位置不变。
    
    参数:
        phi: 二维 numpy 数组，待修正的水平集函数
        steps: 重初始化演化的迭代步数 (通常取 5 ~ 10 步)
    返回:
        phi: 修正后的水平集数组 (具有接近 1 的梯度模长)
    """
    phi = phi.copy().astype(np.float64)
    M, N = phi.shape
    eps = 1e-9
    CFL = 0.5

    if mode == "normalized":
        h = 1.0 / (max(N, M) - 1.0)
        dx = 1.0 / (M - 1.0)
        dy = 1.0 / (N - 1.0)
    else:
        # 像素空间步长 dx = dy = 1.0，保持零水平集两侧每个像素距離变化量为 1
        h = 1.0
        dx = 1.0
        dy = 1.0

    dt = h * CFL
    
    for _ in range(steps):
        # 沿行 (x) 与沿列 (y) 的前向/后向差分 (Neumann 边界)
        xp = (np.vstack([phi[1:, :], phi[-1:, :]]) - phi) / dx
        xn = (phi - np.vstack([phi[:1, :], phi[:-1, :]])) / dx
        yp = (np.hstack([phi[:, 1:], phi[:, -1:]]) - phi) / dy
        yn = (phi - np.hstack([phi[:, :1], phi[:, :-1]])) / dy
        
        # 符号指示掩模: phi >= 0 (外侧/内侧)
        phi_p = (phi >= 0).astype(np.float64)
        phi_n = 1.0 - phi_p
        
        # Godunov 迎风差分格式逼近梯度模长
        Godnov_p = np.sqrt(
            np.maximum(np.maximum(xn, 0.0)**2, np.minimum(xp, 0.0)**2) +
            np.maximum(np.maximum(yn, 0.0)**2, np.minimum(yp, 0.0)**2)
        )
        Godnov_n = np.sqrt(
            np.maximum(np.minimum(xn, 0.0)**2, np.maximum(xp, 0.0)**2) +
            np.maximum(np.minimum(yn, 0.0)**2, np.maximum(yp, 0.0)**2)
        )
        
        # 带有光滑化符号因子的时间步进更新 (Rusanov / Sussman 修正项)
        denom_p = np.sqrt(phi**2 + (Godnov_p**2) * dx * dy + eps)
        denom_n = np.sqrt(phi**2 + (Godnov_n**2) * dx * dy + eps)
        
        phi = (phi
               - dt * phi_p * (Godnov_p - 1.0) * phi / denom_p
               - dt * phi_n * (Godnov_n - 1.0) * phi / denom_n)
        
    return phi
