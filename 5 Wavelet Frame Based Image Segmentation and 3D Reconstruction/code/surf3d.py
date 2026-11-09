import os
import sys
import time
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

# 3D 小波框架模块
sys.path.append(os.path.join(os.path.dirname(__file__), '3D'))
from GenerateFrameletFilter import GenerateFrameletFilter
from FraDecMultiLevel3D import FraDecMultiLevel3D
from FraRecMultiLevel3D import FraRecMultiLevel3D
from CoeffOper3D import CoeffOper3D

def load_surface_mat(mat_path):
    """
    加载 phi 和 f，打印内部键名与分位数
    """
    try:
        data = sio.loadmat(mat_path)
    except NotImplementedError:
        data = {}
        with h5py.File(mat_path, 'r') as hf:
            for k in hf.keys():
                data[k] = np.array(hf[k]).T

    valid_keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"  [.mat keys in {os.path.basename(mat_path)}]: {valid_keys}")

    phi, f = None, None
    for k in valid_keys:
        kl = k.lower()
        if kl in ['phi', 'varphi', 'dist']:
            phi = np.asarray(data[k], dtype=np.float32)
        elif kl == 'f':
            f = np.asarray(data[k], dtype=np.float32)

    if phi is None or f is None:
        raise KeyError(f"Cannot find 'phi' and 'f' in {valid_keys}")

    print(f"  -> f   shape: {f.shape}, min={f.min():.4f}, median={np.median(f):.4f}, max={f.max():.4f}")
    print(f"  -> phi shape: {phi.shape}, min={phi.min():.4f}, median={np.median(phi):.4f}, max={phi.max():.4f}")
    return phi, f


# 构造 3D 阈值结构体
def zeros_like_coeff_3d(ref_coeff):
    """递归构造元素全为 0 嵌套列表"""
    if isinstance(ref_coeff, list):
        return [zeros_like_coeff_3d(item) for item in ref_coeff]
    elif isinstance(ref_coeff, np.ndarray):
        return np.zeros_like(ref_coeff)
    else:
        return ref_coeff

def build_3d_threshold_cell(ref_coeff, thr_3d):
    """
    阈值半径 lambda_{l,j} / mu = (alpha * varphi(x)) / mu
    低频子带 (0, 0, 0) 置 0，高频子带设为 thr_3d
    """
    thr = zeros_like_coeff_3d(ref_coeff)
    Level = len(thr)
    nD = len(thr[0])
    for ki in range(Level):
        for ji in range(nD):
            for jj in range(nD):
                for jk in range(nD):
                    if ji != 0 or jj != 0 or jk != 0:
                        thr[ki][ji][jj][jk] = thr_3d
    return thr


# Algorithm 3
def solve_surface_reconstruction_3d(phi, f, alpha=0.8,
                                    mu=1.0, gamma=0.5,
                                    frame=1, Level=1, max_iter=200, tol=5e-4):
    """
    3D 表面重构
    """
    if mu <= 0 or alpha < 0:
        raise ValueError("须 mu > 0，alpha >= 0")
    phi = np.asarray(phi, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)
    if phi.shape != f.shape:
        raise ValueError(f"phi 和 f 尺寸不匹配: {phi.shape} vs {f.shape}")
    
    D, R = GenerateFrameletFilter(frame)
    W = lambda x: FraDecMultiLevel3D(x, D, Level)
    WT = lambda c: FraRecMultiLevel3D(c, R, Level)

    # u0 = 0, d0 = 0, v0 = 0。
    u = np.zeros_like(f)
    template = W(u)
    d = zeros_like_coeff_3d(template)
    v = zeros_like_coeff_3d(template)
    del template

    # lambda = alpha*phi；T_(lambda/mu)
    distance = distance_transform_edt(phi > 0).astype(np.float32)
    tau_volume = (alpha / mu) * distance
    tau_cell = build_3d_threshold_cell(d, tau_volume)
    print(f"  alpha={alpha:g}, mu={mu:g}, "
            f"threshold range=[{tau_volume.min():.3g}, {tau_volume.max():.3g}]")

    t_start = time.time()
    for k in range(1, max_iter + 1):
        iter_t0 = time.time()
        u_prev = u.copy()

        # 更新 u = P_[0,1]( W^T(d-v) - (2f-1)/mu )
        WT_dv = WT(CoeffOper3D('-', d, v))
        u = np.clip(WT_dv - (2.0 * f - 1.0) / mu, 0.0, 1.0)

        # 更新 d (各向同性软阈值)
        Wu = W(u)
        Wu_plus_v = CoeffOper3D('+', Wu, v)
        d = CoeffOper3D('vs', Wu_plus_v, tau_cell)

        # Bregman / scaled dual 更新
        v = CoeffOper3D('+', v, CoeffOper3D('-', Wu, d))

        rel_chg = np.linalg.norm(u - u_prev) / (np.linalg.norm(u_prev) + 1e-12)
        print(f"  [Iter {k:04d}/{max_iter}] relative_change={rel_chg:.6e} "
              f"| fraction(u<=0.5)={np.mean(u <= gamma):.4f} "
              f"| time={time.time()-iter_t0:.1f}s")

        if k >= 5 and rel_chg < tol:
            print(f"  --> Converged at iteration {k}"
                    f"(Total {time.time()-t_start:.1f}s)")
            break
        
    else:
        print(f"  [Warning] max_iter={max_iter} reached before tolerance {tol:g}")

    return u

# 三维等值面提取
def render_isosurface_to_ax(ax, volume, level=0.5,
                            view_elev=10, view_azim=-90,
                            flip_z=False, flip_x=False, axis_order=(0, 2, 1),
                            title="", dataset=None):
    """提取渲染 0.5 等值面"""
    verts, faces, _, _ = marching_cubes(volume, level=level)
    verts = verts[:, list(axis_order)]

    verts = verts - np.mean(verts, axis=0)
    max_span = np.max(np.ptp(verts, axis=0)) + 1e-8
    verts = verts / max_span

    if flip_z:
        verts[:, 2] = -verts[:, 2]
    if flip_x:
        verts[:, 0] = -verts[:, 0]

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12)

    # 漫反射光照
    el_rad, az_rad = np.deg2rad(view_elev + 20), np.deg2rad(view_azim - 15)
    light_dir = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])
    intensity = np.abs(np.dot(face_normals, light_dir))
    intensity = 0.32 + 0.68 * intensity

    # 斯坦福金黄
    base_color = np.array([1.0, 0.78, 0.08])
    face_colors = np.clip(intensity[:, None] * base_color[None, :], 0.0, 1.0)

    mesh = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolor='none', linewidths=0.0)
    ax.add_collection3d(mesh)

    axis_limit = 0.65 if dataset == "statuette" else 0.45
    
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', y=0.95)

def reproduce_figure4(data_dir="Data",
                      output_dir="output_hw5",
                      datasets=("cndragon", "lucy", "statuette"),
                      alpha=500.0, mu=0.04,
                      max_iter=35, tol=5e-4):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    model_pose = {
        "cndragon":  {"elev": 10, "azim": -90, "flip_z": False, "flip_x": False, "order": (0, 2, 1)},
        "lucy":      {"elev": 10, "azim":  90, "flip_z": False, "flip_x": False, "order": (0, 1, 2)},
        "statuette": {"elev": 10, "azim":  90, "flip_z": False, "flip_x": False, "order": (0, 2, 1)},
    }

    for name in datasets:
        pose = model_pose.get(name, {"elev": 10, "azim": -90, "flip_z": False, "flip_x": False, "order": (0, 2, 1)})
        mat_path = os.path.join(data_dir, f"{name}.mat")
        if not os.path.exists(mat_path):
            print(f"[Warning] {mat_path} not found, skipping {name}...")
            continue

        cache_path = os.path.join(output_dir, f"{name}_u_rec_v.npy")

        if os.path.exists(cache_path):
            print(f"Found cache '{cache_path}', loading directly...")
            _, f = load_surface_mat(mat_path)
            u_rec = np.load(cache_path)
        else:
            print(f"Reconstructing 3D Surface: {name}")
            phi, f = load_surface_mat(mat_path)
            u_rec = solve_surface_reconstruction_3d(
                phi, f, alpha=alpha, mu=mu, gamma=0.5, frame=1, Level=1, max_iter=max_iter, tol=tol
            )
            np.save(cache_path, u_rec)
            print(f"  --> Saved cache to {cache_path}")

        if u_rec.shape != f.shape:
            raise ValueError(f"缓存形状错误：{u_rec.shape} != {f.shape}")
        delta = np.mean(np.abs(u_rec - (1.0 - f)))
        changed = np.mean((u_rec <= 0.5) != (f >= 0.5))
        print(f"  [Check] mean(|u-(1-f)|)={delta:.6g}, "
                f"binary surface-side change fraction={changed:.4%}")
        results[name] = (f, u_rec)
        pose = model_pose.get(name, {"elev": 10, "azim": -90, "flip_z": False, "flip_x": False, "order": (0, 2, 1)})

        fig_single = plt.figure(figsize=(10, 5), dpi=300)
        ax1 = fig_single.add_subplot(1, 2, 1, projection='3d')
        render_isosurface_to_ax(
            ax1, f, level=0.5,
            view_elev=pose["elev"], view_azim=pose["azim"],
            flip_z=pose["flip_z"], flip_x=pose["flip_x"], axis_order=pose["order"],
            title=f"{name}: Initial Surface (f = 0.5)", dataset=name
        )
        ax2 = fig_single.add_subplot(1, 2, 2, projection='3d')
        render_isosurface_to_ax(
            ax2, u_rec, level=0.5,
            view_elev=pose["elev"], view_azim=pose["azim"],
            flip_z=pose["flip_z"], flip_x=pose["flip_x"], axis_order=pose["order"],
            title=f"{name}: Reconstructed Surface (u = 0.5)", dataset=name
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"surf3d_{name}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_single)

    if len(results) > 0:
        cols = len(results)
        fig_all = plt.figure(figsize=(4.5 * cols, 8.5), dpi=300)
        for idx, (name, (f_vol, u_vol)) in enumerate(results.items()):
            pose = model_pose.get(name, {"elev": 10, "azim": -90, "flip_z": False, "flip_x": False, "order": (0, 2, 1)})

            ax_top = fig_all.add_subplot(2, cols, idx + 1, projection='3d')
            render_isosurface_to_ax(
                ax_top, f_vol, level=0.5,
                view_elev=pose["elev"], view_azim=pose["azim"],
                flip_z=pose["flip_z"], flip_x=pose["flip_x"], axis_order=pose["order"],
                title=f"Initial: {name}", dataset=name
            )
            ax_bot = fig_all.add_subplot(2, cols, cols + idx + 1, projection='3d')
            render_isosurface_to_ax(
                ax_bot, u_vol, level=0.5,
                view_elev=pose["elev"], view_azim=pose["azim"],
                flip_z=pose["flip_z"], flip_x=pose["flip_x"], axis_order=pose["order"],
                title=f"Reconstructed: {name}", dataset=name
            )

        plt.tight_layout()
        fig4_path = os.path.join(output_dir, "Figure4_Reproduction.png")
        plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)
        print(f"\n[Success] Reproduced Figure 4 saved to: {fig4_path}")