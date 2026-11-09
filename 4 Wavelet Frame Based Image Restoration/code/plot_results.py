import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

frames = ['Frame 0\n(Haar)', 'Frame 1\n(Linear)', 'Frame 3\n(Cubic)']
x = np.arange(len(frames))

# Noise = 0.01 (Origin: PSNR=34.441, SSIM=0.9323 | TV: PSNR=35.360, SSIM=0.8994, k*=2)
n01 = {
    'orig_psnr': 34.441, 'orig_ssim': 0.9323,
    'tv_psnr': 35.360,   'tv_ssim': 0.8994,   'tv_k': 2,
    'ana_s_psnr':  [35.417, 38.060, 38.377], 'ana_s_ssim':  [0.8983, 0.9608, 0.9824], 'ana_s_k':  [43, 149, 150],
    'ana_vs_psnr': [33.672, 33.966, 34.024], 'ana_vs_ssim': [0.8384, 0.8517, 0.8597], 'ana_vs_k': [3, 3, 3],
    'bal_s_psnr':  [36.071, 38.086, 38.369], 'bal_s_ssim':  [0.9177, 0.9607, 0.9812], 'bal_s_k':  [10, 24, 23],
    'bal_vs_psnr': [35.433, 35.401, 35.335], 'bal_vs_ssim': [0.9110, 0.9105, 0.9142], 'bal_vs_k': [6, 6, 5],
}

# Noise = 0.02 (Origin: PSNR=31.833, SSIM=0.8085 | TV: PSNR=30.017, SSIM=0.6885, k*=2)
n02 = {
    'orig_psnr': 31.833, 'orig_ssim': 0.8085,
    'tv_psnr': 30.017,   'tv_ssim': 0.6885,   'tv_k': 2,
    'ana_s_psnr':  [26.991, 28.516, 32.744], 'ana_s_ssim':  [0.5382, 0.6195, 0.8235], 'ana_s_k':  [2, 100, 150],
    'ana_vs_psnr': [26.078, 26.278, 26.392], 'ana_vs_ssim': [0.4965, 0.5073, 0.5151], 'ana_vs_k': [2, 2, 2],
    'bal_s_psnr':  [31.814, 31.894, 32.671], 'bal_s_ssim':  [0.7978, 0.8025, 0.8175], 'bal_s_k':  [1, 1, 13],
    'bal_vs_psnr': [31.764, 31.763, 31.756], 'bal_vs_ssim': [0.7950, 0.7950, 0.7947], 'bal_vs_k': [1, 1, 1],
}

# PSNR
def plot_fig1_psnr_bars():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    width = 0.18

    configs = [
        (axes[0], n01, 'Noise Level $\\sigma_1 = 0.01$ (aircraft.jpg)', (32.5, 39.8), 'upper left'),
        (axes[1], n02, 'Noise Level $\\sigma_1 = 0.02$ (aircraft.jpg)', (24.5, 34.6), 'upper left')
    ]

    for ax, d, title, ylim, leg_loc in configs:
        b1 = ax.bar(x - 1.5*width, d['ana_s_psnr'],  width, label='Analysis (thresh=s)',  color='#1f77b4', alpha=0.9)
        b2 = ax.bar(x - 0.5*width, d['bal_s_psnr'],  width, label='Balanced (thresh=s)',  color='#ff7f0e', alpha=0.9)
        b3 = ax.bar(x + 0.5*width, d['ana_vs_psnr'], width, label='Analysis (thresh=vs)', color='#aec7e8', edgecolor='#1f77b4', hatch='//')
        b4 = ax.bar(x + 1.5*width, d['bal_vs_psnr'], width, label='Balanced (thresh=vs)', color='#ffbb78', edgecolor='#ff7f0e', hatch='//')

        # Origin 与 TV
        ax.axhline(d['orig_psnr'], color='#7f7f7f', linestyle=':', linewidth=1.8, label=f"Degraded Origin ({d['orig_psnr']:.2f} dB)")
        ax.axhline(d['tv_psnr'],   color='#d62728', linestyle='--', linewidth=1.8, label=f"TV Baseline ({d['tv_psnr']:.2f} dB)")

        for bars in [b1, b2, b3, b4]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(frames, fontsize=10.5)
        ax.set_ylabel('Max PSNR (dB)', fontsize=11)
        ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc=leg_loc, fontsize=8.5, ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../doc/figure/psnr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# SSIM 随小波阶数
def plot_fig2_ssim_trends():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    
    # sigma1 = 0.01
    ax = axes[0]
    d = n01
    ax.plot(x, d['ana_s_ssim'],  'o-',  color='#1f77b4', lw=2.2, ms=7, label='Analysis (thresh=s)')
    ax.plot(x, d['bal_s_ssim'],  's-',  color='#ff7f0e', lw=2.2, ms=7, label='Balanced (thresh=s)')
    ax.plot(x, d['ana_vs_ssim'], '^--', color='#5b9bd5', lw=1.6, ms=6, label='Analysis (thresh=vs)')
    ax.plot(x, d['bal_vs_ssim'], 'D--', color='#f4b183', lw=1.6, ms=6, label='Balanced (thresh=vs)')

    ax.axhline(d['orig_ssim'], color='#7f7f7f', linestyle=':', lw=1.6, label=f"Origin ({d['orig_ssim']:.4f})")
    ax.axhline(d['tv_ssim'],   color='#d62728', linestyle='-.', lw=1.6, label=f"TV Baseline ({d['tv_ssim']:.4f})")

    for i in range(len(x)):
        # Analysis (thresh=s)
        ax.annotate(f"{d['ana_s_ssim'][i]:.4f}", (x[i], d['ana_s_ssim'][i]),
                    textcoords="offset points", xytext=(-20 if i!=1 else 0, 7 if i!=1 else 8),
                    fontsize=8, color='#1f77b4', fontweight='bold')
        # Balanced (thresh=s)
        ax.annotate(f"{d['bal_s_ssim'][i]:.4f}", (x[i], d['bal_s_ssim'][i]),
                    textcoords="offset points", xytext=(8, -13),
                    fontsize=8, color='#d95f02', fontweight='bold')
        # Analysis (thresh=vs)
        ax.annotate(f"{d['ana_vs_ssim'][i]:.4f}", (x[i], d['ana_vs_ssim'][i]),
                    textcoords="offset points", xytext=(0, -12), ha='center',
                    fontsize=7.5, color='#2f5597')
        # Balanced (thresh=vs)
        ax.annotate(f"{d['bal_vs_ssim'][i]:.4f}", (x[i], d['bal_vs_ssim'][i]),
                    textcoords="offset points", xytext=(0, 6 if i!=0 else 7), ha='center',
                    fontsize=7.5, color='#c55a11')

    ax.set_title('SSIM vs. Framelet Order ($\\sigma_1 = 0.01$)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(frames, fontsize=10.5)
    ax.set_ylabel('SSIM (at Best $k^*$)', fontsize=11)
    ax.set_ylim(0.815, 1.015)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)

    # sigma1 = 0.02
    ax = axes[1]
    d = n02
    ax.plot(x, d['ana_s_ssim'],  'o-',  color='#1f77b4', lw=2.2, ms=7, label='Analysis (thresh=s)')
    ax.plot(x, d['bal_s_ssim'],  's-',  color='#ff7f0e', lw=2.2, ms=7, label='Balanced (thresh=s)')
    ax.plot(x, d['ana_vs_ssim'], '^--', color='#5b9bd5', lw=1.6, ms=6, label='Analysis (thresh=vs)')
    ax.plot(x, d['bal_vs_ssim'], 'D--', color='#f4b183', lw=1.6, ms=6, label='Balanced (thresh=vs)')

    ax.axhline(d['orig_ssim'], color='#7f7f7f', linestyle=':', lw=1.6, label=f"Origin ({d['orig_ssim']:.4f})")
    ax.axhline(d['tv_ssim'],   color='#d62728', linestyle='-.', lw=1.6, label=f"TV Baseline ({d['tv_ssim']:.4f})")

    for i in range(len(x)):
        # Analysis (thresh=s)
        ax.annotate(f"{d['ana_s_ssim'][i]:.4f}", (x[i], d['ana_s_ssim'][i]),
                    textcoords="offset points", xytext=(0, 7), ha='center',
                    fontsize=8, color='#1f77b4', fontweight='bold')
        # Balanced (thresh=s)
        ax.annotate(f"{d['bal_s_ssim'][i]:.4f}", (x[i], d['bal_s_ssim'][i]),
                    textcoords="offset points", xytext=(0, 7 if i!=2 else -14), ha='center',
                    fontsize=8, color='#d95f02', fontweight='bold')
        # Analysis (thresh=vs)
        ax.annotate(f"{d['ana_vs_ssim'][i]:.4f}", (x[i], d['ana_vs_ssim'][i]),
                    textcoords="offset points", xytext=(0, -13), ha='center',
                    fontsize=7.5, color='#2f5597')
        # Balanced (thresh=vs)
        ax.annotate(f"{d['bal_vs_ssim'][i]:.4f}", (x[i], d['bal_vs_ssim'][i]),
                    textcoords="offset points", xytext=(0, -13 if i!=2 else 6), ha='center',
                    fontsize=7.5, color='#c55a11')

    ax.set_title('SSIM vs. Framelet Order ($\\sigma_1 = 0.02$)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(frames, fontsize=10.5)
    ax.set_ylabel('SSIM (at Best $k^*$)', fontsize=11)
    ax.set_ylim(0.46, 0.88)
    ax.grid(True, linestyle=':', alpha=0.6)
    # 右子图图例向上挪动到右侧大片空白区 (y 约 0.6 附近)
    ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.38), fontsize=8.2, framealpha=0.92)

    plt.tight_layout()
    plt.savefig('../doc/figure/ssim_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

# 收敛迭代步数 k*
def plot_fig3_convergence_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax1 = axes[0]
    w = 0.32
    b_ana = ax1.bar(x - w/2, n01['ana_s_k'], w, color='#1f77b4', alpha=0.88, label='Analysis-ADMM\n(thresh=s)')
    b_bal = ax1.bar(x + w/2, n01['bal_s_k'], w, color='#2ca02c', alpha=0.88, label='Balanced-APG\n(thresh=s)')

    for idx, bar in enumerate(b_ana):
        h = bar.get_height()
        lbl = '>150' if idx == 2 else f'{int(h)}'
        ax1.annotate(f'{lbl} iters\n({n01["ana_s_psnr"][idx]:.2f} dB)',
                     xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1f77b4')

    for idx, bar in enumerate(b_bal):
        h = bar.get_height()
        ax1.annotate(f'{int(h)} iters\n({n01["bal_s_psnr"][idx]:.2f} dB)',
                     xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2ca02c')

    ax1.set_title('Convergence Speed: Optimal $k^*$ ($\\sigma_1 = 0.01$)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frames, fontsize=10.5)
    ax1.set_ylabel('Iterations to Peak PSNR ($k^*$)', fontsize=11)
    ax1.set_ylim(0, 195)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9)

    # 散点
    ax2 = axes[1]
    ax2.scatter(n01['ana_s_psnr'],  n01['ana_s_ssim'],  s=110, c='#1f77b4', marker='o', label='Analysis-s ($\\sigma_1=0.01$)')
    ax2.scatter(n01['bal_s_psnr'],  n01['bal_s_ssim'],  s=110, c='#ff7f0e', marker='s', label='Balanced-s ($\\sigma_1=0.01$)')
    ax2.scatter(n01['ana_vs_psnr'], n01['ana_vs_ssim'], s=80,  c='#1f77b4', marker='^', alpha=0.5, label='Analysis-vs ($\\sigma_1=0.01$)')
    ax2.scatter(n01['bal_vs_psnr'], n01['bal_vs_ssim'], s=80,  c='#ff7f0e', marker='v', alpha=0.5, label='Balanced-vs ($\\sigma_1=0.01$)')
    ax2.scatter([n01['tv_psnr']],   [n01['tv_ssim']],   s=130, c='#d62728', marker='*', label='TV Baseline ($\\sigma_1=0.01$)')

    for i, txt in enumerate(['F=0', 'F=1', 'F=3']):
        ax2.annotate(txt, (n01['ana_s_psnr'][i], n01['ana_s_ssim'][i]), xytext=(-14, 6), textcoords='offset points', fontsize=8.5)

    ax2.set_title('PSNR vs. SSIM Trade-off ($\\sigma_1 = 0.01$)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Max PSNR (dB)', fontsize=11)
    ax2.set_ylabel('SSIM', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    plt.savefig('../doc/figure/efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_fig1_psnr_bars()
    plot_fig2_ssim_trends()
    plot_fig3_convergence_efficiency()