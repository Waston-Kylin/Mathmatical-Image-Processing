import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 初值敏感度 / 分片对比度
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

init_labels = ['Box Init', 'Circle Init', 'Corner-Circle Init']
x = np.arange(len(init_labels))
width = 0.26

cv_shapes = [0.3378, 0.2392, 0.1017]
convex_obj_shapes = [1 - 0.7839, 1 - 0.7839, 1 - 0.7839] # 互补暗色目标面积 0.2161

b1 = axes[0].bar(x - width/2, cv_shapes, width, label='Original CV (Dark Object Area)', color='#1f77b4', alpha=0.88)
b2 = axes[0].bar(x + width/2, convex_obj_shapes, width, label='Convex CV (Complementary Object Area $1-A$)', color='#2ca02c', alpha=0.88)
axes[0].axhline(0.2163, color='#d62728', linestyle='--', linewidth=1.8, label='Ground Truth Object Area (0.2163)')

axes[0].set_ylabel('Segmented Object Area Ratio', fontsize=11)
axes[0].set_title('(a) Initialization Sensitivity on Clean "shapes"', fontsize=12, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(init_labels, fontsize=10.5); axes[0].set_ylim(0, 0.42)
axes[0].grid(axis='y', linestyle=':', alpha=0.6); axes[0].legend(loc='upper right', fontsize=9.5)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

img_labels = ['shapes', 'weak_edge', 'kanizsa', 'coins']
x2 = np.arange(len(img_labels))
w2 = 0.25

gac_contrast = [abs(0.1469 - 0.8538), abs(0.9382 - 0.1731), abs(0.4183 - 0.9972), abs(0.4946 - 0.3530)]
cv_contrast = [abs(0.0961 - 0.9998), abs(0.5164 - 0.0571), abs(0.3050 - 0.9999), abs(0.3632 - 0.3934)]
convex_contrast = [abs(0.9997 - 0.0000), abs(0.5476 - 0.0731), abs(0.9998 - 0.0002), abs(0.5183 - 0.2586)]

c1_b = axes[1].bar(x2 - w2, gac_contrast, w2, label='GAC', color='#d62728', alpha=0.88)
c2_b = axes[1].bar(x2, cv_contrast, w2, label='Original CV', color='#1f77b4', alpha=0.88)
c3_b = axes[1].bar(x2 + w2, convex_contrast, w2, label='Convex CV', color='#2ca02c', alpha=0.88)

axes[1].set_ylabel('Partition Contrast $|c_1 - c_2|$', fontsize=11)
axes[1].set_title('(b) Region Mean Contrast $|c_1 - c_2|$ Across Datasets (Circle Init)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x2); axes[1].set_xticklabels(img_labels, fontsize=10.5); axes[1].set_ylim(0, 1.15)
axes[1].grid(axis='y', linestyle=':', alpha=0.6); axes[1].legend(loc='upper right', fontsize=9.5)

for bars in [c1_b, c2_b, c3_b]:
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../doc/plot/hw3_fig1_clean_and_init.png', dpi=300, bbox_inches='tight')
plt.close()

# 实验数据
blur_levels = ['Clean ($\sigma_b=0$)', 'Blur $\sigma_b=2$', 'Blur $\sigma_b=4$']
xb = np.arange(len(blur_levels))
gac_box_blur = [0.2287, 0.2066, 0.0851]
gac_cir_blur = [0.0993, 0.0913, 0.0103]
cv_box_blur = [0.3378, 0.3528, 0.3677]
cv_cir_blur = [0.2392, 0.2680, 0.3011]
convex_obj_blur = [1 - 0.7839, 1 - 0.7778, 1 - 0.7696]

noise_levels = ['Clean ($\sigma_n=0$)', 'Noise $\sigma_n=0.01$', 'Noise $\sigma_n=0.02$']
xn = np.arange(len(noise_levels))
gac_box_noise = [0.5044, 0.5043, 0.5047]
gac_cir_noise = [0.1962, 0.1962, 0.1965]
cv_box_noise = [0.8072, 0.8072, 0.8073]
cv_cir_noise = [0.4176, 0.4177, 0.4177]
convex_noise = [0.4705, 0.4760, 0.4777]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Blur Sensitivity
axes[0].plot(xb, gac_box_blur, 'o-', color='#d62728', linewidth=2, markersize=7, label='GAC (Box Init)')
axes[0].plot(xb, gac_cir_blur, 'o--', color='#e06666', linewidth=1.8, markersize=6, label='GAC (Circle Init)')
axes[0].plot(xb, cv_box_blur, 's-', color='#1f77b4', linewidth=2, markersize=7, label='Original CV (Box Init)')
axes[0].plot(xb, cv_cir_blur, 's--', color='#6fa8dc', linewidth=1.8, markersize=6, label='Original CV (Circle Init)')
axes[0].plot(xb, convex_obj_blur, '^-', color='#2ca02c', linewidth=2.5, markersize=8, label='Convex CV ($1-A$, Inits Identical)')

axes[0].set_ylabel('Effective Object Area Ratio', fontsize=11)
axes[0].set_title('(a) Blur Sensitivity on "shapes" ($\sigma_b = 0, 2, 4$)', fontsize=12, fontweight='bold')
axes[0].set_xticks(xb)
axes[0].set_xticklabels(blur_levels, fontsize=10.5)
axes[0].set_ylim(0, 0.48)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.92, fontsize=8.8)

# (b) Noise Stability & Init Gap
axes[1].plot(xn, cv_box_noise, 's-', color='#1f77b4', linewidth=2, markersize=7, label='Original CV (Box Init)')
axes[1].plot(xn, gac_box_noise, 'o-', color='#d62728', linewidth=2, markersize=7, label='GAC (Box Init)')
axes[1].plot(xn, convex_noise, '^-', color='#2ca02c', linewidth=2.5, markersize=8, label='Convex CV (Inits Identical)')
axes[1].plot(xn, cv_cir_noise, 's--', color='#6fa8dc', linewidth=1.8, markersize=6, label='Original CV (Circle Init)')
axes[1].plot(xn, gac_cir_noise, 'o--', color='#e06666', linewidth=1.8, markersize=6, label='GAC (Circle Init)')

axes[1].set_ylabel('Segmented Area Ratio', fontsize=11)
axes[1].set_title('(b) Noise Stability & Init Gap on "coins" ($\sigma_n = 0, 0.01, 0.02$)', fontsize=12, fontweight='bold')
axes[1].set_xticks(xn)
axes[1].set_xticklabels(noise_levels, fontsize=10.5)
axes[1].set_ylim(0.12, 0.92)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='center', bbox_to_anchor=(0.5, 0.66), ncol=2, framealpha=0.95, fontsize=8.5)

plt.tight_layout()
plt.savefig('../doc/plot/hw3_fig2_blur_and_noise.png', dpi=300, bbox_inches='tight')
plt.close()
