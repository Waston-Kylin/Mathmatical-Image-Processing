import numpy as np
import matplotlib.pyplot as plt

images = ['shapes', 'weak_edge', 'kanizsa', 'coins']

# 无噪条件下的分割面积占比 (Clean, sigma = 0)
gac_clean = [0.3032, 0.0222, 0.3901, 0.3837]
cv_clean = [0.2163, 0.4998, 0.2397, 0.4999]
frame_clean = [0.2163, 0.3203, 0.2397, 0.4293]

# 含噪条件下的分割面积占比 (Gaussian noise, sigma = 0.02)
gac_noisy = [0.3027, 0.0244, 0.3900, 0.3846]
cv_noisy = [0.2163, 0.4988, 0.2397, 0.4994]
frame_noisy = [0.2163, 0.3195, 0.2397, 0.4378]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

x = np.arange(len(images))
width = 0.25

c_gac = '#4A90E2' # GAC
c_cv = '#E67E22' # Chan-Vese
c_frame = '#27AE60' # Wavelet Frame

# 无噪对比
rects1 = ax1.bar(x - width, gac_clean, width, label='GAC', color=c_gac, edgecolor='#2c3e50', linewidth=0.6, alpha=0.9)
rects2 = ax1.bar(x, cv_clean, width, label='Chan-Vese', color=c_cv, edgecolor='#2c3e50', linewidth=0.6, alpha=0.9)
rects3 = ax1.bar(x + width, frame_clean, width, label='Wavelet Frame', color=c_frame, edgecolor='#2c3e50', linewidth=0.6, alpha=0.95)

ax1.set_title('(a) Clean Condition ($\sigma = 0$)', fontsize=13, fontweight='bold', pad=12, color='#1F4E79')
ax1.set_ylabel('Segmented Area Ratio', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(images, fontsize=11, fontweight='bold')
ax1.set_ylim(0, 0.60)
ax1.grid(axis='y', linestyle='--', alpha=0.45)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

# 含噪对比
rects4 = ax2.bar(x - width, gac_noisy, width, label='GAC', color=c_gac, edgecolor='#2c3e50', linewidth=0.6, alpha=0.9)
rects5 = ax2.bar(x, cv_noisy, width, label='Chan-Vese', color=c_cv, edgecolor='#2c3e50', linewidth=0.6, alpha=0.9)
rects6 = ax2.bar(x + width, frame_noisy, width, label='Wavelet Frame', color=c_frame, edgecolor='#2c3e50', linewidth=0.6, alpha=0.95)

ax2.set_title('(b) Noisy Condition (Gaussian Noise $\sigma = 0.02$)', fontsize=13, fontweight='bold', pad=12, color='#1F4E79')
ax2.set_ylabel('Segmented Area Ratio', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(images, fontsize=11, fontweight='bold')
ax2.set_ylim(0, 0.60)
ax2.grid(axis='y', linestyle='--', alpha=0.45)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

def autolabel(rects, ax):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for r, a in [(rects1, ax1), (rects2, ax1), (rects3, ax1), (rects4, ax2), (rects5, ax2), (rects6, ax2)]:
    autolabel(r, a)

plt.suptitle('Quantitative Comparison of 2D Segmentation Models: GAC vs. Chan-Vese vs. Wavelet Tight Frame', 
             fontsize=14, fontweight='bold', y=0.98, color='#1F4E79')
plt.tight_layout()
plt.savefig('../doc/segmentation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()