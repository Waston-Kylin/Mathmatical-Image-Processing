import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

os.makedirs("image", exist_ok=True)
size = 256
Y, X = np.mgrid[0:size, 0:size]

# 课件 Page 64 shapes.png
img_shapes = np.ones((size, size)) * 0.75
# 圆环
r_c = np.sqrt((Y - size*0.32)**2 + (X - size*0.28)**2)
img_shapes[(r_c <= size*0.18) & (r_c >= size*0.07)] = 0.2
# 三角形
tri = (Y >= size*0.20) & (Y <= size*0.48) & (X >= size*0.55) & (X <= size*0.85) & ((X - size*0.55) >= (Y - size*0.20))
img_shapes[tri] = 0.2
# 凹槽拱形
rect = (Y >= size*0.56) & (Y <= size*0.86) & (X >= size*0.28) & (X <= size*0.72)
arch = np.sqrt((Y - size*0.80)**2 + (X - size*0.50)**2) <= size*0.14
img_shapes[rect & (~arch)] = 0.2
plt.imsave("image/shapes.png", img_shapes, cmap='gray')

# 无梯度的弱边缘图 weak_edge.png
dist = np.sqrt((Y - size*0.5)**2 + (X - size*0.5)**2)
img_weak = 0.2 + 0.65 * np.exp(-(dist / (size*0.25))**2)
plt.imsave("image/weak_edge.png", img_weak, cmap='gray')

# 虚幻轮廓图 kanizsa.png (Page 45-46)
img_kanizsa = np.ones((size, size)) * 0.8
r = size * 0.16
for cy, cx in [(size*0.3, size*0.3), (size*0.3, size*0.7), (size*0.7, size*0.3), (size*0.7, size*0.7)]:
    img_kanizsa[np.sqrt((Y - cy)**2 + (X - cx)**2) <= r] = 0.2
mid, bh = size * 0.5, size * 0.2
img_kanizsa[(Y >= mid-bh) & (Y <= mid+bh) & (X >= mid-bh) & (X <= mid+bh)] = 0.8
plt.imsave("image/kanizsa.png", img_kanizsa, cmap='gray')

# skimage 硬币图 coins.png
plt.imsave("image/coins.png", data.coins() / 255.0, cmap='gray')
