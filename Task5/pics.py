import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

# 进程数量
processes = np.array([1, 2, 4, 8, 16])

# 矩阵规模
sizes = np.array([128, 256, 512, 1024, 2048])

# 时间
times = np.array([
    [0.014097, 0.070119, 0.543809, 4.048968, 82.492855],
    [0.017437, 0.091779, 0.627493, 3.168582, 48.964278],
    [0.020966, 0.111330, 0.547351, 3.426131, 27.645359],
    [0.038651, 0.156202, 0.697604, 3.350200, 33.643892],
    [0.054763, 0.279289, 5.988570, 6.243032, 41.413419]
])

# 创建一个新的 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用 meshgrid 创建网格
X, Y = np.meshgrid(sizes, processes)

# 绘制曲面
surf = ax.plot_surface(X, Y, times, cmap='coolwarm')

# 添加颜色条
fig.colorbar(surf)

# 设置标签
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Processes')
ax.set_zlabel('Time (s)')
ax.view_init(30, 120)

# 设置x和y轴的刻度为给定的数据值
ax.set_xticks(sizes)
ax.set_yticks(processes)

# 显示图形
plt.savefig('pics.png', dpi=300)
