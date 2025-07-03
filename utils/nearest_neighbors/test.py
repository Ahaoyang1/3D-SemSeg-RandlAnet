import numpy as np  # 导入numpy库并将其别名设置为np，用于处理数组和数值计算。
import lib.python.nearest_neighbors as nearest_neighbors  # 导入自定义模块nearest_neighbors，该模块包含了计算最近邻点的函数。
import time  # 导入time模块，用于计算程序的执行时间。

# 定义了三个变量，分别代表批量数据的大小、每个点云的点个数和最近邻点的数量。
batch_size = 16
num_points = 81920
K = 16

# 使用np.random.rand函数生成一个形状为(batch_size, num_points, 3)的随机浮点型数组，
# 并将其赋值给变量pc。这个数组表示批量大小个点云数据，每个点云由num_points个点组成，每个点有3个坐标轴。
pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

# nearest neighbours

# 记录当前时间，以便后面计算代码执行时间。
start = time.time()

# 调用自定义模块nearest_neighbors中的knn_batch函数，
# 给定点云数据pc、最近邻点数量K和omp=True参数表示使用并行计算。
# 将计算结果赋值给neigh_idx变量，存储了每个点的最近邻点的索引。
neigh_idx = nearest_neighbors.knn_batch(pc, pc, K, omp=True)

# 计算并打印代码执行的时间差，即计算最近邻点的时间。
print(time.time() - start)


