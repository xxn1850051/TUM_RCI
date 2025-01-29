import numpy as np

# 世界坐标
xyz = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]])

# 图像坐标
# uv = np.array([[634.926, 379.78],
#                [604.775, 408.441],
#                [597.474, 355.542],
#                [567.102, 386.328],
#                [639.252, 405.609],
#                [606.225, 436.003],
#                [598.333, 380.358],
#                [565.039, 413.205]])
uv = np.zeros((8, 2))
for i in range(8):
    row = input()
    uv[i] = [float(val) for val in row.split()]

# 将uv和xyz转换为齐次坐标
uv_h = np.hstack((uv, np.ones((8, 1))))
xyz_h = np.hstack((xyz, np.ones((8, 1))))

# 构建A矩阵
A = []
for i in range(8):
    X, Y, Z, W = xyz_h[i]
    u, v, w = uv_h[i]
    A.append([-X, -Y, -Z, -W, 0, 0, 0, 0, u*X, u*Y, u*Z, u*W])
    A.append([0, 0, 0, 0, -X, -Y, -Z, -W, v*X, v*Y, v*Z, v*W])
A = np.array(A)

# 进行SVD分解
_, _, V = np.linalg.svd(A)

# 取最后一列，即V的最小奇异值对应的列向量
P = V[-1].reshape(3, 4)
# if np.linalg.det(P[:,:3])<0:   # det(A)=det(RQ) = det(Q) with det(R)=1, since we want QeSO3, det(Q) has to be 1s
#     P = -P
if P[2][3] < 0:
    P = -P

# 对P进行QR分解，得到投影矩阵P和变换矩阵K
invH = np.linalg.inv(P[:, :3])
tranR, invK = np.linalg.qr(invH)
K = np.linalg.inv(invK)
R = np.transpose(tranR)

T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R
h = P[:,-1].reshape(3, 1)
t = np.linalg.inv(K) @ h

K = K/K[2,2]

RT = np.concatenate((R, t), axis=1)

rounded_K = []
for row in K:
    rounded_row = [round(element) for element in row]
    rounded_K.append(rounded_row)

rounded_K = [[round(element) for element in row] for row in K]

# 输出数据
for row in rounded_K:
    print(*row, sep='\t')

for row in RT:
    print(*row, sep='\t')

