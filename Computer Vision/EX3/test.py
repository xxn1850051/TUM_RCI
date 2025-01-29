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
uv = np.array([[680.939, 348.744],
               [644.057, 331.705],
               [698.02, 309.226],
               [660.63, 292.7],
               [673.468, 346.138],
               [632.962, 327.41],
               [692.275, 302.48],
               [651.154, 284.377]])

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
if np.linalg.det(P[:,:3])<0:   # det(A)=det(RQ) = det(Q) with det(R)=1, since we want QeSO3, det(Q) has to be 1s
    P = -P

if P[2][3] < 0:
    P = -P
    
invH = np.linalg.inv(P[:, :3])
tranR, invK = np.linalg.qr(invH)
K = np.linalg.inv(invK)
R = np.transpose(tranR)

T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R
# print(R)
h = P[:,-1].reshape(3, 1)
# print(h)
t = np.linalg.inv(K) @ h
# print(t)

K = K/K[2,2]

print(K)
print(R)
print(t)


