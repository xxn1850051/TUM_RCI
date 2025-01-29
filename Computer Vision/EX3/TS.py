import numpy as np

def read_data():
    a=[]
    for i in range(8):
        a.append(np.array([float(j) for j in input().split()]))
    return a

def Gram_shmidet(A):
    Q = np.zeros_like(A) # N*N
    cnt = 0

    for a in A.T:
        u = np.copy(a)
        for i in range(0,cnt):
            u -= np.dot(np.dot(Q[:,i].T,a),Q[:,i]) #
        e = u/np.linalg.norm(u)
        Q[:,cnt]=e
        cnt+=1
    R = np.dot(Q.T,A)

    return  (Q,R)

def rq(A):
    '''Implement rq decomposition using QR decomposition

    From Wikipedia,
     The RQ decomposition transforms a matrix A into the product of an upper triangular matrix R (also known as right-triangular) and an orthogonal matrix Q. The only difference from QR decomposition is the order of these matrices.
     QR decomposition is Gram-Schmidt orthogonalization of columns of A, started from the first column.
     RQ decomposition is Gram-Schmidt orthogonalization of rows of A, started from the last row.
    '''
    A = np.asarray(A)

    # A = (A.T).T = (Q'R').T = R'.T * Q'.T

    # Reverse the cols
    reversed_A = A.T[:,::-1]

    # Make rows into column, then find QR
    Q, R = Gram_shmidet(reversed_A)

    #) The returned R is flipped updown, left right of R
    R[:,:] = R[::-1,::-1]

    # The returned Q is the flipped left-right of  Q
    Q[:, :] = Q[:, ::-1]

    return R.T, Q.T

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
uv = np.array([[634.926, 379.78],
               [604.775, 408.441],
               [597.474, 355.542],
               [567.102, 386.328],
               [639.252, 405.609],
               [606.225, 436.003],
               [598.333, 380.358],
               [565.039, 413.205]])
# uv = np.zeros((8, 2))
# for i in range(8):
#     row = input()
#     uv[i] = [float(val) for val in row.split()]

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

K,R = rq(P[:,:3])
h = P[:,-1].reshape(3, 1)
t = np.linalg.inv(K) @ h
# t = np.dot(np.linalg.inv(K),P[:,3])
K = K / K[2, 2]

# # 对P进行QR分解，得到投影矩阵P和变换矩阵K
# invH = np.linalg.inv(P[:, :3])
# tranR, invK = np.linalg.qr(invH)
# K = np.linalg.inv(invK)
# K = K/K[2,2]
# R = np.transpose(tranR)

# Rz = [[-1, 0, 0],
#       [0, -1, 0],
#       [0, 0, 1]]
# diagnal_negative_exist = False

# # for i in range(3):
# #     if K[i][i] <= 0:
# #         Rz[i][i] = -1
# #         diagnal_negative_exist = True

# # def is_diagonal_positive(matrix):
# #     for i in range(3):
# #         if matrix[i][i] <= 0:  # 判断对角线元素是否小于等于0
# #             Rz[i][i] = -1
# #             return False
# #     return True

# # a = not is_diagonal_positive(K)

# # if diagnal_negative_exist:
# K = K @ Rz
# R = Rz @ R

# # T = np.diag(np.sign(np.diag(K)))
# # K = K @ T
# # R = T @ R
# h = P[:,-1].reshape(3, 1)
# # t = invH @ np.transpose(P[:, 3])
# # t = np.linalg.inv(K) @ np.transpose(P[:, 3])
# t = np.linalg.inv(K) @ h

RT = np.concatenate((R, t), axis=1)

# print(K.astype(int))
rounded_K = []
for row in K:
    rounded_row = [round(element) for element in row]
    rounded_K.append(rounded_row)

rounded_K = [[round(element) for element in row] for row in K]

# for row in rounded_K:
#     print(row)

# printK = np.asarray(rounded_K)
# print(printK)

# for row in RT:
#     print(row)

# output data
outputstr = ""
for row in rounded_K:
    for item in row:
        outputstr += str(item)+" "
    outputstr +="\n"

for row in RT:
    for item in row:
        outputstr += str(item)+" "
    outputstr +="\n"

print(outputstr)

