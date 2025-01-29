# import numpy as np

# def null_space(A):
#     Q, _ = np.linalg.qr(A, mode='reduced')
#     return Q[:, Q.shape[1]:]

# # Step 1: Reading Input
# m = int(input().strip())
# nA = int(input().strip())
# A = []
# for i in range(nA):
#     name, *coords = input().strip().split()
#     A.append([float(c) for c in coords])
# nB = int(input().strip())
# B = []
# for i in range(nB):
#     name, *coords = input().strip().split()
#     B.append([float(c) for c in coords])

# A = np.array(A)
# A = A.T
# B = np.array(B)
# B = B.T

# # Step 2: Compute matrix C and vector c
# dA = np.mean(A, axis=1).reshape(-1, 1)
# dB = np.mean(B, axis=1).reshape(-1, 1)

# A = A - dA
# B = B - dB

# A_null = null_space(A.T).T
# B_null = null_space(B.T).T

# # Construct matrix C by concatenating the null spaces
# max_size = max(A_null.shape[1], B_null.shape[1])
# C = np.concatenate((np.hstack((A_null, np.zeros((A_null.shape[0], max_size - A_null.shape[1])))),
#                     np.hstack((B_null, np.zeros((B_null.shape[0], max_size - B_null.shape[1]))))), axis=0)

# # Construct vector c with appropriate dimensions
# c = np.zeros((C.shape[0], 1))

# x_sol = np.linalg.lstsq(C, c, rcond=None)
# c_pred = np.dot(C, x_sol[0])

# if np.sum(np.abs(c - c_pred)) <= 1e-13:
#     output_str = "Y"
#     for item in x_sol[0].squeeze():
#         output_str += " " + str(int(item))
#     print(output_str)
# else:
#     print("N")

import numpy as np

def null_space(A, rcond=None):
    #null space implementation copied from scipy
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def null_space_qr(A):
    Q, _ = np.linalg.qr(A, mode='reduced')
    return Q[:, Q.shape[1]:]

# def null_space_LU(A):
#     lu, piv = scipy.linalg.lu_factor(A)
#     m, n = A.shape
    
#     # Find the number of independent columns in the LU decomposition
#     rank = np.sum(np.abs(np.diag(lu)) > np.finfo(lu.dtype).eps * np.abs(lu).max())
    
#     # Construct the null space basis using LU decomposition
#     null_basis = np.zeros((n, n - rank))
    
#     for i in range(n - rank):
#         x = np.zeros(n)
#         x[i] = 1
#         null_basis[:, i] = scipy.linalg.lu_solve((lu, piv), A @ x)
    
#     return null_basis


def null_space_AL(A):
    m, n = A.shape

    # Compute the reduced row echelon form (rref) of A
    rref, pivots = compute_rref(A)
    
    # Determine the indices of the columns without pivots
    non_pivot_indices = np.setdiff1d(range(n), pivots)
    
    # Construct the null space basis
    null_basis = np.zeros((n, len(non_pivot_indices)))
    
    for i, idx in enumerate(non_pivot_indices):
        # Solve for each basis vector by setting the non-pivot variables to 1
        x = np.zeros(n)
        x[idx] = 1
        
        # Back-substitution to find the values of the pivot variables
        for row in range(m - 1, -1, -1):
            pivot_col = pivots[row]
            x[pivot_col] = (rref[row, -1] - rref[row, pivots].dot(x[pivots])) / rref[row, pivot_col]
        
        null_basis[:, i] = x
    
    return null_basis

def compute_rref(A):
    # Create a copy of A to avoid modifying the original matrix
    rref = A.copy()
    m, n = rref.shape
    
    # Initialize the pivot list
    pivots = []
    
    # Forward elimination
    for pivot_row in range(m):
        # Find the leftmost non-zero entry in the current row
        pivot_col = np.argmax(np.abs(rref[pivot_row:, pivot_row]))
        pivot_col += pivot_row  # Adjust for the current row index
        
        if np.abs(rref[pivot_col, pivot_row]) < np.finfo(rref.dtype).eps:
            # The pivot entry is approximately zero, continue to the next row
            continue
        
        # Swap the rows to move the pivot to the diagonal position
        rref[[pivot_row, pivot_col]] = rref[[pivot_col, pivot_row]]
        
        # Scale the pivot row to make the pivot entry equal to 1
        pivot_val = rref[pivot_row, pivot_row]
        rref[pivot_row] /= pivot_val
        
        # Subtract multiples of the pivot row from the rows below
        for row in range(pivot_row + 1, m):
            factor = rref[row, pivot_row]
            rref[row] -= factor * rref[pivot_row]
        
        # Save the column index of the pivot
        pivots.append(pivot_row)
    
    return rref, pivots




#读取参数
# m = int(input().strip())
# nA = int(input().strip())
# A = []
# for i in range(nA):
#     name, *coords = input().strip().split()
#     A.append([float(c) for c in coords])
# nB = int(input().strip())
# B = []
# for i in range(nB):
#     name, *coords = input().strip().split()
#     B.append([float(c) for c in coords])
m = int(input().strip())
nA = int(input().strip())
A = [list(map(float, input().strip().split()[1:])) for _ in range(nA)]
nB = int(input().strip())
B = [list(map(float, input().strip().split()[1:])) for _ in range(nB)]



A = np.array(A)
A = A.T
B = np.array(B)
B = B.T

C1=A
D=B


# 计算平均值
dA = np.mean(A, axis=1).reshape(-1,1)
dB = np.mean(B, axis=1).reshape(-1,1)

dC1 = np.mean(A, axis=1).reshape(-1,1)
dD = np.mean(B, axis=1).reshape(-1,1)

#仿射空间-->向量空间
E = A -dA
F = B -dB

A = A -dA
B = B -dB

C1 = C1 -dC1
D = D -dD



A_null_space = null_space(A.T).T
B_null_space = null_space(B.T).T

C1 = null_space_qr(C1.T).T
D = null_space_qr(D.T).T



# C = np.concatenate([A,B],axis=0)
# c =np.concatenate([np.dot(A,dA),np.dot(B,dB)])
C = np.vstack((A_null_space, B_null_space))
c = np.concatenate((A_null_space @ dA, B_null_space @ dB))



x_sol = np.linalg.lstsq(C,c,rcond=None)

# c_pred = np.dot(C,x_sol[0])
c_pred = C @ x_sol[0]



if np.sum(np.abs(c-c_pred))<=1e-13:
    output_str="Y"
    for item in x_sol[0].squeeze():
        output_str += " "+ str(item)
    print(output_str)
else:
    print("N")
