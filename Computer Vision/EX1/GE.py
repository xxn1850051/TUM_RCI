# # Importing NumPy Library
# import numpy as np

# n = int(input())
# matrix = []
# for i in range(n):
#     row = list(map(int,input().split()))
#     matrix.append(row)
# matrix = np.array(matrix)

# # # Reading order of matrix
# # n = int(input('Enter order of matrix: '))

# # Making numpy array of n x 2n size and initializing 
# # to zero for storing augmented matrix
# # a = np.zeros((n,2*n))

# # # Reading matrix coefficients
# # print('Enter Matrix Coefficients:')
# # for i in range(n):
# #     for j in range(n):
# #         a[i][j] = float(input( 'a['+str(i)+']['+ str(j)+']='))

# # # Augmenting Identity Matrix of Order n
# # for i in range(n):        
# #     for j in range(n):
# #         if i == j:
# #             a[i][j+n] = 1

# a = np.concatenate((matrix, np.identity(n)), axis=1)

# # Applying Guass Jordan Elimination
# for i in range(n):
#     if a[i][i] == 0.0:
#         for e in range(i+1,n):
#             if a[e][i] != 0:
#                 a[[i, e]] = a[[e, i]]
#                 print("S",i,e)
#                 break
    
#     # if i == n:
#     #     print("DEGENERATE")
#     #     break

#     if a[i][i] == 0.0:
#         continue
        
#     for j in range(n):
#         if i != j:
#             ratio = a[j][i]/a[i][i]

#             for k in range(2*n):
#                 a[j][k] = a[j][k] - ratio * a[i][k]
#             print("A",j,i,-ratio)

# # Row operation to make principal diagonal element to 1
# for i in range(n):
#     divisor = a[i][i]
#     if divisor != 0.0:
#         for j in range(2*n):
#             a[i][j] = a[i][j]/divisor
#         print("M",i,1/divisor)

# # Displaying Inverse Matrix
# print("SOLUTION")
# for i in range(n):
#     for j in range(n, 2*n):
#         print(a[i][j], end='\t')
#     print()

import numpy as np

# Input matrix size
n = int(input())

# Read matrix coefficients
matrix = np.array([list(map(float, input().split())) for _ in range(n)])

# Augment matrix with identity matrix
a = np.hstack((matrix, np.eye(n)))

# Apply Gauss-Jordan elimination
for i in range(n):
    if a[i][i] == 0.0:
        nonzero_row = np.nonzero(a[i+1:, i])[0]
        if len(nonzero_row) > 0:
            pivot_row = i + 1 + nonzero_row[0]
            a[[i, pivot_row]] = a[[pivot_row, i]]
            print("S",i,pivot_row)
        else:
            continue
    for j in range(n):
        if i != j:
            ratio = a[j][i] / a[i][i]
            a[j] -= ratio * a[i]
            print("A",j,i,-ratio)

# Check if matrix is degenerate
if np.any(np.all(a[:, :n] == 0.0, axis=1)):
    print("DEGENERATE")
else:
    # Make principal diagonal elements 1
    for i in range(n):
        a[i] /= a[i][i]
        print("M",i,1/a[i][i])
    
    # Display inverse matrix
    print("SOLUTION")
    for row in a[:, n:]:
        print(*row, sep='\t')