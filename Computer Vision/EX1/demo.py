import numpy as np

def gaussian_elimination(matrix, dimension):
    n = dimension
    augmented_matrix = np.concatenate((matrix, np.identity(n)), axis=1)

    # Gaussian elimination
    for i in range(n):
        if augmented_matrix[i][i] == 0:
            # Swap rows if the diagonal element is zero
            for j in range(i + 1, n):
                if augmented_matrix[j][i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    print("S",i,j)
                    break
            else:
                # If no non-zero pivot element is found, the matrix is degenerate
                return "DEGENERATE"

        pivot = augmented_matrix[i][i]
        augmented_matrix[i] /= pivot
        print("M",i,pivot)

        for j in range(n):
            if j != i:
                factor = augmented_matrix[j][i]
                augmented_matrix[j] -= factor * augmented_matrix[i]
                print("A",j,i,factor)

    inverse = augmented_matrix[:, n:]

    return inverse

dimension = int(input())
matrix = []
for n in range(dimension):
    row = list(map(int,input().split()))
    matrix.append(row)
matrix = np.array(matrix)

'''# Get the input
input_str = input()

# Split the input into lines
lines = input_str.strip().split('\n')

# Extract the matrix elements from the remaining lines
matrix = []
for line in lines[1:]:
    row = list(map(int, line.split()))
    matrix.append(row)
matrix = np.array(matrix)

# Calculate the inverse using Gaussian elimination'''


'''matrix = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,2,1,0,0],[0,0,3,1,0],[0,0,0,1,1]])
dimension = 5'''
inverse_matrix = gaussian_elimination(matrix, dimension)

# Print the inverse matrix
print("SOLUTION")
for row in inverse_matrix:
    print(' '.join(map(str, row)))






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
        else:
            continue
    for j in range(n):
        if i != j:
            ratio = a[j][i] / a[i][i]
            a[j] -= ratio * a[i]

# Check if matrix is degenerate
if np.any(np.all(a[:, :n] == 0.0, axis=1)):
    print("DEGENERATE")
else:
    # Make principal diagonal elements 1
    for i in range(n):
        a[i] /= a[i][i]
    
    # Display inverse matrix
    print("SOLUTION")
    for row in a[:, n:]:
        print(*row, sep='\t')