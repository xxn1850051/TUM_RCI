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


#step 1: Reading Input
m = int(input().strip())
nA = int(input().strip())
A = []
for i in range(nA):
    name, *coords = input().strip().split()
    A.append([float(c) for c in coords])
nB = int(input().strip())
B = []
for i in range(nB):
    name, *coords = input().strip().split()
    B.append([float(c) for c in coords])

# Step 2: Compute matrix C and vector c
A = np.array(A).T  #Affine space of A
B = np.array(B).T  #Affine space of B

#choose mean vector as the origin of the subspace
dA = np.mean(A, axis=1).reshape(-1,1)
dB = np.mean(B, axis=1).reshape(-1,1)

# Conversion of Affine Space to Vector Space
A = A -dA
B = B -dB

# this step actually try to make x stand for affine space
# Originally: A: basis of vector space,  dA: translational vector
#      Wanted: x: spanned vector space + translational vector
#             A * X = 0 - > A is the left null space of X(spanned vector space)

A = null_space(A.T).T
B = null_space(B.T).T


# x has to satisfied constraints of both subspace
C = np.concatenate([A,B],axis=0)
c =np.concatenate([np.dot(A,dA),np.dot(B,dB)])


# Use SVD proceeding least square to perform APPROXIMATION SOL
x_sol = np.linalg.lstsq(C,c)

c_pred = np.dot(C,x_sol[0])


# Then Evalutate whether APPOXIMATION SOL == EXACT SOL, if not, means not exact solution, output N
if np.sum(np.abs(c-c_pred))<=1e-13:
    output_str="Y"
    for item in x_sol[0].squeeze():
        output_str += " "+ str(item)
    print(output_str)
else:
    print("N")


