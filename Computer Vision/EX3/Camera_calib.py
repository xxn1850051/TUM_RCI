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



def main():
    #reading input
    dd_points = read_data()
    ddd_points  = [np.array([0, 0, 0,1]),
                   np.array([0, 0, 1,1]),
                   np.array([0, 1, 0,1]),
                   np.array([0, 1, 1,1]),

                   np.array([1, 0, 0,1]),
                   np.array([1, 0, 1,1]),
                   np.array([1, 1, 0,1]),
                   np.array([1, 1, 1,1]),
                   ]

    # setting up Linear System: Q * m = 0
    Q_list = []
    for i  in range(len(ddd_points)):
        zero_3  = np.array([0,0,0,0])
        row_1 = np.concatenate([ddd_points[i],zero_3,-dd_points[i][0]*ddd_points[i]])
        row_2 = np.concatenate([zero_3,ddd_points[i],-dd_points[i][1]*ddd_points[i]])
        tmp = np.stack([row_1,row_2],axis=0)
        Q_list.append(tmp)

    Q = np.concatenate([Q_list[i] for i in range(len(Q_list))], axis=0)

    # solving linear System with SVD, min(||Qm||2) with ||m||=1   , Q.shape = (2n,12)
    u, s, vh = np.linalg.svd(Q, full_matrices=True)  # shape(2n,2n) ,(12), (12,12)
    v  = vh.T

    s_smallest = 1e10
    index=0
    for i in range(len(s)):
        if s[i]<s_smallest and s[i]>0:
            index=i

    m = s[index] * v[:,index]
    m_normalized = m/np.linalg.norm(m)

    # M = K * [R,t]  <=>  [M1, M2] = K * [R,t]
    # First solving M1 = K * R using RQ decomposition

    M = np.reshape(m_normalized,(3,4))
    if np.linalg.det(M[:,:3])<0:   # det(A)=det(RQ) = det(Q) with det(R)=1, since we want QeSO3, det(Q) has to be 1s
        M = -M
    M1 = M[:,:3]
    M2 = M[:,3]

    K,R = rq(M1)

    t = np.dot(np.linalg.inv(K),M2)
    K = K / K[2, 2]

    # output data
    outputstr = ""
    for row in K:
        for item in row:
            outputstr += str(item)+" "
        outputstr +="\n"

    for row in np.concatenate([R,t[:,None]],axis=-1):
        for item in row:
            outputstr += str(item)+" "
        outputstr +="\n"

    print(outputstr)

    pass

if __name__ == '__main__':
    main()
