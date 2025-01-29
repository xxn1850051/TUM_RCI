import numpy as np

def compute_relative_pose(camera1, camera2, correspondences):
    # Extract camera parameters
    width, height, focal_x, focal_y, center_x, center_y = camera1
    K1 = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]])

    width, height, focal_x, focal_y, center_x, center_y = camera2
    K2 = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]])

    # Prepare correspondences
    correspondences1 = []
    correspondences2 = []
    for corr in correspondences:
        x1, y1, x2, y2 = corr
        correspondences1.append([x1, y1])
        correspondences2.append([x2, y2])
    correspondences1 = np.array(correspondences1)
    correspondences2 = np.array(correspondences2)

    # Normalize correspondences
    normalized_correspondences1 = np.linalg.inv(K1) @ np.concatenate((correspondences1.T, np.ones((1, correspondences1.shape[0]))))
    normalized_correspondences2 = np.linalg.inv(K2) @ np.concatenate((correspondences2.T, np.ones((1, correspondences2.shape[0]))))

    # Construct the A matrix
    A = np.zeros((2 * correspondences1.shape[0], 6))
    A[::2, :3] = normalized_correspondences1.T
    A[1::2, 3:] = normalized_correspondences1.T

    # Compute the SVD of A
    _, _, V = np.linalg.svd(A)

    # Get the last column of V
    v = V[-1, :]

    # Extract rotation and translation matrices
    R = v[:3].reshape(3, 1)
    T = v[3:].reshape(3, 1)

    return R, T

# Camera parameters
camera1 = [1280, 720, 440, 401, 681, 413]
camera2 = [1280, 720, 419, 430, 656, 400]

# Correspondences
correspondences = [
    [550.244, 602.595, 865.761, 447.014],
    [327.648, 337.005, 1103.82, 268.394],
    [253.93, 472.737, 1087.32, 251.541],
    [502.501, 541.123, 935.675, 426.529],
    [341.377, 385.034, 1089.32, 290.079],
    [103.82, 491.141, 972.076, 227.746],
    [726.841, 369.594, 1086.02, 662.634],
    [377.341, 407.312, 951.041, 338.675]
]

R, T = compute_relative_pose(camera1, camera2, correspondences)
print("Rotation matrix:")
print(R)
print("Translation matrix:")
print(T)
