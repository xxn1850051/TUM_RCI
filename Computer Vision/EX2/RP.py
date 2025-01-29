# # import numpy as np

# # def pinhole_camera_projection(width, height, fx, fy, cx, cy, point_3d):
# #     x = point_3d[0]
# #     y = point_3d[1]
# #     z = point_3d[2]
    
# #     u = int((fx * x / z) + cx)
# #     v = int((fy * y / z) + cy)
    
# #     return u, v

# # def fov_camera_projection(width, height, fx, fy, cx, cy, w, point_3d):
# #     x = point_3d[0]
# #     y = point_3d[1]
# #     z = point_3d[2]
    
# #     u = int((fx * x / (w * z)) + cx)
# #     v = int((fy * y / (w * z)) + cy)
    
# #     return u, v

# # def pinhole_camera_backprojection(width, height, fx, fy, cx, cy, point_2d, depth):
# #     u = point_2d[0]
# #     v = point_2d[1]
    
# #     x = ((u - cx) * depth) / fx
# #     y = ((v - cy) * depth) / fy
# #     z = depth
    
# #     return x, y, z

# # def fov_camera_backprojection(width, height, fx, fy, cx, cy, w, point_2d, depth):
# #     u = point_2d[0]
# #     v = point_2d[1]
    
# #     x = ((u - cx) * (w * depth)) / fx
# #     y = ((v - cy) * (w * depth)) / fy
# #     z = depth
    
# #     return x, y, z

# # def reproject_point(camera_model_1, camera_model_2, transformation_matrix, point_2d, depth):
# #     if camera_model_1.startswith('pinhole'):
# #         _, width_1, height_1, fx_1, fy_1, cx_1, cy_1 = camera_model_1.split()
# #         projection_func_1 = pinhole_camera_projection
# #         backprojection_func_1 = pinhole_camera_backprojection
# #     elif camera_model_1.startswith('fov'):
# #         _, width_1, height_1, fx_1, fy_1, cx_1, cy_1, w_1 = camera_model_1.split()
# #         projection_func_1 = fov_camera_projection
# #         backprojection_func_1 = fov_camera_backprojection
# #     else:
# #         raise ValueError("Invalid camera model for the first camera")
    
# #     if camera_model_2.startswith('pinhole'):
# #         _, width_2, height_2, fx_2, fy_2, cx_2, cy_2 = camera_model_2.split()
# #         projection_func_2 = pinhole_camera_projection
# #     elif camera_model_2.startswith('fov'):
# #         _, width_2, height_2, fx_2, fy_2, cx_2, cy_2, w_2 = camera_model_2.split()
# #         projection_func_2 = fov_camera_projection
# #     else:
# #         raise ValueError("Invalid camera model for the second camera")
    
# #     R = transformation_matrix[:, :3]
# #     T = transformation_matrix[:, 3]
    
# #     # Backproject the 2D point to the world frame using the first camera's calibration
# #     point_3d_1 = backprojection_func_1(int(width_1), int(height_1), float(fx_1), float(fy_1), float(cx_1), float(cy_1), float(w_1), point_2d, depth)
    
# #     # Apply the transformation matrix to get the 3D point in another space
# #     point_3d_2 = np.dot(R, point_3d_1) + T
    
# #     # Project the 3D point onto the second image using the second camera's calibration
# #     point_2d_2 = projection_func_2(int(width_2), int(height_2), float(fx_2), float(fy_2), float(cx_2), float(cy_2), float(w_2), point_3d_2)
    
# #     return point_2d_2


import numpy as np

def pinhole_camera_projection(width, height, fx, fy, cx, cy, point_3d):
    x = point_3d[0]
    y = point_3d[1]
    z = point_3d[2]
    
    u = int((fx * x / z) + cx)
    v = int((fy * y / z) + cy)
    
    return u, v

def fov_camera_projection(width, height, fx, fy, cx, cy, w, point_3d):
    x = point_3d[0]
    y = point_3d[1]
    z = point_3d[2]
    
    u = int((fx * x / (w * z)) + cx)
    v = int((fy * y / (w * z)) + cy)
    
    return u, v

def pinhole_camera_backprojection(width, height, fx, fy, cx, cy, point_2d, depth):
    u = point_2d[0]
    v = point_2d[1]
    
    x = ((u - cx) * depth) / fx
    y = ((v - cy) * depth) / fy
    z = depth
    
    return x, y, z

def fov_camera_backprojection(width, height, fx, fy, cx, cy, w, point_2d, depth):
    u = point_2d[0]
    v = point_2d[1]
    
    tan_w_2 = np.tan(w / 2)
    r = np.sqrt(u**2 + v**2)
    z = depth * (w * r) / (2 * fx * tan_w_2)
    # z = np.tan(r * w)/(2 * np.tan(w*0.5))
    x = (z / fx) * (u - cx)
    y = (z / fy) * (v - cy)
    
    return x, y, z

def reproject_point(camera_model_1, camera_model_2, transformation_matrix, point_2d, depth):
    if camera_model_1.startswith('pinhole'):
        _, width_1, height_1, fx_1, fy_1, cx_1, cy_1 = camera_model_1.split()
        projection_func_1 = pinhole_camera_projection
        backprojection_func_1 = pinhole_camera_backprojection
    elif camera_model_1.startswith('fov'):
        _, width_1, height_1, fx_1, fy_1, cx_1, cy_1, w_1 = camera_model_1.split()
        projection_func_1 = fov_camera_projection
        backprojection_func_1 = fov_camera_backprojection
    else:
        raise ValueError("Invalid camera model for the first camera")
    
    if camera_model_2.startswith('pinhole'):
        _, width_2, height_2, fx_2, fy_2, cx_2, cy_2 = camera_model_2.split()
        projection_func_2 = pinhole_camera_projection
    elif camera_model_2.startswith('fov'):
        _, width_2, height_2, fx_2, fy_2, cx_2, cy_2, w_2 = camera_model_2.split()
        projection_func_2 = fov_camera_projection
    else:
        raise ValueError("Invalid camera model for the second camera")
    
    R = transformation_matrix[:, :3]
    T = transformation_matrix[:, 3]
    
    # Backproject the 2D point to the world frame using the first camera's calibration
    point_3d_1 = backprojection_func_1(int(width_1), int(height_1), float(fx_1), float(fy_1), float(cx_1), float(cy_1), float(w_1), point_2d, depth)
    
    # Apply the transformation matrix to get the 3D point in another space
    point_3d_2 = np.dot(R, point_3d_1) + T
    
    # Project the 3D point onto the second image using the second camera's calibration
    point_2d_2 = projection_func_2(int(width_2), int(height_2), float(fx_2), float(fy_2), float(cx_2), float(cy_2), float(w_2), point_3d_2)
    
    return point_2d_2

# Example usage
camera_model_1 = "pinhole 1280 720 448 448 640 360"
camera_model_2 = "fov     1280 720 448 448 640 360 1.1"
transformation_matrix = np.array([[1, 0, 0, -2],
                                  [0, 1, 0, 3],
                                  [0, 0, 1, 4]])
point_2d = (500, 500)
depth = 22

# reprojected_point = reproject_point(camera_model_1, camera_model_2, transformation_matrix, point_2d, depth)
# print("Reprojected point:", reprojected_point)


# import numpy as np

# def reproject(pinhole_A, pinhole_B, T_A_to_B, P_A, depth_A):
#     # 步骤1: 将点P从平面A的坐标系转换到世界坐标系
#     width_A = pinhole_A[0, 0]
#     height_A = pinhole_A[0, 1]
#     focal_x_A = pinhole_A[0, 2]
#     focal_y_A = pinhole_A[0, 3]
#     center_x_A = pinhole_A[0, 4]
#     center_y_A = pinhole_A[0, 5]

#     pixel_size_A_x = width_A / focal_x_A
#     pixel_size_A_y = height_A / focal_y_A

#     x_rel_A = (P_A[0] - center_x_A) * pixel_size_A_x
#     y_rel_A = (P_A[1] - center_y_A) * pixel_size_A_y

#     x_world_A = x_rel_A * depth_A
#     y_world_A = y_rel_A * depth_A
#     z_world_A = depth_A

#     P_world = np.array([x_world_A, y_world_A, z_world_A, 1])

#     # 步骤2: 将点P从世界坐标系转换到平面B的坐标系
#     P_B_homogeneous = np.dot(T_A_to_B, P_world)

#     x_B_homogeneous = P_B_homogeneous[0]
#     y_B_homogeneous = P_B_homogeneous[1]
#     z_B_homogeneous = P_B_homogeneous[2]

#     x_B = x_B_homogeneous / z_B_homogeneous
#     y_B = y_B_homogeneous / z_B_homogeneous

#     # 步骤3: 将点P_B投影到平面B上
#     width_B = pinhole_B[0, 0]
#     height_B = pinhole_B[0, 1]
#     focal_x_B = pinhole_B[0, 2]
#     focal_y_B = pinhole_B[0, 3]
#     center_x_B = pinhole_B[0, 4]
#     center_y_B = pinhole_B[0, 5]

#     pixel_size_B_x = width_B / focal_x_B
#     pixel_size_B_y = height_B / focal_y_B

#     x_rel_B = x_B * pixel_size_B_x
#     y_rel_B = y_B * pixel_size_B_y

#     x_B_final = x_rel_B + center_x_B
#     y_B_final = y_rel_B + center_y_B

#     P_B_projection = np.array([x_B_final, y_B_final])

#     return

# A = []
# typeA, *coordsA = input().strip().split()
# print(coordsA)
# A.append([float(c) for c in coordsA])

# B = []
# typeB, *coordsB = input().strip().split()
# B.append([float(c) for c in coordsB])

# TM = np.array([list(map(float, input().split())) for _ in range(3)])

# Points = np.array([list(map(float, input().split())) for _ in range(9)])

# for i in range(9):
#     P = Points[i]
#     P_A = (P[0], P[1])
#     depth = A[2]


def garct1(pixel_position: np.ndarray, w):
    r = np.linalg.norm(pixel_position)
    return pixel_position * (np.tan(r * w)) / (2 * np.tan(w/2)) / r

def garct2(pixel_position: np.ndarray, w):
    r = np.linalg.norm(pixel_position)
    return pixel_position * np.arctan(2 * r * np.tan(w/2)) / (w * r)

def reproject_point(camera1_model, camera2_model, transformation_matrix, points):
    # Parse camera models
    camera1_params = camera1_model.split()
    camera2_params = camera2_model.split()

    # Extract camera calibration parameters
    camera1_calib = [float(param) for param in camera1_params[1:]]
    camera2_calib = [float(param) for param in camera2_params[1:]]


    # Convert points' projections on the first image to homogeneous coordinates
    points_np = np.array(points)
    points_homogeneous = np.column_stack((points_np[:, :2], np.ones(len(points))))

    # Calculate 3D coordinates of the points in the first camera's coordinate system
    camera1_matrix = np.eye(3)
    camera1_matrix[0, 0] = camera1_calib[2]  # Focal X
    camera1_matrix[1, 1] = camera1_calib[3]  # Focal Y
    camera1_matrix[0, 2] = camera1_calib[4]  # Center X
    camera1_matrix[1, 2] = camera1_calib[5]  # Center Y
    points_homogeneous = (np.linalg.inv(camera1_matrix) @ points_homogeneous.T).T
    for i in range(points_homogeneous.shape[0]):
        if camera1_params[0] == "fov":
            points_homogeneous[i][:2] = garct1(points_homogeneous[i][:2], camera1_calib[6])
        points_homogeneous[i] = points_homogeneous[i] * (points_np[i, 2] / np.linalg.norm(points_homogeneous[i]))

    points_3d_camera1 = np.vstack((points_homogeneous.T, np.ones(len(points))))

    # Transform 3D coordinates to the second camera's coordinate system
    points_3d_camera2 = transformation_matrix @ points_3d_camera1

    # Project the transformed 3D coordinates onto the second image
    camera2_matrix = np.eye(3)
    camera2_matrix[0, 0] = camera2_calib[2]  # Focal X
    camera2_matrix[1, 1] = camera2_calib[3]  # Focal Y
    camera2_matrix[0, 2] = camera2_calib[4]  # Center X
    camera2_matrix[1, 2] = camera2_calib[5]  # Center Y

    points_homogeneous_camera2 = camera2_matrix @ points_3d_camera2
    points_2d_camera2 = (points_homogeneous_camera2 / points_homogeneous_camera2[2, :])
    #print(points_2d_camera2.T)

    if camera2_params[0] == "fov":
        points_2d_camera2 = (np.linalg.inv(camera2_matrix) @ points_2d_camera2).T
        #print(points_2d_camera2)
        for i in range(points_2d_camera2.shape[0]):
            points_2d_camera2[i][:2] = garct2(points_2d_camera2[i][:2], camera2_calib[6])
        #print(points_2d_camera2)
        points_2d_camera2 = (camera2_matrix @ points_2d_camera2.T).T[:, :2]
    else:
        points_2d_camera2 = points_2d_camera2.T[:, :2]

    # Check if projected points lie within the image bounds
    width, height = int(camera2_calib[0]), int(camera2_calib[1])
    valid_points = []
    for point in points_2d_camera2:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            valid_points.append(point)
        else:
            valid_points.append("OB")  # Outside the image
    for idx, point in enumerate(points_3d_camera2.T):
        if point[2] < 0:
            valid_points[idx] = "OB"
    return valid_points

# Get input from the user
camera1_model = input()
camera2_model = input()
transformation_matrix = np.zeros((3, 4))
for i in range(3):
    row = input()

    transformation_matrix[i] = [float(val) for val in row.split()]


points = []
for i in range(9):
    point = input()

    x, y, distance = [float(val) for val in point.split()]
    points.append([x, y, distance])
# Call the reproject_point function
result = reproject_point(camera1_model, camera2_model, transformation_matrix, points)
# Print the result
for _, i in enumerate(result):
    if isinstance(i, str):
        print(i)
    else:
        print("%f %f" % (i[0], i[1]))
