# # 已知的视角差异（即azimuth角差10度）
# import cv2
# import numpy as np

# # 定义相机内参矩阵
# fx = fy = 512  # 焦距
# cx = cy = 256   # 主点
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])

# # 计算旋转矩阵
# theta = np.radians(-20)  # 10度转弧度
# R = np.array([[np.cos(theta), 0, -np.sin(theta)],
#               [0, 1, 0],
#               [np.sin(theta), 0, np.cos(theta)]])

# # 计算旋转变换后的投影矩阵
# H = K @ R @ np.linalg.inv(K)

# # 校正图像
# img_b = cv2.imread('0000.png')  # 读取相机B的图像
# img_b_rectified = cv2.warpPerspective(img_b, H, (img_b.shape[1], img_b.shape[0]))

# cv2.imwrite('warp.png', img_b_rectified)  # 保存校正后的图像

import cv2
import numpy as np

# 图像尺寸
image_size = 512

# 旋转角度（假设围绕图像中心旋转）
azim_degrees = 88
azim_radians = np.radians(azim_degrees)

# 读取图像
img_b = cv2.imread('036.png')

# 确保图像正确读取
if img_b is None:
    raise ValueError("Image not found or unable to read.")

# 图像中心
center = (image_size / 2, image_size / 2)

# 计算二维仿射旋转矩阵（围绕Y轴旋转相当于在平面上缩放）
scale_factor = np.cos(azim_radians)  # 由于是正交投影，缩放因子为cos(azim)
translation_x = (1 - scale_factor) * center[0]

# 生成二维仿射变换矩阵
M = np.array([
    [scale_factor, 0, translation_x],
    [0, 1, 0]
])

# 进行仿射变换
img_b_rectified = cv2.warpAffine(img_b, M, (img_b.shape[1], img_b.shape[0]), borderMode=cv2.BORDER_REFLECT)

# 保存或显示结果
cv2.imwrite('warp.png', img_b_rectified)