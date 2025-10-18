#%%###################################################################################
########3DDATA 2 (xyzc).txt###########################################################
######################################################################################
import glob
from osgeo import gdal
import re
import pandas as pd

# 设置文件夹路径
folder_path = 'G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/3DRiceSeedPRJT/data/3D-TissueSeg4ELD/'
#% 获取文件夹中的所有.tif文件
tif_files = glob.glob(folder_path + '*.tif')
print(tif_files)
# 创建空的数据列表
data = []

# 遍历每个.tif文件
for tif_file in tif_files:
    print(tif_file)
    # 打开.tif文件
    dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
    
    # 获取.tif文件的宽度、高度和深度（波段数）
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    depth = dataset.RasterCount
    
    # 获取Z轴坐标
    z_value = int(re.findall('\d+', tif_file)[-1])
    
    # 遍历每个像素
    for y in range(height):
        for x in range(width):
            # 获取波段对象
            band = dataset.GetRasterBand(1)  # 这里只读取第一个波段的灰度值
            
            # 获取灰度值
            grayscale = band.ReadAsArray(x, y, 1, 1)[0, 0]
            
            # 如果灰度值不为0，则添加到数据列表中
            if grayscale != 0:
                data.append((x, y, z_value, grayscale))

# 打印数据列表
# 输出结果到txt文件
with open(f"{folder_path}output.txt", 'w') as f:
    for item in data:
        f.write(str(item) + '\n')

#%%###################################################################################
########read txt 2 whole_points_3d####################################################
######################################################################################
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 统计行数
folder_path = folder_path
output_path = f'{folder_path}zoo/'

with open(f"{folder_path}output.txt", 'r') as file:
    line_count = 0
    for line in file:
        line_count += 1

lineCount = line_count
print("文件总行数：", lineCount)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取output.txt
with open(f"{folder_path}output.txt", 'r') as file:
    # 
    for _ in range(0):
        next(file)
        
    # 读取数据
    x = []
    y = []
    z = []
    grayscale = []
    line_count = 0
    for line in file:
        # 分割每行数据
        data = line.strip().split(',')
        if len(data) == 4:
            ## 预处理数据
            data[0] = data[0].replace('(', '')
            data[3] = data[3].replace(')', '').strip()
            
            # 解析数据
            x.append(float(data[0]))
            y.append(float(data[1]))
            z.append(float(data[2]))
            grayscale.append(float(data[3]))
        
        line_count += 1
        # 
        if line_count == lineCount:
            break
point_set = np.array([x, y, z, grayscale])
len(point_set)
# # 绘制三维图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=grayscale, cmap='gray',s=1)

# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 显示图形
# plt.show()

# # 计算XYZ坐标的范围
# x_min = min(x)
# x_max = max(x)
# y_min = min(y)
# y_max = max(y)
# z_min = min(z)
# z_max = max(z)

# print("X范围:", x_min, "to", x_max)
# print("Y范围:", y_min, "to", y_max)
# print("Z范围:", z_min, "to", z_max)
# #%% ##############################SURFACE_POINTS###################################
# point_set = np.array([x, y, z, grayscale])
# surface_points = point_set.transpose()
# print(surface_points)
# centroid = np.mean(surface_points, axis=0)
# # # 将centroid的坐标四舍五入取整
# # centroid = np.round(centroid).astype(int)
# print(centroid)
# surface_points_3d = surface_points[:, :3] 
# surface_centroid = centroid[:3] # 中心点的坐标
# print(surface_centroid)

#%% ###############################WHOLE_POINTS#######################################
point_set = np.array([x, y, z, grayscale])
whole_points = point_set.transpose()
print(whole_points)
centroid = np.mean(whole_points, axis=0)
# # 将centroid的坐标四舍五入取整
# centroid = np.round(centroid).astype(int)
print(centroid)
whole_points_3d = whole_points[:, :3] 
whole_centroid = centroid[:3] # 中心点的坐标
print(whole_points_3d)
#%%###################################################################################
########build pseudo 3D zoo###########################################################
######################################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_uniform_vectors(n):
    # 生成球面均匀分布的向量
    vectors = []
    inc = np.pi * (3 - np.sqrt(5))
    off = 2 / float(n)
    for k in range(n):
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y*y)
        phi = k * inc
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        vectors.append([x, y, z])
    return vectors

# 生成n个长度不限，不共线的整数向量
n = 1000
vectors = generate_uniform_vectors(n)

# with open(f"{folder_path}{n}-vectors.txt", 'w') as Vectors:
#     # 输出结果
#     for vector in vectors:
#         Vectors.write(f"{vector}\n")
# n=1000
###################################################################################
# vectors = []
# with open(f"{folder_path}{n}-vectors.txt", 'r') as Vectors:
#     lines = Vectors.readlines()
#     for line in lines:
#         vector = [float(x.strip()) for x in line.strip()[1:-1].split(',')]
#         vectors.append(vector)
print(vectors)
#% 绘制三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [p[0] for p in vectors]
y = [p[1] for p in vectors]
z = [p[2] for p in vectors]

ax.quiver(0, 0, 0, x, y, z, length = 25,linewidth=0.5)

ax.set_xlim(-25, +25)
ax.set_ylim(-25, +25)
ax.set_zlim(-25, +25)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_box_aspect([1, 1, 1])
# plt.savefig(f"{folder_path}{n}-vectors.png")
plt.show()

# 定义截面函数
from numba import cuda
import numpy as np
import math

@cuda.jit
def distance_to_plane_kernel(x, y, z, A, B, C, D, distance, sqrt_3_over_2):
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        dist = abs(A * x[idx] + B * y[idx] + C * z[idx] + D) / math.sqrt(A**2 + B**2 + C**2)
        # print(idx,dist) 
        distance[idx] = dist if dist < sqrt_3_over_2 else -1
        

def calculate_distances(x, y, z, A, B, C, D, distance, sqrt_3_over_2):
    threads_per_block = 128
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
    distance_to_plane_kernel[blocks_per_grid, threads_per_block](x, y, z, A, B, C, D, distance, sqrt_3_over_2)

sqrt_3_over_2 = np.float32(np.sqrt(3) / 2)

x = whole_points_3d[:, 0]  # 使用 whole_points_3d 的坐标列
y = whole_points_3d[:, 1]
z = whole_points_3d[:, 2]
# grayscale[grayscale == 1] = 255

print(len(x))
print(len(y))
print(len(z))
print(len(grayscale))

#%% build zoo
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Assuming CUDA kernel function 'calculate_distances' is properly defined and compiled
# from cuda_module import calculate_distances

# Assuming x, y, z, grayscale, whole_centroid, vectors, sqrt_3_over_2 are defined elsewhere

if __name__ == "__main__":
    # 将 grayscale 中的 1 变为 255
    # grayscale[grayscale == 1] = 255
    
    # 预先定义的值
    distance = np.zeros_like(x, dtype=np.float32)
    num = 1
    
    # 处理前三个向量
    for vector in vectors[:1000]:
        a, b, c = vector
        # print("vector:", vector)
        
        gap = 25
        current_point = whole_centroid
        # print("current_point:", current_point)
        
        i = 1
        # while True:
        while i <= 5:
            savePic = f"{num}-{i}"
            # print(savePic)
            
            d = a * current_point[0] + b * current_point[1] + c * current_point[2]
            A, B, C, D = map(np.float32, [a, b, c, -d])
            # print(A, B, C, D)                                                                                                          

            # 调用 calculate_distances 并传递 sqrt_3_over_2 参数
            calculate_distances(np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z), A, B, C, D, distance, sqrt_3_over_2)
            valid_indices = distance >= 0

            projection_points = [(x[idx], y[idx], z[idx], grayscale[idx]) for idx in valid_indices.nonzero()[0]]
            # print(f"{num}-{i} 上的有效投影点数：{len(projection_points)}")

            if len(projection_points) == 0:
                break

            points = np.array(projection_points)
            # print("投影点数：", len(points))
            centroid = np.mean(points[:, :3], axis=0)
            normal = np.array([A, B, C])
            normal = normal / np.linalg.norm(normal)

            rotation_axis = np.cross(normal, [0, 0, 1])
            if np.linalg.norm(rotation_axis) != 0:
                rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(normal, [0, 0, 1]) / (np.linalg.norm(normal) * np.linalg.norm([0, 0, 1])))
                rotation = Rotation.from_rotvec(rotation_axis_normalized * angle)
                rotated_points = rotation.apply(points[:, :3] - centroid) + centroid
            else:
                print("法向量与 Z 轴平行，无需旋转")
                rotated_points = points[:, :3]

            fig = plt.figure(facecolor='black')
            ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1, facecolor='black')
            ax.axis('off')
            dpi = 100
            # 更改颜色映射
            sc = ax.scatter(rotated_points[:, 0], rotated_points[:, 1], c=points[:, 3], cmap='gray', s=1)

            num_points = np.array([rotated_points[:, 0].max() - rotated_points[:, 0].min(), rotated_points[:, 1].max() - rotated_points[:, 1].min()])
            
            # 设置最小图像尺寸
            min_size = 0.01  # 设置最小尺寸为 0.01 英寸
            num_points[num_points < min_size * dpi] = min_size * dpi

            fig.set_size_inches(num_points / dpi)

            try:
                plt.savefig(f"{folder_path}/{savePic}-cuda.png", dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='black')
                print(savePic)
            except Exception as e:
                print(f"保存图像 {savePic} 时出错：{e}")

            plt.close(fig)

            current_point = np.array(current_point)
            vector = np.array(vector)
            current_point += gap * vector
            i += 1
        num += 1

#%%###################################################################################
########spin & filp###################################################################
######################################################################################
import cv2
import numpy as np
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt

def get_boundaries_and_centroid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    if binary is None or binary.size == 0:
        print("二值图像创建失败")
        return None, None, None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("图像中未找到轮廓")
        return None, None, binary

    all_points = np.vstack(contours)
    M = cv2.moments(all_points)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    centroid = (cx, cy)
    
    return contours, centroid, binary

def find_farthest_point(contours, centroid):
    points = np.vstack(contours)[:, 0, :]
    dists = distance.cdist([centroid], points)[0]
    max_dist_index = np.argmax(dists)
    farthest_point = points[max_dist_index]
    return tuple(farthest_point)

def rotate_points_around_centroid(points, centroid, angle):
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    rotated_points = []
    for point in points:
        x, y = point
        cx, cy = centroid
        x -= cx
        y -= cy
        
        x_new = x * cos_angle - y * sin_angle
        y_new = x * sin_angle + y * cos_angle
        
        x_new += cx
        y_new += cy
        rotated_points.append((x_new, y_new))
    
    return np.array(rotated_points, dtype=np.int32)

def calculate_bounding_box(points):
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def visualize_contours_and_centroid(image, contours, centroid, farthest_point):
    visualized_image = image.copy()
    cv2.drawContours(visualized_image, contours, -1, (0, 255, 0), 2)  # 绘制轮廓为绿色
    cv2.circle(visualized_image, centroid, 5, (0, 0, 255), -1)  # 绘制质心为红色圆点
    cv2.circle(visualized_image, farthest_point, 5, (255, 0, 0), -1)  # 绘制最远点为蓝色圆点
    cv2.line(visualized_image, centroid, farthest_point, (255, 255, 0), 2)  # 绘制质心和最远点的连线为黄色
    return visualized_image

def visualize_rotated_points(rotated_image):
    plt.figure(figsize=(8, 8))  # 设置图像大小
    plt.imshow(rotated_image)
    plt.title('Rotated Contours Visualization')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif'):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"无法加载图像：{img_path}")
                continue
            
            contours, centroid, binary = get_boundaries_and_centroid(image)
            if contours is None or centroid is None:
                print(f"由于处理失败，跳过图像：{img_path}")
                continue
            
            farthest_point = find_farthest_point(contours, centroid)
            
            delta_x = farthest_point[0] - centroid[0]
            delta_y = farthest_point[1] - centroid[1]
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            
            # 将原图像放置在一个足够大的画布上
            h, w, _ = image.shape
            diagonal = int(np.sqrt(h**2 + w**2))
            canvas = np.zeros((diagonal, diagonal, 3), dtype=image.dtype)
            offset_x = (diagonal - w) // 2
            offset_y = (diagonal - h) // 2
            canvas[offset_y:offset_y+h, offset_x:offset_x+w] = image

            new_centroid = (centroid[0] + offset_x, centroid[1] + offset_y)

            rotated_image = np.zeros_like(canvas)
            for i in range(canvas.shape[0]):
                for j in range(canvas.shape[1]):
                    if 0 <= i - offset_y < binary.shape[0] and 0 <= j - offset_x < binary.shape[1] and binary[i - offset_y, j - offset_x] == 255:
                        new_x = int((j - new_centroid[0]) * np.cos(np.radians(-angle)) - (i - new_centroid[1]) * np.sin(np.radians(-angle)) + new_centroid[0])
                        new_y = int((j - new_centroid[0]) * np.sin(np.radians(-angle)) + (i - new_centroid[1]) * np.cos(np.radians(-angle)) + new_centroid[1])
                        if 0 <= new_x < canvas.shape[1] and 0 <= new_y < canvas.shape[0]:
                            rotated_image[new_y, new_x] = canvas[i, j]

            rotated_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_spinning.png")
            cv2.imwrite(rotated_output_path, rotated_image)
            print(f"已处理并保存：{rotated_output_path}")

            # 垂直翻转
            flipped_image = cv2.flip(rotated_image, 0)
            flip_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_flip.png")
            cv2.imwrite(flip_output_path, flipped_image)
            print(f"已保存垂直翻转后的图像：{flip_output_path}")

            # 水平翻转
            horizontal_flip_image = cv2.flip(rotated_image, 1)
            horizontal_flip_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_spinning_hflip.png")
            cv2.imwrite(horizontal_flip_output_path, horizontal_flip_image)
            print(f"已保存旋转后水平翻转的图像：{horizontal_flip_output_path}")

            horizontal_flip_image_flip = cv2.flip(flipped_image, 1)
            horizontal_flip_flip_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_flip_hflip.png")
            cv2.imwrite(horizontal_flip_flip_output_path, horizontal_flip_image_flip)
            print(f"已保存垂直翻转后水平翻转的图像：{horizontal_flip_flip_output_path}")

            # 可视化并保存质心和最远点的连线
            visualized_image = visualize_contours_and_centroid(image, contours, centroid, farthest_point)
            visualized_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_visualized.png")
            cv2.imwrite(visualized_output_path, visualized_image)
            print(f"已保存可视化图像：{visualized_output_path}")

if __name__ == "__main__":
    input_folder = f"{output_path}/"
    output_folder = f"{input_folder}/spinning"
    process_images(input_folder, output_folder)
