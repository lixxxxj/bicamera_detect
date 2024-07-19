import cv2
import params
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from collections import deque
import os 
import pandas as pd
from datetime import datetime

###################################################################图像处理###################################################################
    
def frame_crop(left_detections,left_frame,right_frame):
    """
    Calculates the cropping rectangle based on the detected positions.
    """
    B = 60
    f = params.cameraMatrixL[0][0]
    Z = 1000
    d_u = (f*B)/(Z-100)
    u_top = left_detections[0][0]-d_u
    v_top = left_detections[0][1]
    u_bot = left_detections[0][2]
    v_bot = left_detections[0][3]
    left_frame_crop = left_frame[int(v_top):int(v_bot),int(u_top):int(u_bot)]
    right_frame_crop = right_frame[int(v_top):int(v_bot),int(u_top):int(u_bot)]
    return left_frame_crop,right_frame_crop

def apply_colormap(disp):
    """
    彩色映射并归一化
    """
    # 归一化
    disp_normalized = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 应用颜色映射
    disp_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
    return disp_colored

def preprocess_image(img):
    """
    图像预处理
    """
    # 灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化
    equalized = cv2.equalizeHist(gray)
    
    # 双边滤波
    blurred = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 形态学开运算
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(opened, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(opened, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    
    # 归一化到 0-255
    normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换为 8 位无符号整型
    normalized = normalized.astype(np.uint8)
    
    # 转换为3通道图像
    preprocessed_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    return preprocessed_img


def disp_crop(disp):
    """
    裁剪深度图
    """
    B = 60
    f = params.cameraMatrixL[0][0]
    Z = 1000
    d_u = (f*B)/(Z-100)
    disp_crop = disp[:,int(d_u):]

    return disp_crop

def preprocess_disparity(disp):
    """
    对视差图进行预处理
    Args:
    - disp: 视差图，单通道图像

    Returns:
    - processed_disp: 处理后的视差图
    """
    # 将视差图转换为 32 位浮点数
    disp = disp.astype(np.float32)
    
    # 计算有效点的均值作为阈值
    valid_disp = disp[disp > 0]
    threshold = np.mean(valid_disp)

    # 剔除异常点
    disp[disp < threshold] = 0

    # 使用双边滤波去噪
    filtered_disp = cv2.bilateralFilter(disp, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 进行形态学开运算
    kernel = np.ones((5, 5), np.uint8)
    morph_disp = cv2.morphologyEx(filtered_disp, cv2.MORPH_OPEN, kernel)

    # # 拟合平面
    # points = np.column_stack(np.where(morph_disp > 0))
    # values = morph_disp[morph_disp > 0]
    # A = np.c_[points, np.ones(points.shape[0])]
    # C, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    
    # # 生成平面视差图
    # plane_disp = np.zeros_like(disp, dtype=np.float32)
    # for i in range(disp.shape[0]):
    #     for j in range(disp.shape[1]):
    #         plane_disp[i, j] = C[0] * i + C[1] * j + C[2]

    return morph_disp 




###################################################################点云处理###################################################################
def disparity_to_point_cloud(disparity, left_detections):
    """
    将视差图转换为点云数据
    Args:
    - disparity: 视差图，单通道图像
    - left_detections: 左图中的检测框，用于计算相应的点云

    Returns:
    - points_3D: 点云数据，大小为 (N, 3)
    """
    # 将视差图转换为 32 位浮点数
    disparity = disparity.astype(np.float32)

    # 计算视差的均值，忽略0值
    mean_disparity = np.mean(disparity[disparity > 0])

    # 用均值填充视差为0的地方
    disparity[disparity == 0] = mean_disparity

    # 获取图像尺寸
    h, w = disparity.shape

    # 创建网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 计算 Z 坐标
    Z = -(params.cameraMatrixL[0][0] * params.T[0][0]) / (disparity / 256)

    # 计算 X 和 Y 坐标
    X = -((u + left_detections[0][0] - params.cameraMatrixL[0][2]) * Z) / params.cameraMatrixL[0][0]
    Y = -((v + left_detections[0][1] - params.cameraMatrixL[1][2]) * Z) / params.cameraMatrixL[1][1]

    # 组合为点云数据
    points_3D = np.dstack((X, Y, Z))

    # 过滤掉无效点（视差为0的点）
    mask = disparity > 0
    points_3D = points_3D[mask]

    return points_3D


def visualize_point_cloud_open3d(points_3D):
    """
    使用Open3D可视化点云数据
    Args:
    - points_3D: 点云数据，大小为 (n, 3)
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 将NumPy数组转换为Open3D向量
    pcd.points = o3d.utility.Vector3dVector(points_3D)
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

def mean_filter(cloud, radius):
    """
    使用均值滤波器对点云进行滤波
    """
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    points_copy = np.array(cloud.points)
    points = np.asarray(cloud.points)
    num_points = len(cloud.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)

        points_copy[i] = mean

    cloud.points = o3d.utility.Vector3dVector(points_copy)

def median_filter(pcd, radius):
    """
    使用中值滤波器对点云进行滤波
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        median = np.median(neighbors, 0)

        points_copy[i] = median

    pcd.points = o3d.utility.Vector3dVector(points_copy)


def preprocess_point_cloud(points_3D):
    """
    对点云数据进行预处理，包括滤波和降采样
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)

    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=60,
                                          std_ratio=1.0)
    return np.asarray(cl.points)

    
def visualize_point_cloud_open3d(points_3D):
    """
    使用Open3D可视化点云数据
    Args:
    - points_3D: 点云数据，大小为 (n, 3)
    """
    
    o3d.visualization.draw_geometries([points_3D], window_name="统计滤波",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)


###################################################################表格绘制###################################################################
class RealTimePlotter:
    def __init__(self, max_len=100, file_path='data.xlsx'):
        self.max_len = max_len
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 8))
        self.lines = []
        self.data = {'X': deque(maxlen=max_len), 'Y': deque(maxlen=max_len), 'Z': deque(maxlen=max_len)}
        self.ylim = {'X': [-20, 20], 'Y': [-20, 20], 'Z': [-50, 50]}  # 初始 Y 轴范围
        self.max_value = {'X': 20, 'Y': 20, 'Z': 50}  # 初始最大值
        self.file_path = file_path

        for i, axis in enumerate(['X', 'Y', 'Z']):
            line, = self.ax[i].plot([], [], label=f'{axis} Displacement')
            self.lines.append(line)
            self.ax[i].set_xlim(0, max_len)
            self.ax[i].set_ylim(self.ylim[axis])  # 设置初始 Y 轴范围
            self.ax[i].legend()

        self.fig.tight_layout()
        plt.ion()
        plt.show()

        self.init_excel()

    def init_excel(self):
        """
        Initialize an Excel file with the necessary columns if it doesn't exist.
        """
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=['Timestamp', 'X', 'Y', 'Z'])
            df.to_excel(self.file_path, index=False)
            print(f"Created new Excel file: {self.file_path}")
        else:
            print(f"Excel file already exists: {self.file_path}")

    def update_plot(self, new_data):
        for axis in ['X', 'Y', 'Z']:
            self.data[axis].append(new_data[axis])
            if abs(new_data[axis]) > self.max_value[axis]:
                self.max_value[axis] = abs(new_data[axis])  # 更新最大值
                self.ylim[axis] = [-self.max_value[axis], self.max_value[axis]]  # 更新 Y 轴范围

        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.lines[i].set_xdata(np.arange(len(self.data[axis])))
            self.lines[i].set_ydata(self.data[axis])
            self.ax[i].set_ylim(self.ylim[axis])  # 应用新的 Y 轴范围
            self.ax[i].relim()
            self.ax[i].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # self.update_excel(new_data)  #保存excel数据

    def update_excel(self, new_data):
        """
        Update the Excel file with new X, Y, Z data.
        """
        df = pd.read_excel(self.file_path)
        new_row = pd.DataFrame({'Timestamp': [datetime.now()], 'X': [new_data['X']], 'Y': [new_data['Y']], 'Z': [new_data['Z']]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(self.file_path, index=False)
        print(f"Updated Excel file: {self.file_path} with new data: {new_data}")

    def get_averages(self):
        averages = {}
        for axis in ['X', 'Y', 'Z']:
            if len(self.data[axis]) > 0:
                averages[axis] = np.mean(self.data[axis])
            else:
                averages[axis] = 0.0
        return averages