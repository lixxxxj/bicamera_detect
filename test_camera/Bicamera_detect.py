# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
from utils import *
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
import os
import argparse

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class StereoCameraSystem:
    def __init__(self, args):
        self.model = YOLO("D://YOLO//ultralytics-main//best.pt")
        self.stereo_model = IGEVStereo(args)
        self.stereo_model = torch.nn.DataParallel(self.stereo_model, device_ids=[0])
        self.stereo_model.load_state_dict(torch.load('D://YOLO//ultralytics-main//test_camera//sceneflow//sceneflow.pth'))
        self.stereo_model = self.stereo_model.module
        self.stereo_model.to(DEVICE)
        self.stereo_model.eval()

        self.AUTO = False  # 自动拍照，或手动按s键拍照
        self.INTERVAL = 2  # 自动拍照间隔

        self.plotter = RealTimePlotter(max_len=100,file_path=('D://YOLO//ultralytics-main//test_camera//dis_data//data.xlsx'))  # Initialize the real-time plotter

        
        self.camera = cv2.VideoCapture(0)
        
        # 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720、1280x720
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.counter = 0
        self.utc = time.time()
        self.folder = "D://YOLO//ultralytics-main//test_camera//save_img//"  # 拍照文件目录
        self.num = 0
        self.centroid_tmp = np.zeros(3)
        self.first = 0
        self.camera_index = 0  #摄像头编号

    def shot(self, pos, frame):
        """
        Saves a snapshot of the current frame into a file.
        """
        path = self.folder + pos + "_" + str(self.counter) + ".png"
        cv2.imwrite(path, frame)
        print("snapshot saved into: " + path)

    def load_image_from_array(self, img_array):
        img = torch.from_numpy(img_array).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def stereo_matching(self, left_img, right_img):
        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)
        disp = self.stereo_model(left_img, right_img, iters=16, test_mode=True)
        disp = padder.unpad(disp)
        disp = disp.detach().cpu().numpy().squeeze()
        disp = np.round(disp*256).astype(np.uint16)
        return disp

    def run(self):
        cv2.namedWindow("left")
        cv2.namedWindow("right")
        cv2.namedWindow("disp")
        
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"无法打开摄像头 (index {self.camera_index})")
        else:
            print(f"摄像头 (index {self.camera_index}) 已打开")
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            loop_start = getTickCount()
            
            # 对当前帧进行目标检测并显示结果
            if ret:
                results = self.model.predict(source=frame, device=0)
            
            left_detections = []

            for result in results[0].boxes:
                x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
                if x2 <= 1280:
                    left_detections.append((int(x1), int(y1), int(x2), int(y2), result.cls.cpu().numpy()[0], result.conf.cpu().numpy()[0]))
            print(left_detections)

            # 将双目图像进行裁剪
            left_frame = frame[0:720, 0:1280]
            right_frame = frame[0:720, 1280:2560]
            
            disp_img = None
            if left_detections:
                left_frame_crop, right_frame_crop = frame_crop(left_detections, left_frame, right_frame)
                left_frame_crop = preprocess_image(left_frame_crop)
                right_frame_crop = preprocess_image(right_frame_crop)
                left_img = self.load_image_from_array(left_frame_crop)
                right_img = self.load_image_from_array(right_frame_crop)
                disp = self.stereo_matching(left_img, right_img)
                disp_img = disp_crop(disp)
                disp_process = preprocess_disparity(disp_img)
                point3D = disparity_to_point_cloud(disp_process,left_detections)
                centroid = np.mean(point3D, axis=0)
                print('质心坐标：', centroid)

                if self.num < 10:
                    self.centroid_tmp += centroid
                    self.num += 1
                    print('保存为模板')
                else:
                    displacement = centroid - self.centroid_tmp / 10
                    self.plotter.update_plot({'X': displacement[0], 'Y': displacement[1], 'Z': displacement[2]})
                    averages = self.plotter.get_averages()
                    print(f'X位移为：{displacement[0]}, 平均X位移为：{averages["X"]}')
                    print(f'Y位移为：{displacement[1]}, 平均Y位移为：{averages["Y"]}')
                    print(f'Z位移为：{displacement[2]}, 平均Z位移为：{averages["Z"]}')

            # 对当前帧进行目标检测并绘制结果
            annotated_frame = results[0].plot()

            # 计算FPS
            loop_time = getTickCount() - loop_start
            total_time = loop_time / getTickFrequency()
            FPS = int(1 / total_time)
            
            # 在图像左上角添加FPS文本
            fps_text = f"FPS: {FPS:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 0, 255)  # 红色
            text_position = (10, 30)  # 左上角位置

            left_annotated_frame = annotated_frame[0:720, 0:1280]
            right_annotated_frame = annotated_frame[0:720, 1280:2560]
            cv2.putText(left_annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)

            cv2.imshow("left", left_annotated_frame)
            cv2.imshow("right", right_annotated_frame)
            if disp_img is not None:
                cv2.imshow("disp", disp_img)
                cv2.imshow("disp_process", disp_process)

            # 自动拍照
            now = time.time()
            if self.AUTO and now - self.utc >= self.INTERVAL:
                self.shot("left", left_frame)
                self.shot("right", right_frame)
                self.counter += 1
                self.utc = now

            # 按q键退出,按s键保存图片
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s") and left_detections:
                self.shot("left_crop", left_frame_crop)
                self.shot("right_crop", right_frame_crop)
                if disp_process is not None:
                    self.shot("disp_crop", disp_process)
                self.counter += 1

        self.camera.release()
        cv2.destroyWindow("left")
        cv2.destroyWindow("right")
        cv2.destroyWindow("disp")