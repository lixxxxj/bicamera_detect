# -*- coding: utf-8 -*-
import cv2
import time
from ultralytics import YOLO
from cv2 import getTickCount,getTickFrequency
 
model = YOLO("D://YOLO//ultralytics-main//best.pt")

AUTO  = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2 # 自动拍照间隔
 
cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0)
 
# 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率1280x480；分割为两个640x480、640x480
camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
 
counter = 0
utc = time.time()
folder = "D://YOLO//ultralytics-main//test_camera//save_img//" # 拍照文件目录
 
def shot(pos, frame):
    '''
    拍照
    '''
    global counter
    path = folder + pos + "_" + str(counter) + ".jpg"
 
    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)
 
while camera.isOpened:
    ret, frame = camera.read()
    loop_start = getTickCount()
    # 对当前帧进行目标检测并显示结果
    if ret:
        results = model.predict(source=frame) 
    annotated_frame = results[0].plot()

    # 计算FPS
    loop_time  = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    
    # 在图像左上角添加FPS文本
    fps_text       = f"FPS: {FPS:.2f}"
    font           = cv2.FONT_HERSHEY_SIMPLEX
    font_scale     = 1
    font_thickness = 2
    text_color     = (0, 0, 255)  # 红色
    text_position  = (10, 30)  # 左上角位置

    #将双目图像进行裁剪，在左图上写FPS并显示
    left_frame  = annotated_frame[0:480, 0:640]
    right_frame = annotated_frame[0:480, 640:1280]
    cv2.putText(left_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)
    
    # 自动拍照
    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
        utc = now
    
    # 按q键退出,按s键保存图片
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")