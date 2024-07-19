from utils import *
import cv2

disp = cv2.imread('D://YOLO//ultralytics-main//test_camera//save_img//disp_crop_1.png', cv2.IMREAD_UNCHANGED)
# disp = preprocess_disparity(disp)
# left_detections = [(636, 422, 747, 530, 0.0, 0.9157977)]
# point3D = disparity_to_point_cloud(disp,left_detections)
print(disp)