import cv2
import time
from ultralytics import YOLO

model = YOLO("D://YOLO//ultralytics-main//best.pt")

def cropimg(results,frame):
    t_target = []
    left_top = []
    right_top = []
    for m in results:
        # 获取每个boxes的结果
        box = m.boxes
        # 获取box的位置，
        xyxy = box.xyxy

    for i in range(len(xyxy))  :
        x_top = float(xyxy[i][0])
        y_top = float(xyxy[i][1])
        x_bottom = float(xyxy[i][2])
        y_bottom = float(xyxy[i][3])

        if i == 0:
            left_top.append([x_top,y_top])
            img_crop_left = frame[int(y_top):int(y_bottom), int(x_top):int(x_bottom )]
        if i == 1:
            right_top.append([x_top,y_top])
            img_crop_right = frame[int(y_top):int(y_bottom), int(x_top):int(x_bottom )]
            
        position = [x_top, y_top, x_bottom ,y_bottom]
        t_target.append(position)
    
    return(img_crop_left,img_crop_right,left_top,right_top)



input_path = "D://shuangmu//gezhenzhizuo//biaoding//SIFT//Snapshot001.jpg"
frame = cv2.imread(input_path,0)
results = model.predict(source=frame,device=0)
annotated_frame = results[0].plot()

img_crop_left,img_crop_right,left_top,right_top = cropimg(results,frame)

xyz = cv2.SIFT_create()



