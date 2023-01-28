import torch 
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
import uuid
def counter(labels):
    dic = {}
    for i in labels:
        dic[i] = labels.count(i)
    return dic

# yolov5_model_path = r'C:\Users\LENOVO\Desktop\SRS Project\Rescue\kaggle\working\yolov5\runs\train\exp\weights\best.pt'
yolov5_model_path = r'C:\Users\LENOVO\Desktop\SRS Project\Rescue\Rescueexps\working\yolov5\runs\train\exp5\weights\best.pt'
names = ['human', 'wind/sup-board', 'boat', 'bouy', 'sailboat', 'kayak']
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path, force_reload=True)
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model=model,
    confidence_threshold=0.55,
    device="cpu", # or 'cuda:0'
    load_at_init=True
)
save_dir = 'images/'
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO", 1024, 720)

#Set a video path
video_path = r'C:\Users\LENOVO\Desktop\SRS Project\Safety\rescuetest.avi'
# video_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\dronebeach.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    success,img = cap.read()
    # Get predictions using our model and SAHI
    result = get_sliced_prediction(
    img,
    detection_model,
    slice_height = 512,
    slice_width = 512,
    overlap_height_ratio = 0,
    overlap_width_ratio = 0
)
    #Get model prediction result values
    coco_object_prediction_list = result.to_coco_annotations()
    count_list = []
    for i in coco_object_prediction_list:
        count_list.append(i['category_name'])
    #Count the number of classes predicted
    dic_list = counter(count_list)
    print(dic_list)
    #Export our result to get an image of our predictions
    result.export_visuals(export_dir="./")
    #read back the predictions
    image = cv2.imread('./prediction_visual.png')
    # Show Counts on our Image
    x, y0 = 30,30
    z=0
    for i in dic_list:
        z = z+35
        y = y0 + z 
        var = f'Class: {i}, Count: {dic_list[i]}'
        cv2.putText(image, var, (x,y),cv2.FONT_HERSHEY_COMPLEX ,1.2,(0,0,255),2,cv2.LINE_AA)
    #save our image
    name = os.path.join(save_dir+ str(uuid.uuid1()) + '.jpg')
    # cv2.imwrite(name, image)
    cv2.imshow('YOLO', image)
    if cv2.waitKey(0) & 0xFF == ord('q'): #1000 or 0
        break
    #remove the image file
    os.remove('./prediction_visual.png')
    
cap.release()
cv2.destroyAllWindows()
###########################
