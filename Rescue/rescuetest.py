import torch 
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil

yolov5_model_path = 'Model/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path, force_reload=True)

################################
#Using Normal Yolov5 Model
img_path = '../Media/TestImages/Rescue/a_303.jpg'
names = ['human', 'wind/sup-board', 'boat', 'bouy', 'sailboat', 'kayak']
# results = model(img_path)
# print(results.print())
# df = results.pandas().xyxy[0]
# classes = df['name'].value_counts()
# classes = classes.to_dict()
# img = np.squeeze(results.render())
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO", 1024, 720)
# # Show Counts on our Image
# x, y0 = 30,30
# z=0
# for i in classes:
#     z = z+55
#     y = y0 + z 
#     var = f'Class: {i}, Count: {classes[i]}'
#     cv2.putText(img, var, (x,y),cv2.FONT_HERSHEY_COMPLEX ,2,(0,0,255),2,cv2.LINE_AA)
    
# cv2.imshow('YOLO', img)

# cv2.waitKey(0)

###################################
# Using a Sliced Inference Model with Yolov5
# #Function to count the values of a list to a dictionary
def counter(labels):
    dic = {}
    for i in labels:
        dic[i] = labels.count(i)
    return dic

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model=model,
    confidence_threshold=0.55,
    device="cpu", # or 'cuda:0'
    load_at_init=True
)
result = get_sliced_prediction(
    img_path,
    detection_model,
    slice_height = 512,
    slice_width = 512,
    overlap_height_ratio = 0,
    overlap_width_ratio = 0
)
coco_object_prediction_list = result.to_coco_annotations()
count_list = []
for i in coco_object_prediction_list:
    count_list.append(i['category_name'])

dic_list = counter(count_list)
print(dic_list)
    
result.export_visuals(export_dir="./")
image = cv2.imread('./prediction_visual.png')

# Show Counts on our Image
x, y0 = 30,30
z=0
for i in dic_list:
    z = z+35
    y = y0 + z 
    var = f'Class: {i}, Count: {dic_list[i]}'
    cv2.putText(image, var, (x,y),cv2.FONT_HERSHEY_COMPLEX ,1.2,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('YOLO', image)
cv2.waitKey(0)
os.remove('./prediction_visual.png')


