import torch 
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#Function to get bounding box by class
def get_bbox_list(df):
    worker_list = []
    helmet_list = []
    vest_list = []
    for index, row in df.iterrows():
        if row['name']=='worker':
            worker_list.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])
        if row['name']=='helmet':
            helmet_list.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])
        if row['name']=='vest':
            vest_list.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])
    return worker_list, helmet_list, vest_list

#Function to Get the center points for our inference classes(Helmet and Vest)
def get_centerpoint(bbox):
    #bbox is a list of list of bounding boxes
    center_pt_list = []
    for i in bbox:
        center_pt = (int((i[0]+i[2])/2), int((i[1]+i[3])/2))
        center_pt_list.append(center_pt)
    return center_pt_list

#Function to Check if these points are in the workers bounding boxes
def rectContains(rect,pts):
    #Check if the points are in the worker bounding box
    logic_list = []
    for pt in pts:
        if (pt[0]>rect[0] and pt[0]<rect[2] and pt[1]>rect[1] and pt[1]<rect[3]):
            logic = True
            logic_list.append(logic)
        else:
            logic = False
            logic_list.append(logic)
    return logic_list

# Remove empty list (i forgot to append false)
def clean_list(list):
    for i in list:
        if i==[]:
            i.append(False)
    return list

# Fuction for the logic of the rating system
def box_rating(bbox, helmet, vest):
    rating = []
    if ((helmet==True) and (vest==True)):
        rating.append([bbox, (65,255,78)])
    elif ((helmet==True) and (vest==False)):
        rating.append([bbox, (255,80,65)])
    elif ((helmet==False) and (vest==True)):
        rating.append([bbox, (255,80,65)])
    elif ((helmet==False) and (vest==False)):
        rating.append([bbox, (65,65,255)])
    return rating

yolov5_model_path = r'C:\Users\LENOVO\Desktop\SRS Project\Safety\yolov5\runs\train\exp\weights\best.pt'
# yolov5_model_path = 'Model/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path, force_reload=True)
model.conf = 0.40

# img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SafetyImgs\2_331.jpg'
# img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SafetyImgs\17_0000061.jpg'
# img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SafetyImgs\1451.jpg'
img_path = '../Media/TestImages/Safety/00000002.jpg'
names = ['vest','helmet','worker']
results = model(img_path)
print(results.print())
df = results.pandas().xyxy[0]
##################################################################################
#Counting Algorithm
classes = df['name'].value_counts()
classes = classes.to_dict()
img = np.squeeze(results.render(labels=False))
# img = cv2.resize(img, (1024, 720))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO", 1024, 720)
# Show Counts on our Image
x, y0 = 30,30
z=0
for i in classes:
    z = z+25
    y = y0 + z 
    var = f'Class: {i}, Count: {classes[i]}'
    cv2.putText(img, var, (x,y),cv2.FONT_HERSHEY_COMPLEX ,0.9,(0,0,255),2,cv2.LINE_AA)
#####################################################################################
#####################################################################################
#Safety Algorithm Rating
'''
The Idea was to get the center points for the detected bounding boxes(helmet and vest) based on class 
and check if these points could be found in the main bounding box(worker). Based on this true or false logic we will
append the detected main boxes and the specified color rating.

'''
#Get lists
worker_list, helmet_list, vest_list = get_bbox_list(df)
# print(len(worker_list), len(helmet_list), len(vest_list))

#Get center points
helmet_list_cp = get_centerpoint(helmet_list)
vest_list_cp = get_centerpoint(vest_list)

#Get list logic based on the center point position
helmet_logic_list = []
for i in worker_list:
    temp = rectContains(i, helmet_list_cp)
    helmet_logic_list.append(temp)
vest_logic_list = []
for i in worker_list:
        temp = rectContains(i, vest_list_cp)
        vest_logic_list.append(temp)

#remove empty list
helmet_logic_list = clean_list(helmet_logic_list)
vest_logic_list = clean_list(vest_logic_list)

#Get appended bboxes and color
rating_list = []
for i, element in enumerate(worker_list):
    rating = box_rating(element, vest_logic_list[i][0], helmet_logic_list[i][0])
    rating_list.append(rating)
# rating_list = np.squeeze(rating_list)
rating_list = np.array(rating_list)[:,0]

#Show rating systems
overlay = img.copy()
for i in rating_list:
    box = i[0]
    color = i[1]
    print(box, color)
    cv2.rectangle(overlay,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color, -1)
img_new = cv2.addWeighted(overlay, 0.25, img, 1 - 0.25, 0)
##########################################################################################
# cv2.imshow('YOLO', img)
cv2.imshow('YOLO',img_new)
cv2.waitKey(0)