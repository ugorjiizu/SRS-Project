import torch 
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import uuid

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
    if pts!=[]:
        for pt in pts:
            if (pt[0]>rect[0] and pt[0]<rect[2] and pt[1]>rect[1] and pt[1]<rect[3]):
                logic = True
            else:
                logic = False
    else:
        logic=False
    return logic

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
        rating.append([bbox, (0,255,255)]) 
    elif ((helmet==False) and (vest==True)):
        rating.append([bbox, (0,255,255)])
    elif ((helmet==False) and (vest==False)):
        rating.append([bbox, (0,0,255)])
    return rating

yolov5_model_path = r'C:\Users\LENOVO\Desktop\SRS Project\Safety\yolov5\runs\train\exp\weights\best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path, force_reload=True)
model.conf = 0.40
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO", 1024, 720)
video_path = './safetytest.avi'
names = ['vest','helmet','worker']
save_dir = 'images/'
i = 1
#str(uuid.uuid1())
cap = cv2.VideoCapture(video_path)
while True:
    success,img = cap.read()
    # # Make detections and get the classes
    results = model(img)
    img = np.squeeze(results.render())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    df = results.pandas().xyxy[0]
    classes = df['name'].value_counts()
    classes = classes.to_dict()
    # Show Counts on our Image
    x, y0 = 30,30
    z=0
    for i in classes:
        z = z+25
        y = y0 + z 
        var = f'Class: {i}, Count: {classes[i]}'
        cv2.putText(img, var, (x,y),cv2.FONT_HERSHEY_COMPLEX ,0.9,(0,0,255),2,cv2.LINE_AA)
    #Safety Rating Algorithm
        #Get lists
    worker_list, helmet_list, vest_list = get_bbox_list(df)
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
        rating = box_rating(element, vest_logic_list[i], helmet_logic_list[i])
        rating_list.append(rating)
    # print(rating_list)
    if rating_list!=[]:
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
        img_new = cv2.addWeighted(overlay, 0.3, img, 1 - 0.3, 0)
        name = os.path.join(save_dir+ str(i)+ '.jpg')
        cv2.imwrite(name, img_new)
        cv2.imshow('YOLO', img_new)
    else:
        # img_new = cv2.addWeighted(overlay, 0.3, img, 1 - 0.3, 0)
        name = os.path.join(save_dir+ str(i)+ '.jpg')
        cv2.imwrite(name, img)
        cv2.imshow('YOLO', img)

    if cv2.waitKey(1000) & 0xFF == ord('q'): #1000 or 0
        break
cap.release()
cv2.destroyAllWindows()