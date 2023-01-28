import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
#######################################################################
#Func to get lists
def get_lists(labels, bboxes):
    human_list = []
    gun_list = []
    for i, element in enumerate(labels):
        if element=='Gun':
            gun_list.append(bboxes[i])
        elif element=='Human':
            human_list.append(bboxes[i])
    return human_list, gun_list

#Func. to get center points
def get_centerpoint(bbox):
    #bbox is a list of list of bounding boxes
    center_pt_list = []
    for i in bbox:
        center_pt = (int((i[0]+i[2])/2), int((i[1]+i[3])/2))
        center_pt_list.append(center_pt)
    return center_pt_list

#Function to check if the points are in the bboxes
def rectContains(rect,pts):
    for pt in pts:
        if (pt[0]>rect[0] and pt[0]<rect[2] and pt[1]>rect[1] and pt[1]<rect[3]):
            logic = True
        else:
            logic = False            
    return logic

# Func. to check if the two bounding boxes are interlaping 
def rectIntersects(rect1,rect2):
    p1 = Polygon([(rect1[0],rect1[1]), (rect1[1],rect1[1]),(rect1[2],rect1[3]),(rect1[2],rect1[1])])
    p2 = Polygon([(rect2[0],rect2[1]), (rect2[1],rect2[1]),(rect2[2],rect2[3]),(rect2[2],rect2[1])])
    return(p1.intersects(p2))

#Function to get bbox and color based on the logic system
def box_rating(bbox, gun_logic_list_center, gun_logic_list_intersect):
    rating = []
    color = (0,0,255)
    if (gun_logic_list_center==True) or (gun_logic_list_intersect==True):
        rating.append([bbox, color])
    else:
        rating.append([bbox, (0,255,0)])
    return rating

# Function return a dictionary of counts
def counter(labels):
    dic = {}
    for i in labels:
        dic[i] = labels.count(i)
    return dic
###################################################################################

###################################################################################
# Setting some variables
model_path = r'C:\Users\LENOVO\Desktop\SRS Project\Security\model\202map.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SecurityImgs\White House Shooting  Secret Service Shoot Gun-wielding Man [CAUGHT ON TAPE] (534).jpg'
# img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SecurityImgs\Moment mexican cowboy stopped armed robbery - BBC News (76).jpg'
# img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SecurityImgs\Surveillance Camera Records Deadly State Street Shootout (74).jpg'
img_path = r'C:\Users\LENOVO\Desktop\SRS Project\media\TestImages\SecurityImgs\CNN has obtained videos from inside the Westgate Mall (57).jpg'
classes= ['__background__', 'Gun', 'Human']

# Model building
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
num_classes = 3
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Object detection function
def obj_detector(img,thresh,classes):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0,3,1,2)

    model.eval()

    detection_threshold = thresh
    
    img = list(im.to(device) for im in img)
    output = model(img)
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in output]
    # # print(outputs)
    
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = output[0]['labels'].data.numpy()
        
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        labels = labels[scores >= detection_threshold]
        scores = scores[scores >= detection_threshold]
        
        # # get all the predicited class names
        pred_classes = [classes[i] for i in labels]
    
    sample = img[0].permute(1,2,0).cpu().numpy()
    sample = np.array(sample)
    
    return pred_classes, boxes, sample

cv2.namedWindow("FRCNN", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FRCNN", 1024, 720)

# colors = [(255,0,0), (0,100,45), (255,0,0)]
colors = {'__background__': 0, 'Gun': (0,100,45), 'Human':(255,0,0)}
labels, bboxes, img = obj_detector(img_path,0.7,classes)
cat = counter(labels)
print(cat)
sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
for i,box in enumerate(bboxes):
    cv2.rectangle(sample,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    colors[labels[i]], 2)
    cv2.putText(sample, labels[i], (box[0],box[1]-5),cv2.FONT_HERSHEY_COMPLEX ,0.5,(0,0,255),1,cv2.LINE_AA)

# Show Counts on our Image
x, y0 = 30,30
z=0
for i in cat:
    z = z+25
    y = y0 + z 
    var = f'Class: {i}, Count: {cat[i]}'
    cv2.putText(sample, var, (x,y),cv2.FONT_HERSHEY_COMPLEX ,0.8,(0,0,255),2,cv2.LINE_AA)
    
#############################################################################################
#Security Rating Algorithm
'''
The Idea was to get the center points for the detected bounding boxes(Gun) based on class 
and check if these points could be found in the main bounding box(human), also we check if these bboxes intersect with each other. 
Based on this true or false logic we will append the detected main boxes and the specified color rating.

'''
# Get lists
human_list, gun_list = get_lists(labels, bboxes)

#Get centerpoint
gun_list_cp = get_centerpoint(gun_list)
# print(gun_list_cp)

# Get center point logic
gun_logic_list_center = []
for i in human_list:
    log_list = rectContains(i, gun_list_cp)
    gun_logic_list_center.append(log_list)
# print(gun_logic_list_center)

#Get intersect logic
gun_logic_list_intersect = []
for i in human_list:
    log_list = rectIntersects(i, gun_list[0])
    gun_logic_list_intersect.append(log_list)
# print(gun_logic_list_intersect)

#Get the bbox and the corresponing color
rating_list = []
for i, element in enumerate(human_list):
    rating = box_rating(element, gun_logic_list_center[i], gun_logic_list_intersect[i])
    rating_list.append(rating)
rating_list = np.array(rating_list)[:,0]

#Show color rating system
overlay = sample.copy()
for i in rating_list:
    box = i[0]
    color = i[1]
    cv2.rectangle(overlay,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color, -1)
img_new = cv2.addWeighted(overlay, 0.05, sample, 1 - 0.05, 0)

cv2.imshow('FRCNN', img_new)
cv2.waitKey(0)