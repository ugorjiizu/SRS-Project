**Security**

This folder contains the Security object detection model. The model was built using a Faster RCNN network with a ResNet50 backbone using pytorch.
Most of our images was collected from fellow Kaggler A. N. M. JUBAER but with no annotations available.
The Images were then Hand annotated in yolo format using Labelimg, which is an annotation tool.
The final dataset used contains over 1000 images annotated with two major classes, Human and Gun. 
These annotations were done using labelimg. The classes.txt contains a list of our classes. 
There are around 14 redundant classes which were not annotated for but were listed due to using labelimg default(First time using the tool). 
The All folder contains our images and our annotations in yolo format (https://www.kaggle.com/datasets/ugorjiir/gun-detection).

![CNN has obtained videos from inside the Westgate Mall (57)](https://user-images.githubusercontent.com/66518563/215279416-05afa848-4dd2-40a4-9665-65cb0633a065.jpg)


**Ackwoledgement**
Kaggler A. N. M. JUBAER for majority of the images used to build the dataset.

**Model Building**

For the model building process, you can take a look at the Gundetect notebook, which breaks down the process of building the object detection model. 
From visualizing some images and their annotations to converting the annotation format for our Faster RCNN model to creating our dataloader, to training our model.
You can check out the main model building notebook with different versions on kaggle @ https://www.kaggle.com/code/ugorjiir/gundetect/notebook and also check out the data section of the notebook to get my trained model

**Model Inference**

For the model Inference process, I recommend starting with the securitytest.py file, this file shows the model inference on a single image, I also add the features such as 
the class counting algorithm and a security rating system, which labels people with their nearness to a gun. The rating system works by checking if the human is close to a gun and 
assigning tint color such as red and green for bad and good respectively (Though the system still needs extra work). The securitytestvid.py file shows the model in action with all its algorithms at play.
Example below:

![7896f9ae-9db8-11ed-8963-68f72886fc79](https://user-images.githubusercontent.com/66518563/215282231-462303fd-e00d-44e7-9fad-afb2bc7ba2b8.jpg)

**Results**

The model earned a mAP of over 0.25 for both classes
