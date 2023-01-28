**Rescue**

This Folder contains the file for the Rescue Object Detection model. The model was built using YOLOv5 and the AFO dataset on kaggle 
(https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects).

AFO dataset is the first free dataset for training machine learning and deep learning models for maritime Search and Rescue applications. 
It contains aerial-drone videos with 40,000 hand-annotated persons and objects floating in the water, many of small size, which makes them difficult to detect.
The AFO dataset contains images taken from fifty video clips containing objects floating on the water surface, 
captured by the various drone-mounted cameras (from 1280x720 to 3840x2160 resolutions), which have been used to create AFO. 
From these videos, we have extracted and manually annotated 3647 images that contain 39991 objects.
These have been then split into three parts: the training (67,4% of objects), the test (19,12% of objects), and the validation set (13,48% of objects). 
In order to prevent overfitting of the model to the given data, the test set contains selected frames from nine videos that were not used in either the training or validation sets

![a_303](https://user-images.githubusercontent.com/66518563/215276839-9f499afd-001f-4d95-8ca0-6a73e61f5fc3.jpg)


**Data Ackwoledgements**
This work was supported by the Polish National Science Center under grant no. 2016/21/B/ST6/01461. We would also like to express our thanks to Agnieszka Malonik Taggart,
Wojciech Sulewski, Dariusz Nawrocki, and Wojciech Kubiela - photographs who decided to support our project by sending us videos for the AFO dataset.

**Model Building**

In the first section of the project we used a YOLOV5m model with some layer freezing technique to build our first model, but we noticed low mAP results in some classes, to
combat this we used the data slicing method to build our model. To  build the model i suugest run the notebook file in the kaggle environment for ease. 
You can check out the main model building notebook with different versions on kaggle @ https://www.kaggle.com/code/ugorjiir/rescuefobj/notebook?scriptVersionId=114804963
Or you can use the weights in the model folder for simple inference.

**Model Inference**

For model inference, I suggest looking at the rescuetest.py file first, this file carries out model inference on images one at a time. 
I also built a class counting system to help in counting the detected objects. The other python file, the rescutestvid.py runs inference on video stream. 
I would suggest using files from the media folder in the home directory to carry out inference.
An example of the result is below:

![0b4b3782-9e72-11ed-a84f-68f72886fc79](https://user-images.githubusercontent.com/66518563/215276871-5dbfb472-2b9f-484d-b942-e5527eab7379.jpg)


**Results**

Results can be found in the model folder, with a csv file and some plot images.
