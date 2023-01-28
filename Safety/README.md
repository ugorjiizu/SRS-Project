**Safety**

This Folder contains the file for the Safety Object Detection model. 
The model was built using YOLOv5m and the dataset was sourced from a kaggle dataset https://www.kaggle.com/datasets/mikhailma/railroad-worker-detection-dataset
The dataset contains more then 3 thousands yolo-format labeled images with railway workers.

![1214](https://user-images.githubusercontent.com/66518563/215277435-827f0fe6-1906-4add-b002-67669a0ef76f.jpg)

**Data Ackwolegdement**

A big thank you to kaggler MIKE MAZUROV, you can check out the dataset and also star the project on his github

**Model Building**

To build the safety model, I would use the rescue notebook to build the model, as the steps in model buiding are clearly defined. In the model building process we use the YOLOv5m model with a degree of layer freezing to reduce model training time while sacrificing some degree of mAP. 

You can also check out my kaggle notebook 
(https://www.kaggle.com/code/ugorjiir/safetydetect/notebook) and use the 
kaggle environment to build the model for ease or you can check out the YOLOv5 github on setting up a virtual environment for YOLOv5> Lastly you can use my saved weights in the model folder.

**Model Inference**

For the model Inference process, I recommend starting with the safetytest.py file, in this file i built a class counting syatem, which give the number of type and 
number of classes detected respectively and also, I created a safety rating algorithm, which applies a color tint to the workers detected depending on the level of PPE detected in the worker bounding box. 

The Algorithm has a progression system of Green(all PPE detected on worker), yellow(one PPe detected on worker), and lastly red (no PPE detected on worker).
The testvid.py file shows the model in action on a video stream, with both algorithms working concurrently. Example of the result below.

![list( 955, 742, 1026, 833 ) (0, 255, 255)](https://user-images.githubusercontent.com/66518563/215278365-58883c87-311f-44bc-815c-7b81acf4c310.jpg)

**Result**

Results can be found in the model folder, with a csv file showing various model parameters and some plot images.


https://user-images.githubusercontent.com/66518563/215289444-6f2441cd-26e8-46b1-8468-dea57299b314.mp4




