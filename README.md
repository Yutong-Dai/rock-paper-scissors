# Summary

The code is for the final project titled as **rock-paper-scissors**.
It can detect hand gestures from the live video input and classify it into one of the three classes {paper, rock, scissors}.

The hand detection feature is realized by modifying the general-purpose object detection model [YOLOv5](https://github.com/ultralytics/yolov5).
The gesture classification is realized by **Wenxuan add your description here**.

# Organization of the code

```
- src
    - yolohand: Contains the code for training a hand detector. The majority of the code is cloned from https://github.com/ultralytics/yolov5 with custmization specialized for this project.
        - data: hyper-parameters + yolo-format to specify the train/test/val data
        - models: model architecure; specifically, we adpot the smallest yolov5 model
        - utils: helper functions
        - train: trainning pipline
        - val: validation

- test
    - ./yolohand: Contains all training logs and the final trained model.
    - video: test trained model's hand detection performance.
        

```
