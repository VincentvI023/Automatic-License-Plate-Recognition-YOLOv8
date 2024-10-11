# Automatic-Number-Plate-Recognition-YOLOv8

## Model

- A Yolov8 pre-trained model (YOLOv8n) was used to detect vehicles.
- A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 
- The model is available [here](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing).

## Project Setup

* Make an environment with python **3.8** using Conda 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* Run main.py with the video file you want, after setting the correct values on line 15-19 run:
``` python
python main.py
```

Source: https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8 under MIT licence
