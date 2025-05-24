from ultralytics import YOLO

model = YOLO("yolov11n-face.pt")  
model.train(data="Multi Face Biometry 2.v1i.yolov11/data.yaml", epochs=75, imgsz=640)