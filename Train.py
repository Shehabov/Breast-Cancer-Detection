from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/runs/detect/train3/weights/last.pt")

model.train(data="config.yaml", epochs=300, task='detect', lrf=0.001, lr0=0.001,  batch=32)  # train the model

# continue training the model
# model.train(data="config.yaml", resume=True)


