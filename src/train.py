from ultralytics import yolo



model = YOLO('yolo11n.pt')


model.train(
    data = '/dataset/yolo/dataset.yaml',
    epochs = 50,
    imgsz = 640,
    batch = 16,
    project = 'runs',
    name = 'demo_surv-dl'
)


