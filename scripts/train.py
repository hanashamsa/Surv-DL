from ultralytics import YOLO

from config.paths import YOLO_ROOT

MODEL_NAME = 'yolo11n.pt'

def train():
    
    model = YOLO(MODEL_NAME)
    
    model.train(
          
        data =str(YOLO_ROOT / 'dataset.yaml'),
        epochs = 5,
        imgsz = 640,
        batch = 8,
        workers = 2,
        device = 'cpu',
        project = 'runs',
        name = 'demo_surv-dl',
        exist_ok =True,
        pretrained = True,
        verbose = True
    )
    
if __name__ == '__main__':
    train()


