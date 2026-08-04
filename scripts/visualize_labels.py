import random 
import cv2
from pathlib import Path

from config.paths import TRAIN_IMAGES , TRAIN_LABELS
from config.classes import ID_TO_CLASS

IMAGE_DIR = TRAIN_IMAGES
LABEL_DIR = TRAIN_LABELS



images = list(IMAGE_DIR.glob('*.jpg'))

print(f'imade dir , num of image : {IMAGE_DIR}, {len(images)}')

sample_size = min(len(images),10)
sample = random.sample(images , sample_size)

for img in sample:
    print(img)

for image_path in sample:
    
    image = cv2.imread(str(image_path))
    print(image_path)
    
    if image is None:
        print('failed to read image')
        continue
    
    h, w = image.shape[:2]
             
    label_path = LABEL_DIR / (image_path.stem + '.txt')
    
    if not label_path.exists():
        continue
    
    with open(label_path) as f :
        lines = f.readlines()
        
    for line in lines:
        parts = line.split()
        
        if len(parts) !=5:
            continue
        
        cls, x, y, bw, bh = map(float, parts)
        
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        
        
        class_name = ID_TO_CLASS[int(cls)]
        
        cv2.rectangle(image, (x1,y1) , (x2,y2), (0,255,0) , 2)
        
        cv2.putText(
            image,
            class_name,
            (x1 ,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )
        
        cv2.imwrite(
            f'outputs/{image_path.stem}.jpg',
            image
        )
        print(f'saved{image_path.stem}')
        
        
cv2.destroyAllWindows()
        