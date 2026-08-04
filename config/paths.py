from pathlib import Path 

ROOT = Path(__file__).resolve().parent.parent

VOC_ROOT = ROOT /'dataset'/'VOCdevkit'

VOC2007 = VOC_ROOT / 'VOC2007'
VOC2012 = VOC_ROOT / 'VOC2012'

YOLO_ROOT = ROOT / 'dataset' / 'yolo'

IMAGES_DIR = YOLO_ROOT / 'images'
LABELS_DIR = YOLO_ROOT / 'labels'


TRAIN_IMAGES = IMAGES_DIR / 'train'
VAL_IMAGES = IMAGES_DIR / 'val'
TEST_IMAGES = IMAGES_DIR / 'test'

TRAIN_LABELS = LABELS_DIR / 'train'
VAL_LABELS = LABELS_DIR / 'val'
TEST_LABELS = LABELS_DIR / 'test'

