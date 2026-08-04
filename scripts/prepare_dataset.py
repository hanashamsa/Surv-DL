from pathlib import Path
from py_compile import main
import re
import shutil
import random
import stat
from turtle import st
import xml.etree.ElementTree as ET 


import sys
import pathlib

from scripts.convert_voc_to_yolo import convert_box


root_dir = pathlib.Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))




from config.paths import *
from config.classes import CLASSES
from config.settings import *

stats = {
    'images' : 0,
    'labels' : 0,
    'objects' : 0,
    'skipped' : 0
}

def create_folders():
    
    folders = [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
        TEST_IMAGES,
        TEST_LABELS
    ]
    
    for folder in folders:
        folder.mkdir(parents=True,exist_ok=True)
    print('ok')




def convert_bbox( width , height ,xmin , ymin, xmax, ymax):
    
    
    x_center = (( xmin + xmax ) / 2) / width
    y_center = (( ymin + ymax ) / 2) / height
    
    bbox_width = (xmax - xmin ) / width
    bbox_height = (ymax - ymin ) / height
    
    return(
        x_center,
        y_center,
        bbox_height,
        bbox_width
    )
    



def process_xml(xml_file , split):
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    
    yolo_annotations = []
    has_valid_objects = False
    
    # 1. Read all objects first and build annotations list
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        if class_name not in CLASSES:
            stats['skipped'] += 1
            continue
        
        has_valid_objects = True
        class_id = CLASSES[class_name]
        
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x, y , w, h = convert_bbox(
            width,
            height,
            xmin,
            ymin,
            xmax,
            ymax
        )
        
        yolo_annotations.append(
            f'{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}'
        )

        stats['objects'] += 1
        
    # 2. Only exit if no valid target classes were found in the file
    if not has_valid_objects:
        return

    # 3. Determine the output text folder based on the dataset split
    if split == 'train':
        label_folder = TRAIN_LABELS
        image_destination_folder = TRAIN_IMAGES
    elif split == 'val':
        label_folder = VAL_LABELS
        image_destination_folder = VAL_IMAGES
    else:
        label_folder = TEST_LABELS
        image_destination_folder = TEST_IMAGES
        
    label_path = label_folder / f'{xml_file.stem}.txt'
        
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))
            
    stats['labels'] += 1
        
    # 4. Resolve the image source and copy it over to the split folder
    if 'VOC2007' in str(xml_file):
        image_source = VOC2007 / 'JPEGImages' / filename
    else:
        image_source = VOC2012 / 'JPEGImages' / filename # Fixed directory plural 'JPEGImages' typo
        
    image_destination = image_destination_folder / filename
    
    if image_source.exists():
        shutil.copy2(image_source, image_destination)
        stats['images'] += 1
    else:
        print(f"Warning: Missing image file -> {image_source}")
            


def process_dataset():
    random.seed(RANDOM_SEED)
    
    xml_files = []
    
    xml_files.extend((VOC2007 / 'Annotations').glob('*.xml'))
    xml_files.extend((VOC2012 / 'Annotations').glob('*.xml'))
    
    xml_files = list(xml_files)
    
    print(f'ok and the length is  : {len(xml_files)}')
    
    
    random.shuffle(xml_files)
    
    total = len(xml_files)
    
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)
    
    
    train_files = xml_files[:train_end]
    val_files = xml_files[train_end:val_end]
    test_files = xml_files[val_end:]  
    
    print(f'Train , Val , Test  : {len(train_files)} , {len(val_files)} , {len(test_files)}  ')  
    
    for xml in train_files:
        process_xml(xml,'train')
        
    for xml in val_files:
        process_xml(xml,'val')
        
    for xml in test_files:
        process_xml(xml,'test')
    
    
    



def generate_yaml():
    pass



def verify_dataset():
    pass





def print_stats():
    print(f'images copied : {stats["images"]}')
    print(f'labels generated : {stats["labels"]}')
    print(f'objects converted : {stats["objects"]}')
    print(f'object skipped : {stats["skipped"]}')



def main():
    
    print('done')
    
    
    create_folders()
    
    process_dataset()
    
    generate_yaml()
    
    verify_dataset()
    
    print_stats()
    
    
    
if __name__ == '__main__':
    main()
