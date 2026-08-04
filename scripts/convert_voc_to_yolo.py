import xml.etree.ElementTree as ET 
from pathlib import Path

VOC_ANNOTATIONS = Path('../dataset/VOCdevkit/VOC2007/Annotations')

YOLO_LABELS = Path('../dataset/yolo/labels')

CLASSES = {
    'person': 0,
    'bottle': 1,
    'chair': 2,
    'tvmonitor': 3
}


# VOC box to YOLO

def convert_box(size,box):
    width, height = size 
    
    
    xmin,ymin,xmax,ymax = box 
    
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height 
    
    return(
        x_center,
        y_center,
        box_width,
        box_height
    )

    
    
#read XML

xml_files = sorted(VOC_ANNOTATIONS.glob('*.xml'))

print(f'found {len(xml_files)} annotation files')


for xml_file in xml_files:
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        
        class_name = obj.find('name').text
        
        if class_name not in CLASSES:
            continue
        
        class_id = CLASSES[class_name]
        
        bbox = obj.find('bndbox') 
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x, y , w, h = convert_box(
            (width , height),
            (xmin, ymin, xmax, ymax)
        )
    
        line = f'{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}'
        yolo_lines.append(line)
        
    output_file = YOLO_LABELS / (xml_file.stem + '.txt')
    
    with open(output_file , 'w') as f:
        f.write('\n'.join(yolo_lines))
        
print('successful') 
 