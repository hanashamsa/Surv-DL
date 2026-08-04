
from importlib.resources import path
import xml.etree.ElementTree as ET

from pathlib import Path 

XML_FILE = Path('../dataset/VOCdevkit/VOC2007/Annotations/000001.xml')

if not XML_FILE.exists():
    raise FileNotFoundError(f'could not find file: {XML_FILE}')

tree = ET.parse(XML_FILE)
root = tree.getroot()

#image info

filename = root.find('filename').text

size = root.find('size')
width = int(size.find('width').text)
height = int(size.find('height').text)


print(f'image:{filename}')
print(f'size : {width} * {height}')

#read all

object = root.findall('object')

print(f'total object : {len(object)}')


for index, obj in enumerate(object,start =1):
    
    name = obj.find('name').text
    
    bbox = obj.find('bndbox')
    
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    print(f'object,class,xmin,ymin,xmax,ymax : {index},{name},{xmin},{ymin},{xmax},{ymax}')
     
