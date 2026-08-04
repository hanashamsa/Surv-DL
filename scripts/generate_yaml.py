from pathlib import Path 
import yaml

from config.paths import YOLO_ROOT
from config.classes import CLASSES

def main():
    
    yaml_data = {
        'path' : str(YOLO_ROOT.resolve()),
        'train' : 'images/train',
        'val' : 'images / val',
        'test' : 'images . test',
        'names' : {v: k for k , v in CLASSES.items()}
        
    }
    
    output_file = YOLO_ROOT / 'dataset.yaml'
    
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f'yaml file created : {output_file}')
    
if __name__ == '__main__':
    main()