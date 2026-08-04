from pathlib import Path


from config.paths import TRAIN_IMAGES ,VAL_IMAGES , TEST_IMAGES
from config.paths import TRAIN_LABELS , VAL_LABELS , TEST_LABELS

def verify_split(images_dir , labels_dir , split_name ):
    
    images = {p.stem for p in images_dir.glob('*.jpg')}
    labels = {p.stem for p in labels_dir.glob('*.txt')}
    
    missing_labels = images - labels
    missing_images = labels - images 
    
    
    print(f'[{split_name.upper()}] images , labels : {len(images)} , {len(labels)}')

    
    if not missing_labels:
        print('every imag have labels')
        
    else:
        print('missing label')
        
    if not missing_images:
        print('every label has image')
        
    else:
        print('missing image')
    
def main():
    verify_split(TRAIN_IMAGES ,TRAIN_LABELS, 'train')
    verify_split(VAL_IMAGES , VAL_LABELS, 'val')
    verify_split(TEST_IMAGES , TEST_LABELS , 'test')
    
    print('verified')
    
if __name__ == '__main__':
    main()