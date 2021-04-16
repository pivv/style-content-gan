import cv2
import os
from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict

#image = cv2.imread('/data/afhq/train/dog/flickr_dog_000007.jpg')
#print(image.shape)


with open( f'/data/afhq/white_dog.csv', 'r') as f:
    white_dog_names: List[str] = f.read().splitlines()

print(white_dog_names[:10])
