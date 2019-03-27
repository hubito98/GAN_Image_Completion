#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pathlib import Path
from matplotlib.image import imread
import random

def create_dataset(path, size=100, min_mask_size=20, max_mask_size=40):
    images = []
    masks = []
    names = []
    entries = Path(path)
    for entry in entries.iterdir():
        if entry.suffix == '.jpg':
            names.append(entry.name)
            
    for i in range (0, size):
        name = random.choice(names)
        image = imread(path+name)
        image_float = image.astype(np.float32)
        images.append(image_float)
        
        mask = np.ones_like(image_float)
        
        x = random.randint(0, mask.shape[0]-min_mask_size)
        y = random.randint(0, mask.shape[1]-min_mask_size)
        
        height = random.randint(min_mask_size, max_mask_size)
        width = random.randint(min_mask_size, max_mask_size)
        
        mask[x:x+width, y:y+height] = [0.,0.,0.]
        masks.append(mask)
        
    return images, masks

