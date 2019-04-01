from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class dataset_generator(object):
    
    def __init__(self, image_dimensions=(256,256), rotation=30,
                 batch_size=32, directory='./Data', min_mask=40, max_mask=70):
        self.image_dimensions = image_dimensions
        self.rotation = rotation
        self.batch_size = batch_size
        self.directory = directory
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.create_generator()
        
    def create_generator(self):
        self.generator = keras.preprocessing.image.ImageDataGenerator(dtype=np.float, 
                                                                 rotation_range=self.rotation, 
                                                                 horizontal_flip=True)
    
    def generate_dataset(self):
        self.dataset = self.generator.flow_from_directory(self.directory,
                                                          target_size=self.image_dimensions, batch_size=self.batch_size)
        
    def get_batch(self):
        batch = self.dataset.next()[0]/255.0
        masks = np.ones_like(batch)
        for mask in masks:
            x = randint(0, self.image_dimensions[0] - self.min_mask - 1)
            y = randint(0, self.image_dimensions[1] - self.min_mask - 1)
            height = randint(self.min_mask, self.max_mask)
            width = randint(self.min_mask, self.max_mask)
            mask[x:x+width, y:y+height] = 0
        
        return batch, masks
    

