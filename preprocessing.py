import numpy as np


def mask_images(images, mask):
    return images * mask + (np.abs(mask - 1) / 2.0)
