from models import generator
from preprocessing import mask_images
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


def get_image(path):
    image = plt.imread(path)
    image = image / 255.0
    image = np.array([image])
    image = tf.image.central_crop(image, 1)
    image = tf.image.resize(image, (256, 256))
    return image


def get_mask(x, y, width, height):
    if 0 > x > 256 or 0 > y > 256 or x + width > 256 or x + height > 256 \
            or 0 > width > 80 or 0 > height > 80:
        raise Exception
    mask = np.ones((1, 256, 256, 3), dtype=np.float32)
    mask[:, y:y + height, x:x + width, :] = 0
    return mask


def get_generator(weights_path):
    gen = generator((256, 256, 3))
    gen.load_weights(weights_path)
    return gen


weights_path = "./best_weights/generator_epoch49_metrics0.9265625, 0.996875, 0.0734375, 1303.67607421875.h5"
print("Podaj sciezke do zdjecia:")
image_path = input()

image = get_image(image_path)

print("Podaj ilosc masek:")
masks_number = int(input())

for i in range(masks_number):
    print("Maska {}".format(i+1))
    x = int(input())
    y = int(input())
    width = int(input())
    height = int(input())
    if i == 0:
        masked_image = mask_images(image, get_mask(x, y, width, height))
    else:
        masked_image = mask_images(masked_image, get_mask(x, y, width, height))

gen = get_generator(weights_path)

impainted_image = gen(masked_image)[0]

plt.figure(figsize=(30, 10))
plt.subplot(1, 3, 1)
plt.title("original")
plt.imshow(image[0])
plt.subplot(1, 3, 2)
plt.title("input")
plt.imshow(masked_image[0])
plt.subplot(1, 3, 3)
plt.title("output")
plt.imshow(impainted_image)
plt.show()
