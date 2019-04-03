from dataset_generator import dataset_generator
from models import generator
from preprocessing import mask_images
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

gen = generator((128, 128, 3))
# gen.load_weights("./weights/generator_epoch24_metrics0.89296875, 0.921875, 0.10703125, 2025.21171875.h5")
gen.load_weights("./weights/generator_epoch29_metrics0.994140625, 0.9671875, 0.005859375, 2014.2529296875.h5")

data_gen = dataset_generator(image_dimensions=(256, 256), directory="./images",
                             min_mask=10, max_mask=80, rotation=15, batch_size=3)
data_gen.generate_dataset()

images, masks = data_gen.get_batch()

masked_images = mask_images(images[:1], masks[:1])
# uncomment below to put two or three masks on picture
# masked_images = mask_images(masked_images, masks[1:2])
# masked_images = mask_images(masked_images, masks[2:3])


plt.figure(figsize=(10, 20))
plt.subplot(2, 1, 1)
plt.title("input")
plt.imshow(masked_images[0])
plt.subplot(2, 1, 2)
plt.title("output")
plt.imshow(gen(masked_images)[0])
plt.show()
