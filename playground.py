from dataset_generator import dataset_generator
from models import generator
from preprocessing import mask_images
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

gen = generator((128, 128, 3))
gen.load_weights("./weights/generator.h5")

data_gen = dataset_generator(image_dimensions=(128, 128), directory="./images",
                             min_mask=10, max_mask=10, rotation=10, batch_size=3)
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
