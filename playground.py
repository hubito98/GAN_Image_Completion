from dataset_generator import dataset_generator
from models import generator, generator_2
from preprocessing import mask_images
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

# weights_path = "./weights1/generator_epoch49_metrics0.995703125, 0.93515625, 0.004296875, 1611.6302734375.h5"
# weights_path = "./weights2/generator_epoch49_metrics0.987890625, 0.9890625, 0.012109375, 2273.9853515625.h5"
# weights_path = "./weights3/generator_epoch49_metrics0.55625, 0.999609375, 0.44375, 2267.0431640625.h5"
weights_path = "./best_weights/generator_epoch49_metrics0.981640625, 0.996484375, 0.018359375, 1439.47041015625.h5"

gen = generator((128, 128, 3))
gen.load_weights(weights_path)

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
