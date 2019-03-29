import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2DTranspose, Conv2D, Dense, Flatten, \
    Reshape, BatchNormalization, MaxPooling2D, LeakyReLU, concatenate, Lambda, ReLU
import numpy as np
import matplotlib.pyplot as plt
import h5py

tf.enable_eager_execution()


def generator():
    filters = 32
    inputs = Input(shape=(64, 64, 3), dtype=tf.float32, name='inputs')
    c1 = Conv2D(filters=filters, kernel_size=3, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=.001)(c1)
    p1 = MaxPooling2D(pool_size=2)(c1)

    c2 = Conv2D(filters=filters * 2, kernel_size=3, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=.001)(c2)
    p2 = MaxPooling2D(pool_size=2)(c2)

    c3 = Conv2D(filters=filters * 4, kernel_size=3, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=.001)(c3)
    p3 = MaxPooling2D(pool_size=2)(c3)

    c4 = Conv2D(filters=filters * 8, kernel_size=3, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU(alpha=.001)(c4)
    p4 = MaxPooling2D(pool_size=2)(c4)

    c5 = Conv2D(filters=filters * 16, kernel_size=3, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU(alpha=.001)(c5)

    u6 = Conv2DTranspose(filters=filters * 8, kernel_size=3, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU(alpha=.001)(c6)

    u7 = Conv2DTranspose(filters=filters * 4, kernel_size=3, strides=2, padding='same')(c6)
    u7 = concatenate([u7, Lambda(lambda x: x * 0.8)(c3)])
    c7 = Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = LeakyReLU(alpha=.001)(c7)

    u8 = Conv2DTranspose(filters=filters * 2, kernel_size=3, strides=2, padding='same')(c7)
    u8 = concatenate([u8, Lambda(lambda x: x * 0.4)(c2)])
    c8 = Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = LeakyReLU(alpha=.001)(c8)

    u9 = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(c8)
    u9 = concatenate([u9, Lambda(lambda x: x * 0.2)(c1)])
    c9 = Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = LeakyReLU(alpha=.001)(c9)

    outputs = Conv2D(3, kernel_size=1, activation=tf.keras.activations.sigmoid)(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def discriminator():
    vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg.trainable = False
    flat_vgg = Flatten()(vgg.get_layer('block5_pool').output)
    h1 = Dense(128, activation=tf.keras.activations.relu)(flat_vgg)
    outputs = Dense(1, activation=tf.keras.activations.sigmoid)(h1)

    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    return model


# get generator and discriminator model
gen = generator()
dis = discriminator()

# optimizers for both
gen_optimizer = tf.train.AdamOptimizer(1e-4)
dis_optimizer = tf.train.AdamOptimizer(1e-4)

# dataset
(train_x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
train_x = (train_x / 255.0).astype(dtype=np.float32)
# resizing (cifar10 is 32x32x3, but vgg need 64x64x3)
train_x = tf.image.resize_images(train_x, size=(64, 64))

# 6x6 mask for image (for now it's fixed)
mask = tf.ones_like(train_x).numpy()
mask[:, 10:16, 12:18] = 0

# parameters
epoch_num = 5
batch_size = 128

for i in range(epoch_num):
    gen_avg_loss = np.array(0, dtype=np.float32)
    perceptual_accuracy = np.array(0, dtype=np.float32)
    dis_avg_real_accuracy = np.array(0, dtype=np.float32)
    dis_avg_fake_accuracy = np.array(0, dtype=np.float32)

    for step in range(int(len(train_x) / batch_size)):
        real_images = train_x[step * batch_size: (step + 1) * batch_size]
        batch_mask = mask[step * batch_size: (step + 1) * batch_size]
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            gen_output = gen(real_images * batch_mask, training=True)
            dis_fake_output = dis(gen_output, training=True)

            contextual_loss = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * batch_mask, y_pred=gen_output * batch_mask),
                axis=0))
            perceptual_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_fake_output), y_pred=dis_fake_output))
            gen_loss = contextual_loss + 0.3 * perceptual_loss

            dis_fake_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(dis_fake_output), y_pred=dis_fake_output))

            dis_real_output = dis(real_images, training=True)

            dis_real_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_real_output), y_pred=dis_real_output))

            dis_loss = dis_real_loss + dis_fake_loss

            dis_correct_real_preds_number = tf.count_nonzero(dis_real_output >= 0.5).numpy()
            dis_real_preds_len = len(np.array(dis_real_output))
            dis_real_accuracy = dis_correct_real_preds_number / dis_real_preds_len

            dis_correct_fake_preds_number = tf.count_nonzero(dis_fake_output < 0.5).numpy()
            dis_fake_preds_len = len(np.array(dis_fake_output))
            dis_fake_accuracy = dis_correct_fake_preds_number / dis_fake_preds_len

            gen_correct_preds_number = tf.count_nonzero(dis_fake_output >= 0.5).numpy()
            gen_preds_len = len(np.array(dis_fake_output))
            gen_accuracy = gen_correct_preds_number / gen_preds_len

        gen_avg_loss += gen_loss.numpy()
        perceptual_accuracy += gen_accuracy
        dis_avg_real_accuracy += dis_real_accuracy
        dis_avg_fake_accuracy += dis_fake_accuracy

        gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
        dis_gradients = dis_tape.gradient(dis_loss, dis.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
        dis_optimizer.apply_gradients(zip(dis_gradients, dis.trainable_variables))
        if step % 10 == 9:
            print(".", end="")

    print("\nEpisode {}, dis acc: fake {} real {}, perceptual_acc {}, gen_loss: {}".
          format(i, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))

# saving models weights
# gen.save_weights(filepath="./weights/generator.h5")
# dis.save_weights(filepath="./weights/discriminator.h5")


# see how it works on not seen images
(_, _), (test_x, _) = tf.keras.datasets.cifar10.load_data()
test_x = (test_x / 255.0).astype(dtype=np.float32)

test_x = tf.image.resize_images(test_x[:10], size=(64, 64))

image = gen(test_x[:6] * mask[:6], training=True)
for i, im in enumerate(image):
    plt.subplot(3, 6, i + 1)
    plt.imshow(im)

for i in range(6):
    plt.subplot(3, 6, i + 1 + 6)
    plt.imshow(test_x[i] * mask[0])

for i in range(6):
    plt.subplot(3, 6, i + 1 + 12)
    plt.imshow(test_x[i])
plt.show()

