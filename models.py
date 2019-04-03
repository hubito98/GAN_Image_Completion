import tensorflow as tf


def generator(input_shape=(64, 64, 3)):
    filters = 64
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='inputs')
    c1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.LeakyReLU(alpha=.001)(c1)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=2)(c1)

    c2 = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=3, padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.LeakyReLU(alpha=.001)(c2)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=2)(c2)

    c3 = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=3, padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.LeakyReLU(alpha=.001)(c3)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=2)(c3)

    c4 = tf.keras.layers.Conv2D(filters=filters * 8, kernel_size=3, padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.LeakyReLU(alpha=.001)(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=2)(c4)

    c5 = tf.keras.layers.Conv2D(filters=filters * 16, kernel_size=3, padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.LeakyReLU(alpha=.001)(c5)

    u6 = tf.keras.layers.Conv2DTranspose(filters=filters * 8, kernel_size=3, strides=2, padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.LeakyReLU(alpha=.001)(c6)

    u7 = tf.keras.layers.Conv2DTranspose(filters=filters * 4, kernel_size=3, strides=2, padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, tf.keras.layers.Lambda(lambda x: x * 0.8)(c3)])
    c7 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.LeakyReLU(alpha=.001)(c7)

    u8 = tf.keras.layers.Conv2DTranspose(filters=filters * 2, kernel_size=3, strides=2, padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, tf.keras.layers.Lambda(lambda x: x * 0.4)(c2)])
    c8 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.LeakyReLU(alpha=.001)(c8)

    u9 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, tf.keras.layers.Lambda(lambda x: x * 0.2)(c1)])
    c9 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.LeakyReLU(alpha=.001)(c9)

    outputs = tf.keras.layers.Conv2D(3, kernel_size=1, activation=tf.keras.activations.sigmoid)(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def generator_2(input_shape=(64, 64, 3)):
    filters = 64
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='inputs')
    c1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=7, padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.LeakyReLU(alpha=.001)(c1)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=2)(c1)

    c2 = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=5, padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.LeakyReLU(alpha=.001)(c2)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=2)(c2)

    c3 = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=3, padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.LeakyReLU(alpha=.001)(c3)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=2)(c3)

    c4 = tf.keras.layers.Conv2D(filters=filters * 8, kernel_size=3, padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.LeakyReLU(alpha=.001)(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=2)(c4)

    c5 = tf.keras.layers.Conv2D(filters=filters * 16, kernel_size=3, padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.LeakyReLU(alpha=.001)(c5)

    u6 = tf.keras.layers.Conv2DTranspose(filters=filters * 8, kernel_size=3, strides=2, padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.LeakyReLU(alpha=.001)(c6)

    u7 = tf.keras.layers.Conv2DTranspose(filters=filters * 4, kernel_size=3, strides=2, padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, tf.keras.layers.Lambda(lambda x: x * 0.8)(c3)])
    c7 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.LeakyReLU(alpha=.001)(c7)

    u8 = tf.keras.layers.Conv2DTranspose(filters=filters * 2, kernel_size=3, strides=2, padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, tf.keras.layers.Lambda(lambda x: x * 0.4)(c2)])
    c8 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.LeakyReLU(alpha=.001)(c8)

    u9 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, tf.keras.layers.Lambda(lambda x: x * 0.2)(c1)])
    c9 = tf.keras.layers.Conv2D(filters=12 * 8, kernel_size=3, padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.LeakyReLU(alpha=.001)(c9)

    outputs = tf.keras.layers.Conv2D(3, kernel_size=1, activation=tf.keras.activations.sigmoid)(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def discriminator(input_shape=(64, 64, 3)):
    vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg.trainable = False
    flat_vgg = tf.keras.layers.Flatten()(vgg.get_layer('block5_pool').output)
    h1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(flat_vgg)
    h2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(h1)
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(h2)

    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    return model
