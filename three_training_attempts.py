import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from models import generator, discriminator, generator_2
from preprocessing import mask_images
from dataset_generator import dataset_generator

tf.enable_eager_execution()

######################### FIRST ATTEMPT ###########################

# learning hyperparameters
epoch_num = 50
batch_size = 8
gen_lr = 1e-4
dis_lr = 1e-5
min_mask = 10
max_mask = 80
image_rotation = 15
images_in_epoch = batch_size * 320

# get generator and discriminator model
gen = generator(input_shape=(256, 256, 3))
dis = discriminator(input_shape=(256, 256, 3))

# optimizers for both
gen_optimizer = tf.train.AdamOptimizer(gen_lr)
dis_optimizer = tf.train.AdamOptimizer(dis_lr)

# dataset generator - basically "infinity" dataset
dataset_generator = dataset_generator(image_dimensions=(256, 256), directory="./images",
                                      min_mask=min_mask, max_mask=max_mask, rotation=image_rotation,
                                      batch_size=batch_size)
dataset_generator.generate_dataset()

# training data logging
file = open("loss1.csv", "w+")
file.write("episode;fake_acc;real_acc;perceptual_acc;gen_loss\n")
for episode in range(epoch_num):
    # arrays for epoch summary
    gen_avg_loss = np.array(0, dtype=np.float32)
    perceptual_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range
    dis_avg_real_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range
    dis_avg_fake_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range

    for step in range(int(images_in_epoch / batch_size)):
        # get train batch for epoch
        real_images, batch_mask = dataset_generator.get_batch()
        masked_images = mask_images(images=real_images, mask=batch_mask)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # we want this to look like normal image, no holes
            gen_output = gen(masked_images, training=True)

            # discriminator prediction of photos, whether they are real or generated by generator
            dis_fake_output = dis(gen_output, training=True)
            dis_real_output = dis(real_images, training=True)

            # generator losses
            contextual_loss_valid = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * batch_mask, y_pred=gen_output * batch_mask),
                axis=0))
            contextual_loss_hole = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * np.abs(1 - batch_mask),
                                                    y_pred=gen_output * np.abs(1 - batch_mask)), axis=0))
            contextual_loss = contextual_loss_valid + 6 * contextual_loss_hole
            perceptual_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_fake_output), y_pred=dis_fake_output))
            gen_loss = contextual_loss + 0.1 * perceptual_loss

            # discriminator losses
            dis_fake_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(dis_fake_output), y_pred=dis_fake_output))
            dis_real_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_real_output), y_pred=dis_real_output))
            dis_loss = dis_real_loss + dis_fake_loss

            # values for epoch summary
            dis_correct_real_preds_number = tf.count_nonzero(dis_real_output >= 0.5).numpy()
            dis_real_preds_len = len(np.array(dis_real_output))
            dis_real_accuracy = dis_correct_real_preds_number / dis_real_preds_len

            dis_correct_fake_preds_number = tf.count_nonzero(dis_fake_output < 0.5).numpy()
            dis_fake_preds_len = len(np.array(dis_fake_output))
            dis_fake_accuracy = dis_correct_fake_preds_number / dis_fake_preds_len

            gen_correct_preds_number = tf.count_nonzero(dis_fake_output >= 0.5).numpy()
            gen_preds_len = len(np.array(dis_fake_output))
            gen_accuracy = gen_correct_preds_number / gen_preds_len

        # also values for epoch summary
        gen_avg_loss += gen_loss.numpy()
        perceptual_accuracy += gen_accuracy
        dis_avg_real_accuracy += dis_real_accuracy
        dis_avg_fake_accuracy += dis_fake_accuracy

        # update discriminator weights for 5 episodes then only 1 per 4 epoch
        gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
        if episode < 5 or episode % 8 == 0:
            dis_gradients = dis_tape.gradient(dis_loss, dis.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
        if episode < 5 or episode % 8 == 0:
            dis_optimizer.apply_gradients(zip(dis_gradients, dis.trainable_variables))

        # after every 40 step put "."
        if step % 40 == 39:
            print(".", end="")
    file.write("{};{};{};{};{}\n".
               format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                      perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
    print("\nEpisode {}, dis acc: fake {} real {}, perceptual_acc {}, gen_loss: {}".
          format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
    if episode % 10 == 9:
        # saving generator and discriminator weights with info in name after every epoch
        gen.save_weights(filepath="./weights1/generator_epoch{}_metrics{}, {}, {}, {}.h5"
                         .format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
        dis.save_weights(filepath="./weights1/discriminator_epoch{}_metrics{}, {}, {}, {}.h5"
                         .format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
file.close()


######################### SECOND ATTEMPT ###########################
# differents:
# - contexutal_loss of hole times 12 insted times 6
# - perceptual_loss times 0.2 instead 0.1

# learning hyperparameters
epoch_num = 50
batch_size = 8
gen_lr = 1e-4
dis_lr = 1e-5
min_mask = 10
max_mask = 80
image_rotation = 15
images_in_epoch = batch_size * 320

# get generator and discriminator model
gen = generator(input_shape=(256, 256, 3))
dis = discriminator(input_shape=(256, 256, 3))

# optimizers for both
gen_optimizer = tf.train.AdamOptimizer(gen_lr)
dis_optimizer = tf.train.AdamOptimizer(dis_lr)

# training data logging
file = open("loss2.csv", "w+")
file.write("episode;fake_acc;real_acc;perceptual_acc;gen_loss\n")
for episode in range(epoch_num):
    # arrays for epoch summary
    gen_avg_loss = np.array(0, dtype=np.float32)
    perceptual_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range
    dis_avg_real_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range
    dis_avg_fake_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range

    for step in range(int(images_in_epoch / batch_size)):
        # get train batch for epoch
        real_images, batch_mask = dataset_generator.get_batch()
        masked_images = mask_images(images=real_images, mask=batch_mask)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # we want this to look like normal image, no holes
            gen_output = gen(masked_images, training=True)

            # discriminator prediction of photos, whether they are real or generated by generator
            dis_fake_output = dis(gen_output, training=True)
            dis_real_output = dis(real_images, training=True)

            # generator losses
            contextual_loss_valid = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * batch_mask, y_pred=gen_output * batch_mask),
                axis=0))
            contextual_loss_hole = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * np.abs(1 - batch_mask),
                                                    y_pred=gen_output * np.abs(1 - batch_mask)), axis=0))
            contextual_loss = contextual_loss_valid + 12 * contextual_loss_hole
            perceptual_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_fake_output), y_pred=dis_fake_output))
            gen_loss = contextual_loss + 0.2 * perceptual_loss

            # discriminator losses
            dis_fake_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(dis_fake_output), y_pred=dis_fake_output))
            dis_real_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_real_output), y_pred=dis_real_output))
            dis_loss = dis_real_loss + dis_fake_loss

            # values for epoch summary
            dis_correct_real_preds_number = tf.count_nonzero(dis_real_output >= 0.5).numpy()
            dis_real_preds_len = len(np.array(dis_real_output))
            dis_real_accuracy = dis_correct_real_preds_number / dis_real_preds_len

            dis_correct_fake_preds_number = tf.count_nonzero(dis_fake_output < 0.5).numpy()
            dis_fake_preds_len = len(np.array(dis_fake_output))
            dis_fake_accuracy = dis_correct_fake_preds_number / dis_fake_preds_len

            gen_correct_preds_number = tf.count_nonzero(dis_fake_output >= 0.5).numpy()
            gen_preds_len = len(np.array(dis_fake_output))
            gen_accuracy = gen_correct_preds_number / gen_preds_len

        # also values for epoch summary
        gen_avg_loss += gen_loss.numpy()
        perceptual_accuracy += gen_accuracy
        dis_avg_real_accuracy += dis_real_accuracy
        dis_avg_fake_accuracy += dis_fake_accuracy

        # update discriminator weights for 5 episodes then only 1 per 4 epoch
        gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
        if episode < 5 or episode % 8 == 0:
            dis_gradients = dis_tape.gradient(dis_loss, dis.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
        if episode < 5 or episode % 8 == 0:
            dis_optimizer.apply_gradients(zip(dis_gradients, dis.trainable_variables))

        # after every 40 step put "."
        if step % 40 == 39:
            print(".", end="")
    file.write("{};{};{};{};{}\n".
               format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                      perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
    print("\nEpisode {}, dis acc: fake {} real {}, perceptual_acc {}, gen_loss: {}".
          format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
    if episode % 10 == 9:
        # saving generator and discriminator weights with info in name after every epoch
        gen.save_weights(filepath="./weights2/generator_epoch{}_metrics{}, {}, {}, {}.h5"
                         .format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
        dis.save_weights(filepath="./weights2/discriminator_epoch{}_metrics{}, {}, {}, {}.h5"
                         .format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
file.close()

######################### THIRD ATTEMPT ###########################
# diffs:
#     - same as SECOND ATTEMPT
#     - after 10 episodes discriminator stop to learn
#     - first two kernel_size is 7 and 5, was 3 and 3


# learning hyperparameters
epoch_num = 50
batch_size = 8
gen_lr = 1e-4
dis_lr = 1e-5
min_mask = 10
max_mask = 80
image_rotation = 15
images_in_epoch = batch_size * 320

# get generator and discriminator model
gen = generator_2(input_shape=(256, 256, 3))
dis = discriminator(input_shape=(256, 256, 3))

# optimizers for both
gen_optimizer = tf.train.AdamOptimizer(gen_lr)
dis_optimizer = tf.train.AdamOptimizer(dis_lr)

# training data logging
file = open("loss3.csv", "w+")
file.write("episode;fake_acc;real_acc;perceptual_acc;gen_loss\n")
for episode in range(epoch_num):
    # arrays for epoch summary
    gen_avg_loss = np.array(0, dtype=np.float32)
    perceptual_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range
    dis_avg_real_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range
    dis_avg_fake_accuracy = np.array(0, dtype=np.float32)  # in 0..1 range

    for step in range(int(images_in_epoch / batch_size)):
        # get train batch for epoch
        real_images, batch_mask = dataset_generator.get_batch()
        masked_images = mask_images(images=real_images, mask=batch_mask)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # we want this to look like normal image, no holes
            gen_output = gen(masked_images, training=True)

            # discriminator prediction of photos, whether they are real or generated by generator
            dis_fake_output = dis(gen_output, training=True)
            dis_real_output = dis(real_images, training=True)

            # generator losses
            contextual_loss_valid = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * batch_mask, y_pred=gen_output * batch_mask),
                axis=0))
            contextual_loss_hole = tf.reduce_sum(tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(y_true=real_images * np.abs(1 - batch_mask),
                                                    y_pred=gen_output * np.abs(1 - batch_mask)), axis=0))
            contextual_loss = contextual_loss_valid + 12 * contextual_loss_hole
            perceptual_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_fake_output), y_pred=dis_fake_output))
            gen_loss = contextual_loss + 0.2 * perceptual_loss

            # discriminator losses
            dis_fake_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(dis_fake_output), y_pred=dis_fake_output))
            dis_real_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(dis_real_output), y_pred=dis_real_output))
            dis_loss = dis_real_loss + dis_fake_loss

            # values for epoch summary
            dis_correct_real_preds_number = tf.count_nonzero(dis_real_output >= 0.5).numpy()
            dis_real_preds_len = len(np.array(dis_real_output))
            dis_real_accuracy = dis_correct_real_preds_number / dis_real_preds_len

            dis_correct_fake_preds_number = tf.count_nonzero(dis_fake_output < 0.5).numpy()
            dis_fake_preds_len = len(np.array(dis_fake_output))
            dis_fake_accuracy = dis_correct_fake_preds_number / dis_fake_preds_len

            gen_correct_preds_number = tf.count_nonzero(dis_fake_output >= 0.5).numpy()
            gen_preds_len = len(np.array(dis_fake_output))
            gen_accuracy = gen_correct_preds_number / gen_preds_len

        # also values for epoch summary
        gen_avg_loss += gen_loss.numpy()
        perceptual_accuracy += gen_accuracy
        dis_avg_real_accuracy += dis_real_accuracy
        dis_avg_fake_accuracy += dis_fake_accuracy

        # update discriminator weights for 5 episodes then only 1 per 4 epoch
        gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
        if episode < 10:
            dis_gradients = dis_tape.gradient(dis_loss, dis.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
        if episode < 10:
            dis_optimizer.apply_gradients(zip(dis_gradients, dis.trainable_variables))

        # after every 40 step put "."
        if step % 40 == 39:
            print(".", end="")
    file.write("{};{};{};{};{}\n".
               format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                      perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
    print("\nEpisode {}, dis acc: fake {} real {}, perceptual_acc {}, gen_loss: {}".
          format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
    if episode % 10 == 9:
        # saving generator and discriminator weights with info in name after every epoch
        gen.save_weights(filepath="./weights3/generator_epoch{}_metrics{}, {}, {}, {}.h5"
                         .format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                                 perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
        if episode < 10:
            dis.save_weights(filepath="./weights3/discriminator_epoch{}_metrics{}, {}, {}, {}.h5"
                             .format(episode, dis_avg_fake_accuracy / (step + 1), dis_avg_real_accuracy / (step + 1),
                                     perceptual_accuracy / (step + 1), gen_avg_loss / (step + 1)))
file.close()
