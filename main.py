import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import display
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 禁用GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_custom_images(image_dir, image_size=(256, 256)):
    image_list = []
    for file_path in glob.glob(os.path.join(image_dir, '*.png')):
        image = load_img(file_path, target_size=image_size, color_mode='rgb')
        image = img_to_array(image)
        image = (image - 127.5) / 127.5  # 归一化到[-1, 1]
        image_list.append(image)
    return np.array(image_list)


# 设定图像文件夹路径
image_dir = './images'  # 根据你的实际路径修改
images = load_custom_images(image_dir)

# 创建TensorFlow数据集
BUFFER_SIZE = len(images)
BATCH_SIZE = 4  # 根据你的GPU能力调整
train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(256, 256, 3)),
        layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


# Loss and optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Seed to visualize progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(damaged_images, original_images):
    with tf.GradientTape() as tape:
        restored_images = generator(damaged_images, training=True)
        loss = tf.reduce_mean(tf.abs(original_images - restored_images))

    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, input_images, epoch):
    predictions = model(input_images, training=False)
    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig(f'complement/image_at_epoch_{epoch}_{i}.png')


# Load and prepare the dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Batch and shuffle the data
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Start the training
train(train_dataset, EPOCHS)
