import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Parameters
EPOCHS = 50
BATCH_SIZE = 32
IMAGE_SIZE = (50, 50)
NOISE_DIM = 100


# Load images
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=IMAGE_SIZE)
        img = img_to_array(img)
        img = (img - 127.5) / 127.5
        images.append(img)
    return np.array(images)


damaged_images = load_images("damaged")


# 定义生成器
# 定义生成器
def make_generator_model():
    # 生成器现在接收两个输入：噪声和部分遮挡的图像
    noise = layers.Input(shape=(NOISE_DIM,))
    img = layers.Input(shape=(50, 50, 3))

    x = layers.Dense(13 * 13 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((13, 13, 256))(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    x = layers.Cropping2D(cropping=((1, 1), (1, 1)))(x)

    # 这里我们将部分遮挡的图像和生成的图像连接起来
    x = layers.Concatenate()([x, img])

    # 最后一层是一个卷积层，它将两个输入合并成一个输出
    x = layers.Conv2D(3, (3, 3), padding='same')(x)

    return tf.keras.Model([noise, img], x)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 确保同时传递噪声和图像
        generated_images = generator([noise, images], training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# 定义判别器
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[50, 50, 3]))  # 注意这里改为50x50尺寸
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()

# Define the loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Define the training step
@tf.function
def train_step(images):
    # 使用实际的批次大小来生成噪声
    current_batch_size = tf.shape(images)[0]
    noise = tf.random.normal([current_batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, images], training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Define the training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)


# Prepare the dataset
dataset = tf.data.Dataset.from_tensor_slices(damaged_images).shuffle(len(damaged_images)).batch(BATCH_SIZE)

# Train the model
train(dataset, EPOCHS)


def load_raws(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path)  # 不指定target_size，保持原始尺寸
        img = img_to_array(img)
        img = (img - 127.5) / 127.5  # 归一化
        images.append(img)
        filenames.append(filename)  # 保存文件名
    return np.array(images), filenames


def save_images(num_images, directory, original_images, filenames):
    noise = tf.random.normal([num_images, NOISE_DIM])
    dummy_images = tf.zeros([num_images, 50, 50, 3])
    predictions = generator([noise, dummy_images], training=False)

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, img in enumerate(predictions):
        img = img.numpy()  # Convert to NumPy array
        img = ((img + 1) * 127.5).astype(np.uint8)
        img_patch = Image.fromarray(img)

        # 加载原始图像并确认其尺寸
        original_img = Image.fromarray((original_images[i] * 127.5 + 127.5).astype(np.uint8))
        print(f"Original image size: {original_img.size}")  # 打印原始图像尺寸

        # 计算中心位置
        center_x = (original_img.width - img_patch.width) // 2
        center_y = (original_img.height - img_patch.height) // 2
        print(f"Insert position: ({center_x}, {center_y})")  # 打印插入位置

        # 将生成的图像块插入到原始图像的中心
        original_img.paste(img_patch, (center_x, center_y))

        # 保存修改后的图像，使用原始文件名
        original_img.save(os.path.join(directory, filenames[i]))


# 调用函数，传递原始图像数组和文件名
damaged_images, filenames = load_raws("damaged")
save_images(len(damaged_images), "complement", damaged_images, filenames)
