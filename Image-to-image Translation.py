import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Concatenate, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model

def build_generator():
    inputs = Input(shape=[256, 256, 3])

    e1 = Conv2D(64, 4, strides=2, padding='same')(inputs)
    e2 = LeakyReLU()(e1)
    e2 = Conv2D(128, 4, strides=2, padding='same')(e2)
    e2 = BatchNormalization()(e2)
    e3 = LeakyReLU()(e2)
    e3 = Conv2D(256, 4, strides=2, padding='same')(e3)
    e3 = BatchNormalization()(e3)
    e4 = LeakyReLU()(e3)
    e4 = Conv2D(512, 4, strides=2, padding='same')(e4)
    e4 = BatchNormalization()(e4)
    e5 = LeakyReLU()(e4)
    e5 = Conv2D(512, 4, strides=2, padding='same')(e5)
    e5 = BatchNormalization()(e5)

    d1 = UpSampling2D()(e5)
    d1 = Conv2D(512, 4, padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.5)(d1)
    d1 = Concatenate()([d1, e4])
    d2 = UpSampling2D()(d1)
    d2 = Conv2D(256, 4, padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Concatenate()([d2, e3])
    d3 = UpSampling2D()(d2)
    d3 = Conv2D(128, 4, padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Concatenate()([d3, e2])
    d4 = UpSampling2D()(d3)
    d4 = Conv2D(64, 4, padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Concatenate()([d4, e1])
    outputs = UpSampling2D()(d4)
    outputs = Conv2D(3, 4, activation='tanh', padding='same')(outputs)

    return Model(inputs, outputs, name='Generator')

def build_discriminator():
    inp = Input(shape=[256, 256, 3], name='input_image')
    tar = Input(shape=[256, 256, 3], name='target_image')
    x = Concatenate()([inp, tar])
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(1, 4, strides=1, padding='same')(x)
    return Model([inp, tar], x, name='Discriminator')

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (100 * l1_loss)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

generator = build_generator()
discriminator = build_discriminator()
