import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def simple_generator():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(32, (4, 4), strides=2, padding='same', activation='relu')(inputs)
    x = UpSampling2D()(x)
    outputs = Conv2D(3, (4, 4), padding='same', activation='tanh')(x)
    return Model(inputs, outputs, name='SimpleGenerator')

def simple_discriminator():
    inp = Input(shape=(256, 256, 3))
    tar = Input(shape=(256, 256, 3))
    x = Concatenate()([inp, tar])
    x = Conv2D(32, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    outputs = Conv2D(1, (4, 4), padding='same')(x)
    return Model([inp, tar], outputs, name='SimpleDiscriminator')

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output):
    return loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss

generator = simple_generator()
discriminator = simple_discriminator()
