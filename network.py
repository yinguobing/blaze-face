"""Providing network definition for BlazeFace"""
import tensorflow as tf
from tensorflow import keras

from models.blazenet import blaze_block, double_blaze_block


def blaze_stem(filters):
    stem_layers = [
        keras.layers.Conv2D(filters=filters,
                            kernel_size=(5, 5),
                            strides=(2, 2),
                            padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu')]

    def forward(inputs):
        for layer in stem_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def blaze_net(input_shape):
    blaze_layers = [
        blaze_stem(24),
        blaze_block(24),
        blaze_block(24),
        blaze_block(48, (2, 2)),
        blaze_block(48),
        blaze_block(48),
        double_blaze_block(24, 96, (2, 2)),
        double_blaze_block(24, 96),
        double_blaze_block(24, 96),
        double_blaze_block(24, 96, (2, 2)),
        double_blaze_block(23, 96),
        double_blaze_block(24, 96)]

    inputs = keras.Input((input_shape), dtype=tf.float32)
    x = inputs
    for layer in blaze_layers:
        x = layer(x)
    model = keras.Model(inputs=inputs, outputs=x, name='blaze_net')

    return model


if __name__ == "__main__":
    model = blaze_net((128, 128, 3))
    model.summary()
