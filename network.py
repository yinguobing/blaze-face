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
    # Stacking layers for 16x16 feature maps.
    blaze_layers_16x16 = [
        blaze_stem(24),
        blaze_block(24),
        blaze_block(24),
        blaze_block(48, (2, 2)),
        blaze_block(48),
        blaze_block(48),
        double_blaze_block(24, 96, (2, 2)),
        double_blaze_block(24, 96),
        double_blaze_block(24, 96)]

    # Stacking layers for 8x8 feature maps.
    blaze_layers_8x8 = [
        double_blaze_block(24, 96, (2, 2)),
        double_blaze_block(24, 96),
        double_blaze_block(24, 96)]

    # Inputs defined here.
    inputs = keras.Input((input_shape), dtype=tf.float32)

    # Forward propgation till feature map 16x16 got.
    x = inputs
    for layer in blaze_layers_16x16:
        x = layer(x)
    x_16 = x

    # Keep moving forward till feature map 8x8 got.
    for layer in blaze_layers_8x8:
        x = layer(x)
    x_8 = x

    # Get the classification and box from 16x16 feature map.
    classes_16 = keras.layers.Conv2D(2, (1, 1))(x_16)
    classes_16 = keras.layers.Reshape((-1, 1))(classes_16)
    boxes_16 = keras.layers.Conv2D(4*2, (1, 1))(x_16)
    boxes_16 = keras.layers.Reshape((-1, 4))(boxes_16)

    # Get the classification and box from 8x8 feature map.
    classes_8 = keras.layers.Conv2D(6, (1, 1))(x_8)
    classes_8 = keras.layers.Reshape((-1, 1))(classes_8)
    boxes_8 = keras.layers.Conv2D(4*6, (1, 1))(x_8)
    boxes_8 = keras.layers.Reshape((-1, 4))(boxes_8)

    # Assemble the results.
    classifications = keras.layers.Concatenate(axis=-2)([classes_16, classes_8])
    boxes = keras.layers.Concatenate(axis=-2)([boxes_16, boxes_8])
    outputs = keras.layers.Concatenate(axis=-1)([boxes, classifications])

    # Finally, build the model.
    model = keras.Model(inputs=inputs, outputs=outputs, name='blaze_net')

    return model


if __name__ == "__main__":
    model = blaze_net((128, 128, 3))
    model.summary()
