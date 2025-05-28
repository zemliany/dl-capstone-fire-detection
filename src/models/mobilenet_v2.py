#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam #


def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=1, learning_rate=0.001):
    """
    MobileNetV2 model for binary image classification for fire/non_fire images

    :param input_shape: Size of input images (height, width, channels).
    :param num_classes: Number of output classes (1 for binary classification).
    :param learning_rate: Learning rate for Adam optimizer.
    :return: Compiled Keras model.
    """

    # Loading MobileNetV2 with weights pre-trained on ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def unfreeze_and_compile(model: Model, learning_rate=0.0001, unfreeze_layers_from_end=50):
    """
    Unfreezes the top layers of MobileNetV2 for fine-tuning

    :param model: The previous Keras model to unfreeze.
    :param learning_rate: The learning rate for the Adam optimizer for fine-tuning.
    :param unfreeze_layers_from_end: The number of layers to unfreeze from the end of the base model.
    :return: The recompiled Keras model.
    """
    # Unfreeze the top layers with BatchNormalization
    for layer in model.layers[-unfreeze_layers_from_end:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print("Run MobileNetV2 test...")
    test_model = build_mobilenet_model()
    test_model.summary()
    print("\nThe model is built and compiled..")

    trainable_count = sum(1 for layer in test_model.layers if layer.trainable)
    print(f"Number of trained layers after build_mobilenet_model: {trainable_count}")

    test_model_unfrozen = unfreeze_and_compile(test_model, unfreeze_layers_from_end=30)
    test_model_unfrozen.summary()
    trainable_count_unfrozen = sum(1 for layer in test_model_unfrozen.layers if layer.trainable)
    print(f"Number of trained layers after unfreeze_and_compile: {trainable_count_unfrozen}")