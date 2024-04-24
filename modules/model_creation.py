"""
This module contains functions for building
VGG, ResNet, Inception and DenseNet models, as well as a custom CNN model
"""

import tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.optimizers import Adam


def build_custom(input_shape, dense_units, dropout, learning_rate, data_augmentation=True):
    """
    CNN regression model with 3 Conv and MaxPooling layers; Adam optimizer, MSE loss, metrics: MAE,RMSE.
    input_shape: tuple
    dense_units: int
    dropout: float
    learning_rate: float
    data_augmentation: bool
    """
    model = models.Sequential()
    if data_augmentation:
        # data augmentation layers are only applied on the train set
        model.add(layers.RandomFlip(
            mode="horizontal_and_vertical", input_shape=input_shape))
        # random rotation of the image in the range [-2pi, 2pi]
        model.add(layers.RandomRotation(1.0))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[
                  MeanAbsoluteError(), RootMeanSquaredError()])
    return model


def build_vgg(input_shape, dense_units, dropout, learning_rate, data_augmentation=True, pretrained_weights=True):
    """
    VGG16-based regression model with no top and GlobalAveragePooling2D, Dense and Dropout layers added;
    If pretrained_weights=True, pretrained weights are set to untrainable;
    If pretrained_weights=False initialised with random weights, all trainable.
    Adam optimizer, MSE loss, metrics: MAE,RMSE.
    input_shape: tuple
    dense_units: int
    dropout: float
    learning_rate: float
    data_augmentation: bool
    """
    model = models.Sequential()
    if data_augmentation:
        # data augmentation layers are only applied on the train set
        model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
        # random rotation of the image in the range [-2pi, 2pi]
        model.add(layers.RandomRotation(1.0))
    if pretrained_weights:
        model.add(layers.Lambda(tensorflow.keras.applications.vgg16.preprocess_input,
                  input_shape=input_shape))  # VGG image preprocessing function for imagenet weights
        model.add(VGG16(include_top=False,
                  input_shape=input_shape, weights='imagenet'))
        for layer in model.layers:
            layer.trainable = False
    else:
        model.add(VGG16(include_top=False, input_shape=input_shape, weights=None))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[
                  MeanAbsoluteError(), RootMeanSquaredError()])
    return model


def build_resnet(input_shape, dense_units, dropout, learning_rate, data_augmentation=True, pretrained_weights=True):
    """
    ResNet50-based regression model with no top and GlobalAveragePooling2D, Dense and Dropout layers added;
    If pretrained_weights=True, pretrained weights are set to untrainable;
    If pretrained_weights=False initialised with random weights, all trainable.
    Adam optimizer, MSE loss, metrics: MAE,RMSE.
    input_shape: tuple
    dense_units: int
    dropout: float
    learning_rate: float
    data_augmentation: bool
    """
    model = models.Sequential()
    if data_augmentation:
        # data augmentation layers are only applied on the train set
        model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
        # random rotation of the image in the range [-2pi, 2pi]
        model.add(layers.RandomRotation(1.0))
    if pretrained_weights:
        model.add(layers.Lambda(tensorflow.keras.applications.resnet50.preprocess_input,
                  input_shape=input_shape))  # ResNet image preprocessing function for imagenet weights
        model.add(ResNet50(include_top=False,
                  input_shape=input_shape, weights='imagenet'))
        for layer in model.layers:
            layer.trainable = False
    else:
        model.add(ResNet50(include_top=False,
                  input_shape=input_shape, weights=None))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[
                  MeanAbsoluteError(), RootMeanSquaredError()])
    return model


def build_inception(input_shape, dense_units, dropout, learning_rate, data_augmentation=True, pretrained_weights=True):
    """
    InceptionV3-based regression model with no top and GlobalAveragePooling2D, Dense and Dropout layers added;
    If pretrained_weights=True, pretrained weights are set to untrainable;
    If pretrained_weights=False initialised with random weights, all trainable.
    Adam optimizer, MSE loss, metrics: MAE,RMSE.
    input_shape: tuple
    dense_units: int
    dropout: float
    learning_rate: float
    data_augmentation: bool
    """
    model = models.Sequential()
    if data_augmentation:
        # data augmentation layers are only applied on the train set
        model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
        # random rotation of the image in the range [-2pi, 2pi]
        model.add(layers.RandomRotation(1.0))
    if pretrained_weights:
        model.add(layers.Lambda(
            tensorflow.keras.applications.inception_v3.preprocess_input, input_shape=input_shape))  # Inception image preprocessing function for imagenet weights
        inception = InceptionV3(
            include_top=False, input_shape=input_shape, weights='imagenet')
        for layer in inception.layers:
            layer.trainable = False
    else:
        inception = InceptionV3(
            include_top=False, input_shape=input_shape, weights=None)

    model.add(inception)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[
                  MeanAbsoluteError(), RootMeanSquaredError()])
    return model


def build_densenet(input_shape, dense_units, dropout, learning_rate, data_augmentation=True, pretrained_weights=True):
    """
    InceptionV3-based regression model with no top and GlobalAveragePooling2D, Dense and Dropout layers added;
    If pretrained_weights=True, pretrained weights are set to untrainable;
    If pretrained_weights=False initialised with random weights, all trainable.
    Adam optimizer, MSE loss, metrics: MAE,RMSE.
    input_shape: tuple
    dense_units: int
    dropout: float
    learning_rate: float
    data_augmentation: bool
    """
    model = models.Sequential()
    if data_augmentation:  # data augmentation layers are only applied on the train set
        model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
        # random rotation of the image in the range [-2pi, 2pi]
        model.add(layers.RandomRotation(1.0))
    if pretrained_weights:
        model.add(layers.Lambda(
            tensorflow.keras.applications.densenet.preprocess_input, input_shape=input_shape))  # DenseNet image preprocessing function for imagenet weights
        densenet = DenseNet169(
            include_top=False, input_shape=input_shape, weights='imagenet')
        for layer in densenet.layers:
            layer.trainable = False
    else:
        densenet = DenseNet169(
            include_top=False, input_shape=input_shape, weights=None)

    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[
                  MeanAbsoluteError(), RootMeanSquaredError()])
    return model
