import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet169

def build_inception(input_shape,pretrained_weights=True,data_augmentation=True,dropout=0):
  model = models.Sequential()
  if data_augmentation:
    model.add(layers.RandomFlip(mode="horizontal_and_vertical",input_shape=input_shape))
    model.add(layers.RandomRotation(1.0))
  if pretrained_weights:
    inception = InceptionV3(include_top=False,input_shape=input_shape, weights='imagenet')
  else:
    inception = InceptionV3(include_top=False,input_shape=input_shape)

  for layer in inception.layers:
    layer.trainable = False

  model.add(inception)

  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(dropout))  # Add dropout for regularization
  model.add(layers.Dense(1, activation='linear'))  # Use linear for regression
  return model


def build_densenet(input_shape,pretrained_weights=True,data_augmentation=True,dropout=0):
  model = models.Sequential()
  if data_augmentation:
    model.add(layers.RandomFlip(mode="horizontal_and_vertical",input_shape=input_shape))
    model.add(layers.RandomRotation(1.0))
  if pretrained_weights:
    dn = DenseNet169(include_top=False,input_shape=input_shape, weights='imagenet')
  else:
    dn = DenseNet169(include_top=False,input_shape=input_shape)

  for layer in dn.layers:
    layer.trainable = False

  model.add(dn)


  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(dropout))  # Add dropout for regularization
  model.add(layers.Dense(1, activation='linear'))  # Use linear for regression
  return model
