import tensorflow 
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanAbsoluteError,RootMeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.optimizers import Adam

def build_custom(input_shape,dense_units,dropout,learning_rate,data_augmentation=True):

    model = models.Sequential()
    model.add(layers.RandomFlip(mode="horizontal_and_vertical",input_shape=input_shape))
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

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
    return model

def build_vgg(input_shape,dense_units,dropout,learning_rate,data_augmentation=True,pretrained_weights=True):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.vgg16.preprocess_input, input_shape=input_shape))
  if data_augmentation:
    model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.RandomRotation(1.0))
  if pretrained_weights:
    model.add(VGG16(include_top=False,input_shape=input_shape, weights='imagenet'))
  else:
    model.add(VGG16(include_top=False,input_shape=input_shape, weights=None))
  
  for layer in model.layers:
    layer.trainable = False
  
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(dense_units, activation='relu'))
  model.add(layers.Dropout(dropout))  
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model

def build_resnet(input_shape,dense_units,dropout,learning_rate,data_augmentation=True,pretrained_weights=True):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.resnet50.preprocess_input, input_shape=input_shape))
  if data_augmentation:
    model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.RandomRotation(1.0))
  if pretrained_weights:
    model.add(ResNet50(include_top=False,input_shape=input_shape, weights='imagenet'))
  else:
    model.add(ResNet50(include_top=False,input_shape=input_shape, weights=None))

  for layer in model.layers:
    layer.trainable = False
  
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(dense_units, activation='relu'))
  model.add(layers.Dropout(dropout)) 
  model.add(layers.Dense(1, activation='linear'))  

  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model


def build_inception(input_shape,dense_units,dropout,learning_rate,data_augmentation=True,pretrained_weights=True):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.inception_v3.preprocess_input, input_shape=input_shape))
  if data_augmentation:
    model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.RandomRotation(1.0))
  if pretrained_weights:
   inception = InceptionV3(include_top=False,input_shape=input_shape, weights='imagenet')
  else:
    inception = InceptionV3(include_top=False,input_shape=input_shape, weights=None)
  for layer in inception.layers:
    layer.trainable = False
  model.add(inception)

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(dense_units, activation='relu'))
  model.add(layers.Dropout(dropout))
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model


def build_densenet(input_shape,dense_units,dropout,learning_rate,data_augmentation=True,pretrained_weights=True):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.densenet.preprocess_input, input_shape=input_shape))
  if data_augmentation:
    model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.RandomRotation(1.0))
  if pretrained_weights:
    densenet = DenseNet169(include_top=False,input_shape=input_shape, weights='imagenet')
  else:
    densenet = DenseNet169(include_top=False,input_shape=input_shape, weights=None)
  for layer in densenet.layers:
    layer.trainable = False
  model.add(densenet)

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(dense_units, activation='relu'))
  model.add(layers.Dropout(dropout))
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model