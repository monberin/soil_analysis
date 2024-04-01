import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import Hyperband,Objective

def vgg_builder(hp):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.vgg16.preprocess_input, input_shape=(256, 256, 3)))
  model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
  model.add(layers.RandomRotation(1.0))
  model.add(VGG16(include_top=False,input_shape=(256,256,3), weights='imagenet'))

  for layer in model.layers:
    layer.trainable = False
  hp_dense_units = hp.Choice('units', values=[128,256,512,1024,2048])
  hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(hp_dense_units, activation='relu'))
  model.add(layers.Dropout(hp_dropout))  # Add dropout for regularization
  model.add(layers.Dense(1, activation='linear'))  # Use linear for regression

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model

def resnet_builder(hp):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.resnet50.preprocess_input, input_shape=(256, 256, 3)))
  model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
  model.add(layers.RandomRotation(1.0))
  model.add(ResNet50(include_top=False,input_shape=(256,256,3), weights='imagenet'))

  for layer in model.layers:
    layer.trainable = False

  hp_dense_units = hp.Choice('units', values=[128,256,512,1024,2048])
  hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(hp_dense_units, activation='relu'))
  model.add(layers.Dropout(hp_dropout))
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model

def inception_builder(hp):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.inception_v3.preprocess_input, input_shape=(256, 256, 3)))
  model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
  model.add(layers.RandomRotation(1.0))
  inception = InceptionV3(include_top=False,input_shape=(256,256,3), weights='imagenet')
  for layer in inception.layers:
    layer.trainable = False
  model.add(inception)

  hp_dense_units = hp.Choice('units', values=[128,256,512,1024,2048])
  hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(hp_dense_units, activation='relu'))
  model.add(layers.Dropout(hp_dropout))
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model

def densenet_builder(hp):
  model = models.Sequential()
  model.add(layers.Lambda(tensorflow.keras.applications.densenet.preprocess_input, input_shape=(256, 256, 3)))
  model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
  model.add(layers.RandomRotation(1.0))
  densenet = DenseNet169(include_top=False,input_shape=(256,256,3), weights='imagenet')
  for layer in densenet.layers:
    layer.trainable = False
  model.add(densenet)

  hp_dense_units = hp.Choice('units', values=[128,256,512,1024,2048])
  hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(hp_dense_units, activation='relu'))
  model.add(layers.Dropout(hp_dropout))
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
  return model


stop_early = EarlyStopping(monitor='val_loss', patience=3)

def tuner_set_search(builder_func,X_train,y_train,valid_data,proj_name):
  tuner = Hyperband(
    builder_func,
    objective=Objective('val_root_mean_squared_error', direction='min'),
    max_epochs=10,
    factor=3,
    directory='tuner_dir',
    project_name=proj_name,
    overwrite=False
  )

  tuner.search(X_train, y_train, epochs=15, validation_data=valid_data, callbacks=[stop_early])
  best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

  print(f"""
  {proj_name}\n
  The optimal number of units :{best_hps.get('units')} \n
  The optimal dropout: {best_hps.get('dropout')} \n
  The optimal learning rate for the optimizer:{best_hps.get('learning_rate')}.
  """)
  return best_hps

