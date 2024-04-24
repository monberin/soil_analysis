"""
This module contains the main k-fold validation loop, as well as the configuration dictionary for model hyperparameters
"""
import numpy as np
from prep_functions import scale_fit_transform
from plot_functions import plot_metrics, plot_predictions, plot_all_validation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# hyperparameter dictionary
model_dict = {
    'XGBRegressor': {
        'n_estimators': 300,
        'max_depth': 2,
        'learning_rate': 0.02
    },
    'RandomForest': {
        'n_estimators': 40,
        'max_depth': 3,
    },
    'VGG': {
        'input_shape': (256, 256, 3),
        'dense_units': 512,
        'dropout': 0.1,
        'learning_rate': 0.01
    },

    'ResNet': {
        'input_shape': (256, 256, 3),
        'dense_units': 128,
        'dropout': 0.4,
        'learning_rate': 0.001
    },

    'Inception': {
        'input_shape': (256, 256, 3),
        'dense_units': 512,
        'dropout': 0.3,
        'learning_rate': 0.001
    },

    'DenseNet': {
        'input_shape': (256, 256, 3),
        'dense_units': 512,
        'dropout': 0,
        'learning_rate': 0.01
    },
}


def run_5fold(model_name, ds_name, i_name, model_dict, splits, X_list, y_list, is_cnn, cnn_build_func=None):
    """
    Main k-fold validation loop. 
    model_name: string; name of the model used(for hyperparameters and plots)
    ds_name: string; name of the dataset used(for plots)
    i_name: string; name of the indicator used(for plots)
    model_dict: dict; hyperparameter dictionary
    splits: list; k-fold splits
    X_list: numpy array; array of images
    y_list: numpy array; array of the responce values
    is_cnn: bool; is the model used a CNN or other ML model
    cnn_build_func: func; build function of the CNN used; None if not a CNN. 
    """
    all_val_preds = []
    all_val_true = []
    i = 0
    # interation over the splits
    for (train_idx, test_idx) in splits:
        print('-'*66)
        print('-'*30+f'Fold {i}'+'-'*30)
        print('-'*66)
        i += 1
        # selecting the  values on the indexes in the current split
        X_train = np.take(X_list, train_idx, axis=0)
        y_train = np.take(y_list, train_idx, axis=0)
        X_val = np.take(X_list, test_idx, axis=0)
        y_val = np.take(y_list, test_idx, axis=0)
        # standardising the responce values
        y_train, y_val = scale_fit_transform(y_train, y_val)

        # retrieving the hyperparameters
        params = model_dict[model_name]
        # if CNN, build the model with the hyperparameters, fit, and plot metrics
        if is_cnn:
            model = cnn_build_func(
                params['input_shape'], params['dense_units'], params['dropout'], params['learning_rate'])
            h = model.fit(X_train, y_train, epochs=20,
                          batch_size=32, validation_data=(X_val, y_val))
            plot_metrics(h, ['loss', 'root_mean_squared_error', 'mean_absolute_error'],
                         f'{ds_name} {i_name}: {model_name} fold {i}')
        #if not CNN, scale the input vectors, build and fit the model with the hyperparameters
        else:
            X_train, X_val = scale_fit_transform(X_train, X_val)
            if model_name == 'XGBRegressor':
                model = XGBRegressor(
                    n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'])
            if model_name == 'RandomForest':
                model = RandomForestRegressor(
                    max_depth=params['max_depth'], n_estimators=params['n_estimators'])
            model.fit(X_train, y_train.ravel())

        # predict the values on train and validation sets
        y_val_pred = model.predict(X_val)
        y_train_pred = model.predict(X_train)
        all_val_preds.append(y_val_pred)
        all_val_true.append(y_val)
        
        # plot current fold results
        plot_predictions(y_val.reshape(-1, 1), y_val_pred, y_train.reshape(-1, 1),
                         y_train_pred, f'{ds_name} {i_name}: {model_name}  fold {i}')
        print(
            f'Train RMSE: {mean_squared_error(y_train,y_train_pred,squared=False)}, MAE: {mean_absolute_error(y_train,y_train_pred)}')
        print(
            f'Validation RMSE: {mean_squared_error(y_val,y_val_pred,squared=False)}, MAE: {mean_absolute_error(y_val,y_val_pred)}')

    # plot all 5-fold validation
    plot_all_validation(all_val_true, all_val_preds,
                        f'{ds_name} {i_name}: {model_name} 5-fold total validation')

    print('Total validation RMSE:', mean_squared_error(np.concatenate(
        all_val_true).ravel(), np.concatenate(all_val_preds).ravel(), squared=False))
    print('Total validation MAE:', mean_absolute_error(np.concatenate(
        all_val_true).ravel(), np.concatenate(all_val_preds).ravel()))
