import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
# helper function

def create_images_list(directory_path, img_resolution):
  colour_images_list = []
  images_list = []
  labels_list = []
  for file in glob.glob(directory_path):
    img = cv2.imread(file,0) #in grayscale
    img_scaled = cv2.resize(img, img_resolution) #scale down to 256
    images_list.append(img_scaled)
    labels_list.append(file.split('/')[-1])

    imgc = cv2.imread(file,1) #in colour
    imgc_scaled = cv2.resize(imgc, img_resolution) #scale down to 256
    colour_images_list.append(imgc_scaled)
  return images_list,labels_list,colour_images_list


def lin_regression(y,predictions):
  # fitting a linear regression through a set of points
      regr = LinearRegression()
      regr.fit(y,predictions)
      pred = regr.predict(y)
      return pred

def plot_metrics(history,metrics_name_list,val = True):
  fig, ax = plt.subplots(1,len(metrics_name_list),figsize=(12,4))
  axs = ax.ravel()
  i = 0
  for metric_name in metrics_name_list:
    training_m = history.history[metric_name]
    if val:
      val_m = history.history['val_'+metric_name]

    # Create count of the number of epochs
    epoch_count = range(1, len(training_m) + 1)

    # Visualize loss history
    ax[i].plot(epoch_count, training_m, 'r--')
    if val:
      ax[i].plot(epoch_count, val_m, 'b-')
      ax[i].legend(['Training '+metric_name, 'Test '+metric_name])
    else:
      ax[i].legend(['Training '+metric_name])
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel(metric_name)
    i+=1
  plt.tight_layout()
  plt.show()

def plot_predictions(y_val,val_predictions,y_train,train_predictions):
    # Scatter plot of predictions
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    for ax,y_true,y_preds,name in zip(axs,[y_train,y_val],[train_predictions,val_predictions],['Train','Validation']):
      ax.scatter(y_true, y_preds)
      ax.set_xlabel('True Values')
      ax.set_ylabel('Predictions')
      ax.set_title(name)

      ax.set_ylim(min(min(y_preds),min(y_true)), max(max(y_preds),max(y_true)))
      ax.set_xlim(min(y_true), max(y_true))

      lin_pr = lin_regression(y_true,y_preds)
      ax.plot(y_true, lin_pr, color ='black',label='lin.regr. fit through predictions')
      x_vals = np.linspace(min(y_true)-0.1, max(y_true)+0.1,100)
      y_vals = [y_true.mean()] * len(x_vals)
      ax.axline((0, 0), (1, 1), linewidth=2, color='r')
      ax.plot(x_vals, y_vals, color='red', linestyle='--', label='mean of true values')
      ax.legend(loc='upper left')

    plt.tight_layout()
    

    plt.show()