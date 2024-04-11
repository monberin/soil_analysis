from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def lin_regression(y,predictions):
  # fitting a linear regression through a set of points
      regr = LinearRegression()
      regr.fit(y,predictions)
      pred = regr.predict(y)
      return pred

def plot_metrics(history,metrics_name_list,val = True,name=''):
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
  fig.subplots_adjust(top=0.8)
  fig.suptitle(name,y=0.98)
  plt.show()


def plot_all_validation(list_y_true,list_y_preds,name):
    # Scatter plot of predictions
    colours = ['blue','green','yellow','orange','red']
    i = 0
    for c,y_true, y_preds in zip(colours,list_y_true,list_y_preds):
      plt.scatter(y_true, y_preds,color=c,alpha=0.5,label=f'fold {i}')
      plt.xlabel('True Values')
      plt.ylabel('Predictions')
      i+=1
    
    flat_y_true = np.concatenate(list_y_true).ravel()
    # flat_y_preds = [el for sub in list_y_preds for el in list_y_preds]
    flat_y_preds = np.concatenate(list_y_preds).ravel()

    plt.ylim(min(min(flat_y_preds),min(flat_y_true)), max(max(flat_y_preds),max(flat_y_true)))
    plt.xlim(min(flat_y_true), max(flat_y_true))

    lin_pr = lin_regression(flat_y_true.reshape(-1, 1),flat_y_preds.reshape(-1, 1))
    plt.plot(flat_y_true, lin_pr, color ='black',label='lin.regr. fit through predictions')
    x_vals = np.linspace(min(flat_y_true)-0.1, max(flat_y_true)+0.1,100)
    y_vals = [flat_y_true.mean()] * len(x_vals)
    plt.axline((0, 0), (1, 1), linewidth=2, color='r')
    plt.plot(x_vals, y_vals, color='red', linestyle='--', label='mean of true values')
    plt.legend(loc='upper left')
    plt.title(name)

    plt.show()

def plot_predictions(y_val,val_predictions,y_train,train_predictions,gen_name):
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
    fig.suptitle(gen_name)
    

    plt.show()
