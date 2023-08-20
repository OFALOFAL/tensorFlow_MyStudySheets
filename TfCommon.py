import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import itertools
import numpy as np
import zipfile
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

def make_future_forecast(into_future, model, values, window_size, verbose=2):
  """
  Makes future forecasts into_future steps after values ends.

  Returns future forecasts as list of floats.
  """
  values_copy = values.copy()
  future_forecast = []

  for x in range(into_future):
    pred = model.predict(np.expand_dims(values_copy[-window_size:], axis=0))
    if verbose==2:
      print(f'Iterration {x}: \n\tvalues: {values_copy[-window_size:]} \n\tprediction: {pred}')

    values_copy = np.append(values_copy, pred)
    future_forecast.append(pred)
   
  if verbose:
    print()
    print('Predictions:')
    print('-'*15)
    for x, pred in enumerate(future_forecast):
      print(f'Day {x+1}: {pred}')

  return future_forecast

def evaluate_preds(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  ev_dict = {
      'MAE': keras.metrics.mean_absolute_error(y_true, y_pred).numpy(),
      'MASE': mean_absolute_scaled_error(y_true, y_pred).numpy(),
      'MSE': keras.metrics.mean_squared_error(y_true, y_pred).numpy(),
      'rMSE': tf.math.sqrt(keras.metrics.mean_squared_error(y_true, y_pred)).numpy(),
      'MAPE': keras.metrics.mean_absolute_percentage_error(y_true, y_pred).numpy(),
  }
  return ev_dict

def make_ensemble_preds(ensemble_models, data):
  ensemble_preds = []
  for model in ensemble_models:
    preds = model.predict(data)
    ensemble_preds.append(preds)
  return tf.constant(tf.squeeze(ensemble_preds))

def get_ensemble_models(horizon, 
                        train_data,
                        test_data,
                        num_iter=10, 
                        num_epochs=100, 
                        loss_fns=["mae", "mse", "mape"]):
  """
  Returns a list of num_iter models each trained on MAE, MSE and MAPE loss.

  For example, if num_iter=10, a list of 30 trained models will be returned:
  10 * len(["mae", "mse", "mape"]).
  """
  ensemble_models = []

  for i in range(num_iter):
    for loss_function in loss_fns:
      print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")

      model = tf.keras.Sequential([
        layers.Dense(128, kernel_initializer="he_normal", activation="relu"), 
        layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
        layers.Dense(HORIZON)                                 
      ])

      model.compile(
          loss=loss_function,
          optimizer=tf.keras.optimizers.Adam(),
          metrics=["mae", "mse"]
      )
      
      model.fit(
          train_data,
          epochs=num_epochs,
          verbose=0,
          validation_data=test_data,
          callbacks=[
              tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss",
                  patience=200,
                  restore_best_weights=True
              ),
              tf.keras.callbacks.ReduceLROnPlateau(
                  monitor="val_loss",
                  patience=100,
                  verbose=1
              )
          ]
      )
      
      ensemble_models.append(model)

  return ensemble_models

def make_windows(x, window_size=7, horizon=1):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # print(f"Window step:\n {window_step}")

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)

def get_labelled_windows(x, horizon=1):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1 (default)
  Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
  """
  return x[:, :-horizon], x[:, -horizon:]

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)

def pred_timer(model, samples):
  """
  Times how long a model takes to make predictions on samples.
  
  Args:
  ----
  model = a trained model
  sample = a list of samples

  Returns:
  ----
  total_time = total elapsed time for model to make predictions on samples
  time_per_pred = time in seconds per single sample
  """
  start_time = time.perf_counter() # get start time
  model.predict(samples) # make predictions
  end_time = time.perf_counter() # get finish time
  total_time = end_time-start_time # calculate how long predictions took to make
  time_per_pred = total_time/len(val_sentences) # find prediction time per sample
  return total_time, time_per_pred

def predict_on_sentence(model, sentence):
  """
  Uses model to make a prediction on sentence.
  Returns the sentence, the predicted label and the prediction probability.
  """
  pred_prob = model.predict([sentence])
  pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
  print(f"Pred: {pred_label}", "(real disaster)" if pred_label > 0 else "(not real disaster)", f"Prob: {pred_prob[0][0]}")
  print(f"Text:\n{sentence}")

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.io.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

def autolabel(rects): # From: https://matplotlib.org/examples/api/barchart_demo.html
  """
  Attach a text label above each bar displaying its height (it's value).
  Needs ax variable
  """
  for rect in rects:
    width = rect.get_width()
    ax.text(width+0.02, rect.get_y() + rect.get_height(),
            f"{width:.2f}",
            ha='center', va='bottom')

def get_random_img(target_dir, target_class, seed=-1, verbose=1):
  """
  Get random image from directory.
  """
  if seed >= 0:
    random.seed(seed)

  target_folder = (target_dir+'/' if target_dir[-1]!='/' else target_dir) + target_class
  random_img_path = random.sample(os.listdir(target_folder), 1)

  img = mpimg.imread(target_folder+'/'+random_img_path[0])

  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off')

  if verbose:
    print(f'Image shape: {img.shape}')
    print(f'Image name: {random_img_path}')
  return img

def plot_hist_curves(histories, metrics=[], val_data=True, models_names=None, fig_size=(5, 2)):
  """
  Plot histories of passed metrics of many models to compare their loss and accuracy curves.
  Args:
    histories: passed is as list of model histories (any length>0)
    metrics: passed is as list of metrics (different than loss) to plot (any length)
    val_data: True if fit methods of models used validation_data
    fig_size: Sets size of every plot
  """
  metrics_copy= metrics.copy()+['loss']

  def set_ax_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

  if len(histories) > 1:
    if models_names is None:
      models_names=[]
      for x in range(len(histories)):
        models_names.append('model'+str(x))
    if len(models_names) > len(histories):
      for x in range(len(models_names) - len(histories)):
        models_names.pop()
    if len(models_names) < len(histories):
      for x in range(len(histories) - len(models_names)):
        models_names.append('model'+str(len(histories)+x))

    fig, axis = plt.subplots(len(metrics_copy), len(histories))
    fig.suptitle('Models comparison\n', fontweight ="bold")
    for x, history in enumerate(histories):
      for y, metric_name in enumerate(metrics_copy):
        metric = history.history[metric_name]
        if val_data:
          val_metric = history.history['val_'+metric_name]
        else:
          val_metric=None

        epochs = range(len(history.history[metric_name]))

        axis[y, x].plot(epochs, metric, label='train_'+metric_name)
        if val_data:
          axis[y, x].plot(epochs, val_metric, label='val_'+metric_name)

        set_ax_size(fig_size[0]*len(histories), fig_size[1]+len(metrics))
        axis[y, x].set_title(f'{metric_name} {models_names[x]}')
        if x==0:
          if y==len(metrics_copy)-1:
            axis[y, x].set_xlabel('Epochs')
          else:
            axis[y, x].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
          axis[y, x].legend()
        else:
          if y==len(metrics_copy)-1:
            axis[y, x].tick_params(top=False, labeltop=False, left=False, right=False, labelleft=False)
          else:
            axis[y, x].tick_params(top=False, labeltop=False, left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
  else:
    history = histories[0]
    for y, metric_name in enumerate(metrics_copy):
      metric = history.history[metric_name]
      if val_data:
        val_metric = history.history['val_'+metric_name]
      else:
          val_metric=None

      epochs = range(len(history.history[metric_name]))

      plt.figure(figsize=fig_size)
      plt.plot(epochs, metric, label='training_'+metric_name)
      if val_data:
        plt.plot(epochs, val_metric, label='val_'+metric_name)
      plt.title(metric_name)
      plt.xlabel('Epochs')
      plt.legend()

    
def diffrent_data_comparison(data_x, data_y, title_x='DataX', title_y='DataY', batch_size=32, seed=-1):
  """
  Compare 2 image-type data, used to compare augmented and non-augmented data.
  """
  if seed >= 0:
    random.seed(seed)

  images_x, labels_x = data_x.next()
  images_y, labels_y = data_y.next()

  rand_num = random.randint(0, batch_size)

  plt.figure()
  plt.imshow(images_x[rand_num])
  plt.title(title_x)
  plt.axis(False)

  plt.figure()
  plt.imshow(images_y[rand_num])
  plt.title(title_y)
  plt.axis(False);

def predict_img(model, filename, class_names, img_shape=224, verbose=1):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  img = tf.io.read_file(filename)
  # decode and resize
  img = tf.image.resize(tf.image.decode_jpeg(img, channels=3), size=[img_shape, img_shape])
  # normalize
  img = img/255.

  # visualise
  prediction_prabability = model.predict(tf.expand_dims(img, axis=0))
  if len(prediction_prabability[0]) > 1:
    pred_class = class_names[tf.argmax(*prediction_prabability)]
  else:
    pred_class = class_names[int(tf.round(prediction_prabability))]

  if verbose>=0:
    print(f'{pred_class}: {prediction_prabability}')
  # plot
  raw_img = mpimg.imread(filename)
  plt.imshow(raw_img)
  plt.axis(False)
  plt.title(f'Prediction: {pred_class}')

def plot_decision_boundary(model, x, y, verbose=1):
  """
  Plots boundries of moddel decisions on passed data.
  """
  x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
  y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

  x_in = np.c_[xx.ravel(), yy.ravel()]

  y_pred = model.predict(x_in, verbose=verbose)

  if model.output_shape[-1] > 1:
    if verbose:
      print("doing multiclass classification...")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    if verbose:
      print("doing binary classifcation...")
    y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

def sigmoid(x):
  return 1 / (1 + tf.exp(-x))

def reLu(x):
  return tf.maximum(x, 0)

def confiusion_matrix_graph(y_true, y_pred, text_size=15, figsize=(10, 7), classes=False, savefig=False):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  ax.set(
      title='Confiusion Matrix',
      xlabel='Predicted Label',
      ylabel='True Label',
      xticks=np.arange(n_classes),
      yticks=np.arange(n_classes),
      xticklabels=labels,
      yticklabels=labels
  )
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  ax.xaxis.label.set_size(20)
  ax.yaxis.label.set_size(20)

  ax.title.set_size(25)

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)',
             horizontalalignment='center',
             color='white' if cm[i, j] > threshold else 'black',
             size=text_size)
  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

def compare_hystories(original_history, new_history, initial_epochs=5, verbose=1):
  """
  Compares two TensorFlow model History objects.
  
  Args:
    original_history: History object from original model (before new_history)
    new_history: History object from continued model training (after original_history)
    initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    verbose: Decides if function prints out additional information
  """
  # Get original history measurements
  acc = original_history.history["accuracy"]
  loss = original_history.history["loss"]

  if verbose:
    print(len(acc))

  val_acc = original_history.history["val_accuracy"]
  val_loss = original_history.history["val_loss"]

  # Combine original history with new history
  total_acc = acc + new_history.history["accuracy"]
  total_loss = loss + new_history.history["loss"]

  total_val_acc = val_acc + new_history.history["val_accuracy"]
  total_val_loss = val_loss + new_history.history["val_loss"]

  if verbose:
    print(len(total_acc))
    print(total_acc)

  # Make plots
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label='Training Accuracy')
  plt.plot(total_val_acc, label='Validation Accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label='Training Loss')
  plt.plot(total_val_loss, label='Validation Loss')
  plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()
  
def count_trainable_layers(model, ret=0):
  """
  Prints out number of trainable and nontrainable layers in model
  Args:
    model: Model to count the layers on.
    ret: Decides if return value is returned, defaulted to 0 for no return
  """
  trainable_layers = [0, 0]
  for layer_number, layer in enumerate(model.layers):
    trainable_layers[layer.trainable]+=1
  print(f'There are {trainable_layers[0]} untrainable layers and {trainable_layers[1]} trainable layers in model')
  if ret:
    return trainable_layers
