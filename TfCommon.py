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

def plot_loss_curves(histories):
  """
  Plot histories of many models to compare their loss and accuracy curves.
  Args:
    histories: passed is as list of model histories (any length)
  """
  if len(histories) > 1:
    plt.figure(figsize=(5*len(histories), 10))
    figure, axis = plt.subplots(2, len(histories))
    figure.suptitle('Models comparison\n', fontweight ="bold")
    for x, history in enumerate(histories):
      loss = history.history['loss']
      val_loss = history.history['val_loss']

      accuracy = history.history['accuracy']
      val_accuracy = history.history['val_accuracy']

      epochs = range(len(history.history['loss']))

      axis[0, x].plot(epochs, loss, label='training_loss')
      axis[0, x].plot(epochs, val_loss, label='val_loss')
      axis[0, x].set_title(f'Loss m{x}')
      if x==0:
        axis[0, x].tick_params(bottom = False, labelbottom = False)
        axis[0, x].legend()
      else:
        axis[0, x].tick_params(left = False, right = False, labelleft = False,
                labelbottom = False, bottom = False)

      axis[1, x].plot(epochs, accuracy, label='training_accuracy')
      axis[1, x].plot(epochs, val_accuracy, label='val_accuracy')
      axis[1, x].set_title(f'Accuracy m{x}')
      if x==0:
        axis[1, x].set_xlabel('Epochs')
        axis[1, x].legend()
      else:
        axis[1, x].tick_params(left = False, right = False, labelleft = False)
  else:
    history = histories[0]
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
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

def confiusion_matrix_graph(y_true, y_pred, text_size=15, figsize=(10, 7), classes=False):
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

def compare_historys(original_history, new_history, initial_epochs=5, verbose=1):
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
