import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import itertools
import numpy as np

def get_random_img(target_dir, target_class, seed=-1, verbose=1):
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
