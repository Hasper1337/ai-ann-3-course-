# 5 lab

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd


def first_step():

  def sinusoid(x):
      return tf.sin(x)
  
  x = np.arange(-20, 20, 0.1)
  y = np.sin(x) + np.sin(np.sqrt(2)*x)
  # Разделить на обучающую и тестовую выборки.
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=13)

  model = Sequential([
      Dense(8, input_dim=1, activation=sinusoid),
      Dense(4, activation=sinusoid),
      Dense(1)
  ])

  # Обучить с указанными параметрами.
  model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
  model.fit(X_train, y_train, epochs=150, batch_size=10)

  # Посчитать и вывести точность на обучающем и тестовом наборах данных.
  _, accuracy = model.evaluate(X_train, y_train)
  _, accuracy2 = model.evaluate(X_test, y_test)

  predictions = model.predict(X_train)
  plt.figure(figsize=(10., 10.))

  plt.title("\nточность обучения %.2f" % (accuracy*100))
  for i in range(len(X_train)):
      plt.scatter((X_train)[i], predictions[i], c='red', label=('prediction' if i==0 else None))

  plt.plot(x, y, c='blue', label='T', marker='X')
  plt.scatter(X_test, model.predict(X_test), c='green', label='тест')

def second_step():

  def sinusoid(x):
      return tf.sin(x)
  
  x = np.arange(-20, 20, 0.1)
  y = np.sin(x) + np.sin(np.sqrt(2)*x)

  
  # Разделить на обучающую и тестовую выборки.
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=13)

  t_train = X_train // (2*np.pi)
  fi_train = X_train % (2*np.pi)

  t_test = X_test // (2*np.pi)
  fi_test = X_test % (2*np.pi)

  model = Sequential([
      Dense(64, input_dim=2, activation='relu'),
      Dense(32, activation='relu'),
      Dense(20, activation=sinusoid),
      Dense(8, activation=sinusoid),
      Dense(1)
  ])

  # Обучить с указанными параметрами.
  model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
  model.fit(np.column_stack((t_train, fi_train)), y_train, epochs=150, batch_size=10)

  # Посчитать и вывести точность на обучающем и тестовом наборах данных.
  _, accuracy = model.evaluate(np.column_stack((t_train, fi_train)), y_train)
  _, accuracy2 = model.evaluate(np.column_stack((t_test, fi_test)), y_test)

  predictions = model.predict(np.column_stack((t_test, fi_test)))
  plt.figure(figsize=(10., 10.))
  plt.plot(x, y, c='blue', label='T', marker='X')
  #plt.scatter(X_test, model.predict(np.column_stack((t_test, fi_test))), c='green', label='тест')

  plt.title("\nточность обучения %.2f" % (accuracy*100))
  plt.scatter(X_test, predictions, label='Предсказания', color='red') 



if __name__ == '__main__':
  #first_step()
  second_step()
