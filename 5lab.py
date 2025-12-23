# 5 lab

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from keras import Model
from keras.layers import Concatenate
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
  plt.legend()
  plt.show()

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

  predictions = model.predict(np.column_stack((t_train, fi_train)))
  plt.figure(figsize=(10., 10.))
  plt.plot(x, y, c='blue', label='T', marker='X')

  plt.title("\nточность обучения %.2f" % (accuracy*100))
  for i in range(len(X_train)):
      plt.scatter((X_train)[i], predictions[i], c='red', label=('prediction' if i==0 else None))
  #plt.scatter(X_train, predictions, label='Предсказания', color='red')
  plt.scatter(X_test, model.predict(np.column_stack((t_test, fi_test))), label='тест', color='green')
  plt.legend()
  plt.show()

def third_step():

  def sinusoid(x):
    return tf.sin(x)

  x = np.arange(-20, 20, 0.1)
  y = np.sin(x) + np.sin(np.sqrt(2)*x)

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

  model_T = Sequential([
      Dense(32, activation=sinusoid, input_shape=(1,)),
      Dense(1)
  ])

  model_T.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
  model_T.fit(X_train, X_train // (2 * np.pi), epochs=150, batch_size=10)

  model_fi = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(1)
  ])

  model_fi.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
  model_fi.fit(X_train, X_train % (2 * np.pi), epochs=150, batch_size=10)

  input_T = Input(shape=(1,))
  input_fi = Input(shape=(1,))
  concatenated = Concatenate(axis=-1)([model_T(input_T), model_fi(input_fi)])
  output = Dense(16, activation=sinusoid)(concatenated)
  output = Dense(1)(output)

  model_combined = Model(inputs=[input_T, input_fi], outputs=output)

  model_combined.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
  model_combined.fit([X_train, X_train], y_train, epochs=150, batch_size=10)

  _, accuracy = model_combined.evaluate([X_train, X_train], y_train)
  print("Точность на обучающей выборке:", accuracy * 100)

  _, accuracy2 = model_combined.evaluate([X_test, X_test], y_test)
  print("Точность на тестовой выборке:", accuracy2 * 100)

  #y_pred = model_combined.predict([X_train, X_train])

  # plt.figure(figsize=(10., 10.))
  # plt.plot(x, y, c='blue', label='T', marker='X')
  # plt.title("\nточность обучения %.2f" % (accuracy*100))
  # plt.scatter(X_test, y_pred, label='Предсказания', color='red', alpha=1)
  # plt.xlabel('x')
  # plt.ylabel('y')
  # plt.legend()
  # plt.show()

  plt.figure(figsize=(10., 10.))
  plt.plot(x, y, c='blue', label='T', marker='X')

  plt.title("\nточность обучения %.2f" % (accuracy*100))
  plt.scatter(X_train, model_combined.predict([X_train, X_train]), c='red', label='prediction')
  #plt.scatter(X_train, predictions, label='Предсказания', color='red')
  plt.scatter(X_test, model_combined.predict([X_test, X_test]), label='тест', c='green')
  plt.legend()
  plt.show()



if __name__ == '__main__':
  #first_step()
  #second_step()
  third_step()
