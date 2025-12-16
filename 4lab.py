import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def first_task():
    # Создаем набор данных X от -20 до 20 с шагом 0.1 и Y
    x = np.arange(-20, 20, 0.1)
    y = np.arange(-20, 20, 0.1)

    # Разделяем на обучающую и тестовую выборки.
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        Dense(1, input_dim=1)
    ])

    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    # Обучаем
    model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=0)

    # Оценка на обучающей и тестовой выборках
    train_loss, train_mae = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)

    # Вывод метрик
    print(f"Обучение — loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
    print(f"Тест      — loss: {test_loss:.6f}, MAE: {test_mae:.6f}")

    # Выводим график функции и точек предсказания нейронки на тестовом наборе.
    plt.scatter(y_test, model.predict(x_test), color='r', label='Predictions')
    plt.plot(x, y, label='Function')
    plt.legend()
    plt.show()

def second_task():

    x = np.arange(-20, 20, 0.1)
    y = np.abs(x)

    # Разделяем на обучающую и тестовую выборки.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(10, activation='relu', input_dim=1),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    x_train = x_train.reshape(-1, 1)
    x_test  = x_test.reshape(-1, 1)

    # Обучаем
    model.fit(x_train, y_train, epochs=10, batch_size=10)

    # Выводим точность на обучающей и тестовой выборке.
    _, accuracy = model.evaluate(x_train, y_train)
    _, accuracy2 = model.evaluate(model.predict(x_test), y_test)

    # Выводим график функции и точек предсказания нейронки на тестовом наборе.
    predictions = model.predict(x_train)

    plt.title("\nточность обучения %.2f, точность на тесте %.2f" % (accuracy*100, accuracy2*100))
    for i in range(len(x_train)):
        plt.scatter((x_train)[i], predictions[i], c='red', label=('prediction' if i==0 else None))

    plt.plot(x, y, label='y=|x|')
    plt.scatter(x_test, model.predict(x_test), c='green', label='тест')
    plt.legend()

def third_task():

    # набор данных
    angles = np.arange(0, 2*np.pi, 0.01)
    radius = 5

    # Координаты точек на окружности
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # Массив с координатами
    coords = np.column_stack([x_coords, y_coords])

    # Разделить на обучающую и тестовую выборки
    angles_train, angles_test, coords_train, coords_test = train_test_split(
        angles, coords, test_size=0.2, random_state=42)

    # Создаем модель
    model = Sequential([
        Dense(64, activation='relu', input_dim=1),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='linear')  # 2 выхода:  x и y координаты
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Нормализация
    angles_train = angles_train.reshape(-1, 1)
    angles_test = angles_test.reshape(-1, 1)

    model.fit(angles_train, coords_train, epochs=100, batch_size=32, verbose=1)

    # Выводим точность на обучающей и тестовой выборке
    _, accuracy = model.evaluate(angles_train, coords_train, verbose=0)
    _, accuracy2 = model.evaluate(angles_test, coords_test, verbose=0)

    # Предсказания
    predictions_test = model.predict(angles_test)

    plt.figure(figsize=(10, 10))
    plt.title("Точность обучения %.4f, точность на тесте %.4f" % (accuracy*100, accuracy2*100))

    # окружность
    circle_angles = np.linspace(0, 2*np.pi, 1000)
    circle_x = radius * np.cos(circle_angles)
    circle_y = radius * np.sin(circle_angles)
    plt.plot(circle_x, circle_y, 'b-', linewidth=2, label='Окружность r=5')

    # Предсказания на тестовой выборке
    plt.scatter(predictions_test[:, 0], predictions_test[:, 1], 
              c='green', s=30, alpha=0.7, marker='^', label='Предсказания (test)')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def fourth_task():

    x = np.arange(-20, 20, 0.1)
    y = np.sin(x)

    # Разделяем на обучающую и тестовую выборки.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = x_train.reshape(-1, 1)
    x_test  = x_test.reshape(-1, 1)

    model = Sequential([
        Dense(1000, input_dim=1, activation='relu'),
        Dense(200, activation='sigmoid'),
        Dense(50, activation='relu'),
        Dense(20, activation='softmax'),
        Dense(1, activation='linear')
    ])

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    model.fit(x_train, y_train, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(x_train, y_train)
    _, accuracy2 = model.evaluate(model.predict(x_test), y_test)

    predictions = model.predict(x_train)
    plt.figure(figsize=(10., 10.))

    plt.title("\nточность обучения %.2f, точность на тесте %.2f" % (accuracy*100, accuracy2*100))
    for i in range(len(x_train)):
        plt.scatter((x_train)[i], predictions[i], c='red', label=('prediction' if i==0 else None))

    plt.plot(x, y, c='blue', label='y=sin(x)', marker='X')
    plt.scatter(x_test, model.predict(x_test), c='green', label='тест')
    plt.legend()

def firth_task():
  #https://www.kaggle.com/datasets/uciml/iris
  
  # Набор данных с Kaggle
  data = pd.read_csv('/content/Iris[1].csv')

  # 4 столбца данных для обучения
  X = data.iloc[:, 1:5].values

  # дает 1 если вид "Iris-setosa", иначе 0
  y = (data['Species'] == 'Iris-setosa').astype(int).values

  # Разделить на обучающую и тестовую выборки
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = Sequential([
      Dense(4, input_dim=4, activation='relu'),
      Dense(1, activation='sigmoid')
  ])

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # обучение
  history = model.fit(X_train, y_train, epochs=50, batch_size=10)

  train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
  test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
  print(f"Train Accuracy: {train_acc:.4f}")
  print(f"Test Accuracy: {test_acc:.4f}")

  # как меняется ошибка в процессе обучения
  plt.plot(history.history['loss'], label='Loss')
  plt.title('Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Binary Crossentropy')
  plt.legend()
  plt.show()

if __name__ == '__main__':
    #first_task()
    #second_task()
    #third_task()
    #fourth_task()
    firth_task()
