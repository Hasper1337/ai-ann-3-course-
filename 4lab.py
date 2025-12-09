import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def first_task():
    # Создаем набор данных X от -20 до 20 с шагом 0.1 и Y
    x = np.arange(-20, 20, 0.1)
    y = np.arange(-20, 20, 0.1)

    # Разделяем на обучающую и тестовую выборки.
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(Dense(units=1, input_dim=1))

    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Обучаем
    model.fit(x_train, y_train, epochs=10, batch_size=4)

    # Выводим точность на обучающей и тестовой выборке.
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точность на обучающей выборке: {train_loss:.4f}")
    print(f"Точность на тестовой выборке: {test_loss:.4f}")

    # Выводим график функции и точек предсказания нейронки на тестовом наборе.
    plt.scatter(y_test, model.predict(x_test), color='r', label='Predictions')
    plt.plot(x, y, label='Function')
    plt.legend()
    plt.show()

def second_task():

    x = np.arange(-20, 20, 0.1)
    y = np.abs(x)

    # Разделяем на обучающую и тестовую выборки.
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Обучаем
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    # Выводим точность на обучающей и тестовой выборке.
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точность на обучающей выборке: {train_loss:.4f}")
    print(f"Точность на тестовой выборке: {test_loss:.4f}")

    # Выводим график функции и точек предсказания нейронки на тестовом наборе.
    plt.scatter(y_test, model.predict(x_test), color='r', label='Predictions')
    plt.plot(x, y, label='Function')
    plt.legend()
    plt.show()

def third_task():

    x = np.arange(-20, 20, 0.1)
    y = np.sin(x)

    # Разделяем на обучающую и тестовую выборки.
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Обучаем
    model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test))

    # Выводим точность на обучающей и тестовой выборке.
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точность на обучающей выборке: {train_loss:.4f}")
    print(f"Точность на тестовой выборке: {test_loss:.4f}")

    # Выводим график функции и точек предсказания нейронки на тестовом наборе.
    plt.scatter(y_test, model.predict(x_test), color='r', label='Predictions')
    plt.plot(x, y, label='Function')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    ##first_task()
    second_task()
    ##third_task()
