import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

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

def fourth_task():

    x = np.arange(-20, 20, 0.1)
    y = np.sin(x)

    # Разделяем на обучающую и тестовую выборки.
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    model = Sequential()
    model.add(Dense(1000, input_dim=1, activation='relu'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='rmsprop', loss='mse')

    # Обучаем
    history = model.fit(x_train, y_train, epochs=150, batch_size=10, validation_data=(x_test, y_test), verbose=1)

    y_pred_train = model.predict(x_train, verbose=0)
    y_pred_test = model.predict(x_test, verbose=0)

    # Выводим точность на обучающей и тестовой выборке.
    accuracy = model.evaluate(x_train, y_train)
    accuracy2 = model.evaluate(model.predict(x_test), y_test)
    print(f"Точность на обучающей выборке: {accuracy:.2f}")
    print(f"Точность на тестовой выборке: {accuracy2:.2f}")

    plt.figure(figsize=(12, 8))

    # Синяя линия - истинная функция sin(x)
    plt.plot(x, y, 'b-', label='y = sin(x) (с шагом 0.1)', linewidth=1, marker='o', markersize=2)

    # Красные точки - предсказания на train
    plt.scatter(x_train, y_pred_train, color='red', s=20, label='prediction', alpha=0.7, zorder=5)

    plt.plot(x, y, label='Function')
    plt.title("\nточность обучения %.2f, точность на тесте %.2f" % (accuracy * 100, accuracy2 * 100))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    ##first_task()
    ##second_task()
    fourth_task()
