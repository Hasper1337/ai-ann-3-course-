# 2 lab

import numpy as np

# класс полносвязной однослойной нейронной сети
class NeuralNetwork:
  weight : np.ndarray
  bias : np.ndarray
  
  # произвольного задания количества нейронов(m) и произвольного количества входов нейронов(n)
  def __init__(self, n: int, m: int):
    # инициализацию значений весов случайными малыми значениями
    self.weight = np.random.uniform(0.001, 0.2, (m, n))

    self.bias = np.random.uniform(0.001, 0.02, m)

  # Функцию активации использовать линейную
  def _activation_function(self, x: np.ndarray) -> int:
    return x
  
  def predict(self, x : np.ndarray) -> np.ndarray:
    z = np.dot(x, self.weight.T) + self.bias

    output = self._activation_function(z)
    return output
  
  def fit_1(self, x : np.ndarray, y : np.ndarray):
    alpha = 0.01 #скорость обучения
    E_m = 0.01 #желлаемая среднеквадр ошибка
    max_epochs = 100

    n_samples = x.shape[0]

    for epoch in range(max_epochs):
      E_total = 0

      for k in range(n_samples):
        x_k = x[k]
        y_k = y[k]

        y_pred = self.predict(x_k)

        error = y_k - y_pred


if __name__ == "__main__":


