import numpy as np

class Neuron:
    weight: np.ndarray[float] # Вес
    bias: float # Смещение

    def __init__(self, w: np.ndarray[float] = None, b: float = None):
        if w is None:
            self.weight = np.array([-1.0, 1.0])
        else:
            self.weight = w

        if b is None:
            self.bias = 0.5
        else:
            self.bias = b
    
    def _threshold_function(self, u: float) -> int:
        return 1 if u >= 0 else 0

    def predict(self, x: np.ndarray[int]) -> int:
        u = np.dot(self.weight, x) + self.bias
        return self._threshold_function(u)


if __name__ == "__main__":
    neuron = Neuron()
    print(f"Вариант 3 - импликация\nВеса: {neuron.weight}\nСмещение: {neuron.bias}\n")

    test_cases = [
        np.array([0, 0]),  # x1(0) -> x2(0) = 1
        np.array([0, 1]),  # 0 -> 1 = 1
        np.array([1, 0]),  # 1 -> 0 = 0
        np.array([1, 1])  # 1 -> 1 = 1
    ]

    print("Таблица истинности для импликации:\nx1 | x2 | Результат")
    print("-" * 19)
    for i, inputs in enumerate(test_cases):
        result = neuron.predict(inputs)
        print(f" {inputs[0]} |  {inputs[1]} |     {result}")