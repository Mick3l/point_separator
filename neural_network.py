import numpy as np


class neural_network(object):
    def __init__(self, input_dim=None, h_dim=None, output_dim=None, **kwargs):
        if 'cache' in kwargs.keys():
            self._upload_from_cache(kwargs['cache'])
        else:
            self.INPUT_DIM = input_dim
            self.H_DIM = h_dim
            self.OUTPUT_DIM = output_dim
            self.ALPHA = kwargs.get('alpha', 1e-3)
            self.W1 = np.random.rand(self.INPUT_DIM, self.H_DIM)
            self.b1 = np.random.rand(1, self.H_DIM)
            self.W2 = np.random.rand(self.H_DIM, self.OUTPUT_DIM)
            self.b2 = np.random.rand(1, self.OUTPUT_DIM)

    # todo
    def _upload_from_cache(self, cache_path):
        ...

    def cache(self):
        with open('neural_network_cache.nnc', 'w') as out:
            out.write(f'{self.INPUT_DIM} {self.H_DIM} {self.OUTPUT_DIM} {self.ALPHA}\n')
            out.write(self.W1.__str__() + '\n')
            out.write(self.b1.__str__() + '\n')
            out.write(self.W2.__str__() + '\n')
            out.write(self.b2.__str__() + '\n')

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def _relu_deriv(x: np.ndarray) -> np.ndarray:
        return np.array([[0. if i < 0 else 1. for i in x[0]]]).astype(float)

    @staticmethod
    def _softmax(a: np.ndarray) -> np.ndarray:
        temp = np.exp(a)
        return temp / np.sum(temp)

    def _study(self, x: np.ndarray, y: np.ndarray):
        t1 = x @ self.W1 + self.b1
        h1 = neural_network._relu(t1)
        t2 = h1 @ self.W2 + self.b2

        z = neural_network._softmax(t2)
        E = -np.sum(np.log(z) @ y.T)
        print(E)

        dE_dt2 = z - y
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ self.W2.T
        dE_dt1 = dE_dh1 * neural_network._relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        self.W1 -= self.ALPHA * dE_dW1
        self.b1 -= self.ALPHA * dE_db1
        self.W2 -= self.ALPHA * dE_dW2
        self.b2 -= self.ALPHA * dE_db2

    def study(self, data: list, results: list) -> None:
        if len(data) != len(results):
            raise ValueError("neural_network.study(): size of data is not equal to size of answers")
        for i in range(len(data)):
            self._study(data[i], results[i])

    def get_ans(self, x: np.ndarray) -> np.ndarray:
        t1 = x @ self.W1 + self.b1
        h1 = neural_network._relu(t1)
        t2 = h1 @ self.W2 + self.b2

        return neural_network._softmax(t2)
