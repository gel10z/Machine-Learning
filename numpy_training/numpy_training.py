import numpy as np

'''1. Подсчитать произведение ненулевых элементов на диагонали прямоугольной матрицы.

2. Даны два вектора x и y. Проверить, задают ли они одно и то же мультимножество.

3. Найти максимальный элемент в векторе x среди элементов, перед которыми стоит нулевой.

4. Операции с изображением.
Дан трёхмерный массив, содержащий изображение, размера (height, width, numChannels),
а также вектор длины numChannels. Сложить каналы изображения с указанными весами,
и вернуть результат в виде матрицы размера (height, width). Преобразовать цветное
изображение в оттенки серого, использовав коэффициенты np.array([0.299, 0.587, 0.114]).

5. Реализовать кодирование длин серий (Run-length encoding). Дан вектор x.
Необходимо вернуть кортеж из двух векторов одинаковой длины. Первый содержит числа,
а второй - сколько раз их нужно повторить.'''

def product_of_diagonal_elements_vectorized(matrix: np.array):
    diagonal = matrix.diagonal()
    return diagonal[np.nonzero(diagonal)].prod()


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    return (np.sort(x) == np.sort(y)).all()


def max_before_zero_vectorized(x: np.array):
    return x[np.where(x[:-1] == 0)[0] + 1].max()


def add_weighted_channels_vectorized(image: np.array):
    return image.dot(np.array([0.299, 0.587, 0.114]))


def run_length_encoding_vectorized(x: np.array):
    x1 = np.concatenate((x[1:], np.array([x[-1]-1]))) != x
    return (x[x1], np.diff(np.concatenate((np.array([-1]), np.where(x1)[0]))))
