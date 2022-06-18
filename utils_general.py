import numpy as np

# Basics
def g3d(x, x0, f):
    c = 343
    r = row_wise_norm2(x - x0)
    k = 2 * np.pi * f / c
    return np.e ** (-1j * k * r) / r


def g3dv(x, x0, f, c=343):
    dif = x - x0
    r = row_wise_norm2(dif)
    k = 2 * np.pi * f / c
    return np.multiply(dif.T, (1 + 1j / (k * r)) * np.e ** (-1j * k * r) / r ** 2)


def square(complex_number):
    return complex_number.real ** 2 + complex_number.imag ** 2


def decibels(pressure):
    return 20 * np.log10(pressure / (2 * 10 ** (-5)))


def pressure(dB):
    return 2 * 10 ** (-5) * 10 ** (dB / 20)


def row_wise_norm2(array, p=2):
    return np.sum(np.abs(array) ** p, axis=-1) ** (1. / p)


def column_wise_norm2(array):
    return np.sum(np.abs(array) ** p, axis=0) ** (1. / p)


def normalized(x):
    norm2 = column_wise_norm2(x)
    norm2[norm2 == 0] = 1
    return x / norm2


def rotation_45(array):
    return np.flip(array.T, axis=0)


def rotation_inverse_45(array):
    return array.T


def print_array_2d(array, name, is_complex):
    if is_complex:
        string = '[[' + '],\n ['.join([', '.join([str(x)[1:-1] for x in row]) for row in array]) + ']]'
    else:
        string = '[[' + '],\n ['.join([', '.join([str(x) for x in row]) for row in array]) + ']]'
    print(name + ' = ' + string + '\n')


def print_array_3d(array, name, is_complex):
    if is_complex:
        string = '[[[' + ']],\n [['.join(
            ['],\n ['.join([', '.join([str(x)[1:-1] for x in r_2]) for r_2 in r_1]) for r_1 in array]) + ']]]'
    else:
        string = '[[[' + ']],\n [['.join(
            ['],\n ['.join([', '.join([str(x) for x in r_2]) for r_2 in r_1]) for r_1 in array]) + ']]]'
    print(name + ' = ' + string + '\n')


def name_counter(type_region, max_count=5):
    count = 0
    while True:
        ratio = count / max_count
        yield f'{type_region} region {count}', COLOR[type_region] * (1 - ratio) + np.asarray([0, 1, 0]) * ratio
        count += 1


COLOR = {'sound': np.asarray([0, 0.5, 0]),
         'silence': np.asarray([0, 0, 0.5]),
         'complete': np.asarray([0, 1, 0])}

BIG_NUMBER = 10 ** 15
SMALL_NUMBER = 10 ** (-15)

SIDES = ['left', 'right']
