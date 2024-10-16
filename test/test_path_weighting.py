import numpy as np


def calc_path_weighting1(paths, max_age):
    sum = 0
    for path in paths:
        sum += 1 / (1 - path / (max_age + 1))
    result = 1 - 1 / sum
    return result


def calc_path_weighting2(paths, max_age):
    sum = 0
    for path in paths:
        sum += (1 - path / (max_age + 1))
    result = 1 - sum / len(paths)
    return result


def calc_path_weighting3(paths, max_age):
    result = np.sum(paths) / max_age
    result = np.clip(result, 0, 1)
    return result


max_age = 2000

paths = [10]
print(calc_path_weighting1(paths, max_age))
print(calc_path_weighting2(paths, max_age))
print(calc_path_weighting3(paths, max_age))
print()

paths = [0, 1, 2, 500]
print(calc_path_weighting1(paths, max_age))
print(calc_path_weighting2(paths, max_age))
print(calc_path_weighting3(paths, max_age))
print()

paths = [0, 1, 100, 1000]
print(calc_path_weighting1(paths, max_age))
print(calc_path_weighting2(paths, max_age))
print(calc_path_weighting3(paths, max_age))
print()

paths = [1000]
print(calc_path_weighting1(paths, max_age))
print(calc_path_weighting2(paths, max_age))
print(calc_path_weighting3(paths, max_age))
print()

paths = [700, 800, 900, 1000]
print(calc_path_weighting1(paths, max_age))
print(calc_path_weighting2(paths, max_age))
print(calc_path_weighting3(paths, max_age))
print()
