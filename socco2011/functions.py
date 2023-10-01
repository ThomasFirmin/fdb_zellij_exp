import numpy as np
from zellij.utils.benchmarks import (
    Sphere,
    Schwefel_2_21,
    Rosenbrock,
    Rastrigin,
    Griewank,
    Ackley,
    Schwefel_2_22,
    Schwefel_1_2,
    F10,
    Bohachevsky,
    Schaffer,
    CF9F1_25,
    CF9F3_25,
    CF9F4_25,
    CF10F7_25,
    CF9F1_75,
    CF9F3_75,
    CF9F4_75,
    CF10F7_75,
)


class Read_data:
    def __init__(self, func_id):
        self.func_id = func_id

    def shuffle_data(self, d):
        return None

    def shift_data(self, d):
        if self.func_id >= 1 and self.func_id <= 6:
            return np.loadtxt(f"./socco2011/input_data/f{self.func_id}_shift.txt")[:100]
        elif self.func_id == 12 or self.func_id == 16:
            return [
                None,
                np.loadtxt(f"./socco2011/input_data/f{1}_shift.txt")[:100],
            ]
        elif self.func_id == 13 or self.func_id == 17:
            return [
                None,
                np.loadtxt(f"./socco2011/input_data/f{3}_shift.txt")[:100],
            ]
        elif self.func_id == 14 or self.func_id == 18:
            return [
                None,
                np.loadtxt(f"./socco2011/input_data/f{4}_shift.txt")[:100],
            ]
        else:
            return None

    def rotate_data(self, d):
        return None


functionssocco = [
    Sphere(-100, 100, -450, bias=-450),
    Schwefel_2_21(-100, 100, -450, bias=-450),
    Rosenbrock(-100, 100, 390, bias=390),
    Rastrigin(-5, 5, -330, bias=-330),
    Griewank(-600, 600, -180, bias=-180),
    Ackley(-32, 32, -140, bias=-140),
    Schwefel_2_22(-10, 10, 0, bias=0),
    Schwefel_1_2(-65.536, 65.536, 0, bias=0),
    F10(-100, 100, 0, bias=0),
    Bohachevsky(-15, 15, 0, bias=0),
    Schaffer(-100, 100, 0, bias=0),
    # CF9F1_25(-100, 100, 0, bias=0),
    # CF9F3_25(-100, 100, 0, bias=0),
    # CF9F4_25(-5, 5, 0, bias=0),
    # CF10F7_25(-10, 10, 0, bias=0),
    # CF9F1_75(-100, 100, 0, bias=0),
    # CF9F3_75(-100, 100, 0, bias=0),
    # CF9F4_75(-5, 5, 0, bias=0),
    # CF10F7_75(-10, 10, 0, bias=0),
]

dimensions = [10, 15, 20, 30, 50, 100]
datasocco = [
    Read_data(1),
    Read_data(2),
    Read_data(3),
    Read_data(4),
    Read_data(5),
    Read_data(6),
    Read_data(7),
    Read_data(8),
    Read_data(9),
    Read_data(10),
    Read_data(11),
    Read_data(12),
    Read_data(13),
    Read_data(14),
    Read_data(15),
    Read_data(16),
    Read_data(17),
    Read_data(18),
    Read_data(19),
]
