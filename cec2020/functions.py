import numpy as np
from zellij.utils.benchmarks import (
    Cigar,
    Modified_schwefel,
    Lunacek_bi_rastrigin,
    Expanded_rosenbrock_griewangk,
    H1,
    H2,
    H3,
    C1,
    C2,
    C3,
)


class Read_data:
    def __init__(self, func_id):
        self.func_id = func_id

    def shuffle_data(self, d):
        if (
            self.func_id == 4
            or self.func_id == 6
            or (self.func_id >= 11 and self.func_id <= 20)
        ) and (d > 2 and d <= 100):
            return (
                np.loadtxt(
                    f"./cec2020/input_data/shuffle_data_{self.func_id}_D{d}.txt"
                ).astype(int)
                - 1
            )
        else:
            return None

    def shift_data(self, d):
        if self.func_id == 7:
            res = np.loadtxt(
                f"./cec2020/input_data/shift_data_{self.func_id}.txt"
            )[0]
            return res
        else:
            return np.loadtxt(
                f"./cec2020/input_data/shift_data_{self.func_id}.txt"
            )

    def rotate_data(self, d):
        if self.func_id < 20:
            return np.loadtxt(f"./cec2020/input_data/M_{self.func_id}_D{d}.txt")
        elif self.func_id == 22:
            res = np.loadtxt(f"./cec2020/input_data/M_{self.func_id}_D{d}.txt")[
                : 3 * d
            ]
            shape = (3, int(np.ceil(res.shape[0] / 3)), d)
            return res.reshape(shape)
        elif self.func_id == 24:
            res = np.loadtxt(f"./cec2020/input_data/M_{self.func_id}_D{d}.txt")[
                : 4 * d
            ]
            shape = (4, int(np.ceil(res.shape[0] / 4)), d)
            return res.reshape(shape)
        elif self.func_id == 25:
            res = np.loadtxt(f"./cec2020/input_data/M_{self.func_id}_D{d}.txt")[
                : 5 * d
            ]
            shape = (5, int(np.ceil(res.shape[0] / 5)), d)
            return res.reshape(shape)


functionscec = [
    Cigar(-100, 100, 100, bias=100),
    Modified_schwefel(-100, 100, 1100, bias=1100),
    Lunacek_bi_rastrigin(-100, 100, 700, bias=700),
    Expanded_rosenbrock_griewangk(-100, 100, 1900, bias=1900),
    H1(-100, 100, 1700, bias=1700),
    H2(-100, 100, 1600, bias=1600),
    H3(-100, 100, 2100, bias=2100),
    C1(-100, 100, 2200, bias=2200),
    C2(-100, 100, 2400, bias=2400),
    C3(-100, 100, 2500, bias=2500),
]

dimensions = [10, 15, 20, 30, 50, 100]
datacec = [
    Read_data(1),
    Read_data(2),
    Read_data(3),
    Read_data(7),
    Read_data(4),
    Read_data(16),
    Read_data(6),
    Read_data(22),
    Read_data(24),
    Read_data(25),
]
