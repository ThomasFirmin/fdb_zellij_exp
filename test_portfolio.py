from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold
from zellij.utils.converters import FloatMinmax, ArrayConverter

from algorithms import (
    FDA_souquet,
    Direct_c,
    Direct_l,
    Direct_restart,
    Soo_c,
    Direct_Bs,
    FDA_souquet_Bs,
    FDA_souquet_DBs,
    Soo_Bs,
    FDA_souquet_centered,
    FDA_souquet_D,
)

import sys, getopt

relations = {
    "FDA": FDA_souquet,
    "DIRECT": Direct_c,
    "DIRECTL": Direct_l,
    "DIRECTR": Direct_restart,
    "SOO": Soo_c,
    "DIRECTBs": Direct_Bs,
    "FDABs": FDA_souquet_Bs,
    "FDADBs": FDA_souquet_DBs,
    "SOOBs": Soo_Bs,
    "FDAC": FDA_souquet_centered,
    "FDAD": FDA_souquet_D,
}
verbose = False

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.objective_functions import sharpe_ratio
import pandas as pd


# Define Sharpe ratio (objective function)
class Sharpe:
    def __init__(self, dataset):
        self.df = pd.read_csv(dataset, delim_whitespace=True, header=None)
        self.mu = mean_historical_return(self.df)
        self.S = CovarianceShrinkage(self.df).ledoit_wolf()
        self.size = self.df.shape[1]

    def __call__(self, w):
        w /= w.sum()
        return {"sharpe": sharpe_ratio(w, self.mu, self.S, negative=True)}


def main(argv):
    algorithm = None

    try:
        opts, args = getopt.getopt(
            argv,
            "a:b:f:d:i",
            ["algorithm=", "id="],
        )
    except getopt.GetoptError:
        print("test.py -a <algorithm>")
        sys.exit(2)

    id = ""

    for opt, arg in opts:
        if opt in ("-a", "--algorithm"):
            save = arg
            algorithm = relations[save]
        elif opt in ("-i", "--id"):
            print(arg)
            id = int(arg)

    if algorithm is None:
        algorithm = list(relations.values())
        save = list(relations.keys())

    # Define objective function
    f = Sharpe("wk_price_SP500.txt")
    dim = f.size
    # Wrap loss function with Zellij loss object
    loss = Loss(
        save=True,
        only_score=True,
    )(f)

    # Define decision variables
    values = ArrayVar([])
    for i in range(dim):
        values.append(
            FloatVar(
                f"d{i}",
                0.0,
                1.0,
                tolerance=1e-15,
            )
        )
    # Define optimization algorithm
    alg = algorithm(values, loss)
    # Define stoping criterion
    stop3 = Threshold(loss, "calls", 5000 * dim)
    # Define Experiment
    exp = Experiment(
        alg,
        stop3,
        save=f"./experiments_sp500/{save}{id}/D{dim}_{f.__class__.__name__}_save",
    )
    # Launch experiment
    exp.run()
    del loss
    del exp


if __name__ == "__main__":
    main(sys.argv[1:])
