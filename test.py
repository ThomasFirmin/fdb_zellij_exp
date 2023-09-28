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
from socco2011.functions import functionssocco, dimensions, datasocco
from cec2020.functions import functionscec, datacec

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


def main(argv):
    algorithm = None
    benchmark = None
    function_idx = None
    dimension = None
    try:
        opts, args = getopt.getopt(
            argv,
            "a:b:f:d:i",
            ["algorithm=", "benchmark=", "function=", "dimension=", "id="],
        )
    except getopt.GetoptError:
        print("test.py -a <algorithm> -b <benchmark> -f <function> -d <dimension>")
        sys.exit(2)

    id = ""

    for opt, arg in opts:
        if opt in ("-a", "--algorithm"):
            save = arg
            algorithm = relations[save]
        elif opt in ("-b", "--benchmark"):
            benchmark = arg
        elif opt in ("-f", "--function"):
            function_idx = arg
        elif opt in ("-d", "--dimension"):
            dimension = [int(arg)]
        elif opt in ("-i", "--id"):
            print(arg)
            id = int(arg)

    # Select an algorithm
    if algorithm is None:
        algorithm = list(relations.values())
        save = list(relations.keys())

    # Select a benchmark
    if benchmark is None:
        benchmark = "socco2011"

    # Select a function within a benchmark
    if function_idx is None:
        # Select all functions
        if benchmark == "socco2011":
            function = functionssocco
        elif benchmark == "cec2020":
            function = functionscec
        else:
            print("Unknown benchmark")
            sys.exit(2)
    else:
        # Select one function according to its ID
        if benchmark == "socco2011":
            function = [functionssocco[int(function_idx)]]
        elif benchmark == "cec2020":
            function = [functionscec[int(function_idx)]]

    # Select a dimension
    if dimension is None:
        dimension = dimensions

    # select shift and rotation matrices
    if benchmark == "socco2011":
        if function_idx is None:
            data = datasocco
        else:
            data = [datasocco[int(function_idx)]]
    elif benchmark == "cec2020":
        if function_idx is None:
            data = datacec
        else:
            data = [datacec[int(function_idx)]]

    # Launch optimization process
    for dim in dimension:
        for f, d in zip(function, data):
            # Define decision variables
            values = ArrayVar([], converter=ArrayConverter())
            for i in range(dim):
                values.append(
                    FloatVar(
                        f"d{i}",
                        f.lower,
                        f.upper,
                        tolerance=1e-15,
                        converter=FloatMinmax(),
                    )
                )

            # Initialize shift, rotation and shuffle for benchmark functions
            f.shift = d.shift_data(dim)
            f.rotate = d.rotate_data(dim)
            f.shuffle = d.shuffle_data(dim)

            # Wrap loss function into Zellij loss object
            loss = Loss(
                save=True,
                only_score=True,
            )(f)

            # Initialize optimization algorithm
            alg = algorithm(values, loss)
            # Define stoping criterion
            stop3 = Threshold(loss, "calls", 5000 * dim)
            # Define experiment
            exp = Experiment(
                alg,
                stop3,
                save=f"./experiments_{benchmark}/{save}{id}/D{dim}_{f.__class__.__name__}_save",
            )

            # Run experiment
            exp.run()
            del loss
            del exp


if __name__ == "__main__":
    main(sys.argv[1:])
