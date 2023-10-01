# Fractal-based decomposition experiments using *Zellij*
Experiments for the paper "A fractal-based decomposition framework for continuous optimization".

## Zellij

Experiments use the in development [Zellij](https://github.com/ThomasFirmin/zellij/) version.
Please intall the Zellij version from [develop_t](https://github.com/ThomasFirmin/zellij/tree/develop_t) branch.

```
$ pip install https://github.com/ThomasFirmin/repository/archive/develop_t
```

## Benchmarks data

Shift vectors, rotation matrices and shuffled dimensions are stored in `cec2020/input_data` from the [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) github, and `socco2011/input_data`.
Functions are defined in `cec2020/functions.py` and `socco2011/functions.py`, and imported from Zellij `benchmarks.py`.

The `wk_price_SP500.txt` dataset, can be downloaded from  original [dataset](https://data.mendeley.com/datasets/g5579mmc9k/2).
(Man-Fai Leung: Datasets for Portfolio Optimization. Mendeley (2022). https://doi.org/10.17632/G5579MMC9K.2)

## Algorithms

Algorithms are defined in `algorithms.py` using Zellij.

Algorithms are instantited within `test.py` and `test_portfolio.py`:
```
$ algo = algorithm(decision_variables, loss_function)
```
Default parameters are used.

## Experiments

Files `test.py` and `test_portfolio` contains code launching experiments.*
Use following commands in a terminal to launch an experiment :
```
$ python3 ./test.py -a FDA -b cec2020 -f 0 -d 10
```
- The option `-a, --algorithm` is for selecting an algorithm within `FDA, DIRECT, DIRECTL, DIRECTR, SOO, DIRECTBs, FDABs, FDADBs, SOOBs, FDAC, FDAD`. If no option is passed, then experiments are computed for all algorithms.
- The option `-b, --benchmark` is for selecting a benchmark within `cec2020, socco2011`.
- The option `-f, --function` is for selecting a function ID within the selected benchmark. Within `[0,9]` for `cec2020`, and `[0,10]` for `socco2011`. If no option is passed, then experiments are computed for all functions.
- The option `-d, --dimension` is for selecting a dimension within `10, 15, 20, 30, 50, 100`. If no option is passed, then experiments are computed for all dimmensions.

For SP500 portfolio optimization:
`wk_price_SP500.txt` must be downloaded and included in the working directory.
```
$ python3 ./test_portfolio.py -a FDA
```
- The option `-a, --algorithm` is for selecting an algorithm within `FDA, DIRECT, DIRECTL, DIRECTR, SOO, DIRECTBs, FDABs, FDADBs, SOOBs, FDAC, FDAD`. If no option is passed, then experiments are computed for all algorithms.

All experimental data will be stored in individual folders for each algorithm, function and dimension, within `experiments_cec2020`, `experiments_socco2011` and `experiments_sp500`.
The total size can reach 30 Gb.

## Analysis

Analysis are made in `analysis.ipynb`.

## Experimental datasets
Firmin, Thomas, 2023, "Experimental results for the paper "A fractal-based decomposition framework for continuous optimization"", https://doi.org/10.57745/0JEUEK, Recherche Data Gouv, V1, UNF:6:lW7nJ0QKwy3UZSdtxZVz/w== [fileUNF] 
