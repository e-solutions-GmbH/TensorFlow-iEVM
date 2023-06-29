import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable TensorFlow warnings

import argparse
import logging
import random
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.framework import random_seed

from tf_ievm.ievm import TFiEVM
from tf_ievm.util import batch_generator


def fit_benchmark(x_train, y_train, x_test, eagerly=True, reduce=False, **evm_kwargs):
    tf.config.run_functions_eagerly(eagerly)
    random.seed(seed)
    np.random.seed(seed)
    random_seed.set_seed(seed)

    evm = TFiEVM(**evm_kwargs)

    start = time.time()
    evm.fit(x_train, y_train)
    _times = {"fit": np.around(time.time() - start, decimals=3)}

    if reduce:
        start = time.time()
        evm.reduce()
        _times["reduce"] = np.around(time.time() - start, decimals=3)

    start = time.time()
    evm.predict(x_test)
    _times["predict"] = np.around(time.time() - start, decimals=3)

    return _times


def partial_fit_benchmark(x_train, y_train, batch_size=128, eagerly=True, reduce=False, **evm_kwargs):
    tf.config.run_functions_eagerly(eagerly)
    random.seed(seed)
    np.random.seed(seed)
    random_seed.set_seed(seed)

    evm = TFiEVM(**evm_kwargs)

    partial_fit_times = []
    reduce_times = []
    for cx, cy in batch_generator(x_train, y_train, batch_size=batch_size):
        start = time.time()
        evm.partial_fit(cx, cy)
        partial_fit_times.append(time.time() - start)

        if reduce:
            start = time.time()
            evm.reduce()
            reduce_times.append(time.time() - start)

    _times = {"total_fit": np.around(np.sum(partial_fit_times), decimals=3),
              "partial_fit_avg": np.around(np.mean(partial_fit_times), decimals=3)}

    if reduce:
        _times["reduce_avg"] = np.around(np.mean(reduce_times), decimals=3)

    return _times


def gridsearch_benchmark(x_train, y_train, eagerly=True, **evm_kwargs):
    tf.config.run_functions_eagerly(eagerly)
    random.seed(seed)
    np.random.seed(seed)
    random_seed.set_seed(seed)

    param_grid = {"tail_size": [50, 100, 200],
                  "distance_multiplier": [0.6, 0.8, 0.9]}
    cv = GridSearchCV(TFiEVM(**evm_kwargs), param_grid=param_grid, refit=False)

    start = time.time()
    cv.fit(x_train, y_train)
    _times = {"gs-total": np.around(time.time() - start, decimals=3),
              "gs-mean-fits": np.around(np.mean(cv.cv_results_["mean_fit_time"]), decimals=3)}

    return _times


def do_benchmark(_bm, _times):
    # --------------------------------
    # single fit
    # --------------------------------
    if _bm == 0:
        name = "Fit small eagerly"
        _times[name] = fit_benchmark(_x_train_small, _y_train_small, _x_test_small, eagerly=True,
                                     tail_size=tail_size, distance_multiplier=distance_multiplier)

    elif _bm == 1:
        name = "Fit small non-eagerly"
        _times[name] = fit_benchmark(_x_train_small, _y_train_small, _x_test_small, eagerly=False,
                                     tail_size=tail_size, distance_multiplier=distance_multiplier)

    elif _bm == 2:
        name = "Fit large eagerly"
        _times[name] = fit_benchmark(_x_train, _y_train, _x_test, eagerly=True,
                                     tail_size=tail_size, distance_multiplier=distance_multiplier)

    elif _bm == 3:
        name = "Fit large non-eagerly"
        _times[name] = fit_benchmark(_x_train, _y_train, _x_test, eagerly=False,
                                     tail_size=tail_size, distance_multiplier=distance_multiplier)

    # --------------------------------
    # partial fit
    # --------------------------------
    elif _bm == 4:
        name = "Partial fit eagerly; tail_size={}; batch_size={}".format(tail_size, _batch_size)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier)
        tf.keras.backend.clear_session()
        name = "Partial fit eagerly; tail_size={}; batch_size={}".format(tail_size_small, batch_size_small)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=batch_size_small, eagerly=True,
                                             tail_size=tail_size_small, distance_multiplier=distance_multiplier)

    elif _bm == 5:
        name = "Partial fit non-eagerly; tail_size={}; batch_size={}".format(tail_size, _batch_size)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=False,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier)
        tf.keras.backend.clear_session()
        name = "Partial fit non-eagerly; tail_size={}; batch_size={}".format(tail_size_small, batch_size_small)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=batch_size_small, eagerly=False,
                                             tail_size=tail_size_small, distance_multiplier=distance_multiplier)

    elif _bm == 6:
        name = "Partial fit eagerly; tail_track_mode = 'max'; tail_size={}; batch_size={}".format(
            tail_size, _batch_size)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="max")
        tf.keras.backend.clear_session()
        name = "Partial fit eagerly; tail_track_mode = 'max'; tail_size={}; batch_size={}".format(
            tail_size_small, batch_size_small)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=batch_size_small, eagerly=True,
                                             tail_size=tail_size_small, distance_multiplier=distance_multiplier,
                                             tail_track_mode="max")

    elif _bm == 7:
        name = "Partial fit non-eagerly; tail_track_mode = 'max'; tail_size={}; batch_size={}".format(
            tail_size, _batch_size)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=False,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="max")
        tf.keras.backend.clear_session()
        name = "Partial fit non-eagerly; tail_track_mode = 'max'; tail_size={}; batch_size={}".format(
            tail_size_small, batch_size_small)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=batch_size_small, eagerly=False,
                                             tail_size=tail_size_small, distance_multiplier=distance_multiplier,
                                             tail_track_mode="max")

    elif _bm == 8:
        name = "Partial fit eagerly; tail_track_mode = 'all'; tail_size={}; batch_size={}".format(
            tail_size, _batch_size)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all")
        tf.keras.backend.clear_session()
        name = "Partial fit eagerly; tail_track_mode = 'all'; tail_size={}; batch_size={}".format(
            tail_size_small, batch_size_small)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=batch_size_small, eagerly=True,
                                             tail_size=tail_size_small, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all")

    elif _bm == 9:
        name = "Partial fit non-eagerly; tail_track_mode = 'all'; tail_size={}; batch_size={}".format(
            tail_size, _batch_size)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=False,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all")
        tf.keras.backend.clear_session()
        name = "Partial fit non-eagerly; tail_track_mode = 'all'; tail_size={}; batch_size={}".format(
            tail_size_small, batch_size_small)
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=batch_size_small, eagerly=False,
                                             tail_size=tail_size_small, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all")

    # --------------------------------
    # partial fit with fix sized model reduction
    # --------------------------------
    elif _bm == 10:
        name = "Partial fit eagerly; model_reduction = 'bisection_set_cover'"
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=True,
                                             reduce=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all", reduction_mode="bisection_set_cover",
                                             reduction_param=max_k_evs)

    elif _bm == 11:
        name = "Partial fit non-eagerly; model_reduction = 'bisection_set_cover'"
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=False,
                                             reduce=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all", reduction_mode="bisection_set_cover",
                                             reduction_param=max_k_evs)

    elif _bm == 12:
        name = "Partial fit eagerly; reduction_mode = 'weighted_set_cover'"
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=True,
                                             reduce=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all", reduction_mode="weighted_set_cover",
                                             reduction_param=max_k_evs)

    elif _bm == 13:
        name = "Partial fit non-eagerly; reduction_mode = 'weighted_set_cover'"
        _times[name] = partial_fit_benchmark(_x_train_small, _y_train_small, batch_size=_batch_size, eagerly=False,
                                             reduce=True,
                                             tail_size=tail_size, distance_multiplier=distance_multiplier,
                                             tail_track_mode="all", reduction_mode="weighted_set_cover",
                                             reduction_param=max_k_evs)

    # --------------------------------
    # gridsearch (multiple small fits)
    # --------------------------------
    elif _bm == 14:
        name = "Gridsearch eagerly"
        _times[name] = gridsearch_benchmark(_x_train_small, _y_train_small, eagerly=True)

    elif _bm == 15:
        name = "Gridsearch non-eagerly"
        _times[name] = gridsearch_benchmark(_x_train_small, _y_train_small, eagerly=False)

    else:
        raise ValueError("unknown benchmark '{}'".format(_bm))

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        default=np.arange(16, dtype=int).tolist(),
                        help="")
    args = parser.parse_args()

    tf.get_logger().setLevel(logging.ERROR)
    seed = 123
    tail_size = 100
    tail_size_small = 25
    distance_multiplier = 0.8
    max_k_evs = 50
    _batch_size = 128
    batch_size_small = 16

    # Load entire dataset
    _x, _y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    _x_train, _x_test, _y_train, _y_test = train_test_split(_x, _y, train_size=50000, random_state=seed)

    # Use only a subset for some benchmarks
    perm = np.random.permutation(len(_x_train))[:10000]
    _x_train_small = _x_train[perm]
    _y_train_small = _y_train[perm]
    perm = np.random.permutation(len(_x_test))[:10000]
    _x_test_small = _x_test[perm]

    scaler = StandardScaler()
    _x_train_small = scaler.fit_transform(_x_train_small)
    _x_test_small = scaler.transform(_x_test_small)
    del scaler, _x, _y, _y_test

    times = {}
    for bm in args.benchmarks:
        bm = int(bm)
        print("--- benchmark {} ---".format(bm))
        do_benchmark(bm, times)

        for n, ts in times.items():
            for k, v in ts.items():  # noqa
                print("{} - {} took: {} s".format(n, k, v))
            print("----")
