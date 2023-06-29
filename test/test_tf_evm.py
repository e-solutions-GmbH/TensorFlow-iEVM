import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable TensorFlow warnings

from copy import deepcopy
import logging
import random
import unittest

import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
import sklearn.metrics as skmetrics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.framework import random_seed

from tf_ievm import util as my_h
from tf_ievm.ievm import TFiEVM

# Check whether optional tests are executable
try:
    # noinspection PyUnresolvedReferences
    from evm_wrapper import EVMWrapper

    SKIP_OPT_EVM_TESTS = False
except ImportError as e:
    logging.info("will ignore optional tests: {}".format(e.msg))
    SKIP_OPT_EVM_TESTS = True


class TestEVM(unittest.TestCase):

    @staticmethod
    def __np_all_close(a, b, rtol=0., atol=1e-7):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    def setUp(self) -> None:
        self.eps = tf.keras.backend.epsilon()
        self.seed = 123
        random.seed(self.seed)
        np.random.seed(self.seed)
        random_seed.set_seed(self.seed)

        tf.get_logger().setLevel(logging.ERROR)

    def test_init_exceptions(self):
        """
        Test exception throw in __init__ call.
        """

        with self.assertRaises(ValueError):
            TFiEVM(tail_size=0)

        with self.assertRaises(ValueError):
            TFiEVM(distance_multiplier=0.0)

        with self.assertRaises(ValueError):
            TFiEVM(distance_multiplier=1.0 + self.eps)

        with self.assertRaises(NotImplementedError):
            TFiEVM(distance="foo")

        with self.assertRaises(ValueError):
            TFiEVM(reduction_mode="foo")

        with self.assertRaises(ValueError):
            TFiEVM(reduction_param=int(0))

        with self.assertRaises(ValueError):
            TFiEVM(reduction_param=0.0 - self.eps)

        with self.assertRaises(ValueError):
            TFiEVM(reduction_param=1.0 + self.eps)

        with self.assertRaises(ValueError):
            TFiEVM(reduction_tolerance=0.0)

        with self.assertRaises(ValueError):
            TFiEVM(tail_track_mode="foo")

        with self.assertRaises(TypeError):
            TFiEVM(dbscan_kwargs={"foo": 0})

        with self.assertRaises(ValueError):
            TFiEVM(max_iter=0)

        with self.assertRaises(ValueError):
            TFiEVM(tolerance=0.0)

        with self.assertRaises(ValueError):
            TFiEVM(precision=0.0)

        with self.assertRaises(ValueError):
            TFiEVM(batch_size=-1)

        with self.assertRaises(ValueError):
            TFiEVM(newton_init_mode="foo")

        with self.assertRaises(ValueError):
            TFiEVM(dtype=tf.int32)

        with self.assertRaises(ValueError):
            TFiEVM(statistics="foo")

    def test_setget_params(self):
        param_dict = {"tail_size": 5,
                      "distance_multiplier": 0.7,
                      "distance": "squared_euclidean",
                      "max_iter": 50,
                      "tolerance": 0.1,
                      "reduction_mode": "bisection_set_cover",
                      "reduction_param": 10,
                      "reduction_tolerance": 1e-3,
                      "reduce_after_fit": True,
                      "tail_track_mode": "max",
                      "dbscan_kwargs": "normal",
                      "batch_size": 512,
                      "precision": 1e-5,
                      "vectorized": False,
                      "parallel": 5,
                      "newton_init_mode": "mean",
                      "cpu_fallback": 500,
                      "dtype": "float64",
                      "verbose": 1,
                      "statistics": "all"}

        evm = TFiEVM()
        self.assertNotEqual(param_dict, evm.get_params())
        evm.set_params(**param_dict)
        self.assertDictEqual(param_dict, evm.get_params())

    def test_emptiness(self):
        evm = TFiEVM()
        self.assertTrue(evm.is_empty())

        evm.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        self.assertFalse(evm.is_empty())

    def __test_deepcopy(self):
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        evm = TFiEVM(tail_size=50, distance_multiplier=0.75)
        evm.fit(x_train, y_train)

        evm2 = deepcopy(evm)
        self.assertDictEqual(evm.get_params(), evm2.get_params())

        y_proba = evm.predict_proba(x_test)
        y_proba2 = evm2.predict_proba(x_test)
        self.assertTrue(TestEVM.__np_all_close(y_proba, y_proba2))

        y_pred = evm.predict_from_proba(y_proba)
        y_pred2 = evm2.predict_from_proba(y_proba2)
        self.assertTrue(np.all(np.equal(y_pred, y_pred2)))

    def test_deepcopy(self):
        tf.config.run_functions_eagerly(False)
        self.__test_deepcopy()

    def test_deepcopy_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_deepcopy()

    def __test_weibull_partial_fit_modes_against_fit_float32(self):
        tail_size = 10
        mini_batch_size = 25

        x, y = load_iris(return_X_y=True)
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        n_samples = len(y)
        perm = np.random.permutation(n_samples)
        x = x[perm]
        y = y[perm]

        # classic single fit
        tfevm0 = TFiEVM(tail_size=tail_size, batch_size=len(y))
        tfevm0.fit(x, y)

        for mode in ["none", "max", "all"]:
            tfevm1 = TFiEVM(tail_size=tail_size, tail_track_mode=mode)
            for i, [cx, cy] in enumerate(my_h.batch_generator(x, y, batch_size=mini_batch_size)):
                tfevm1.partial_fit(cx, cy)

            self.assertTrue(TestEVM.__np_all_close(tfevm0._ev_w_shape, tfevm1._ev_w_shape, atol=7e-4))
            self.assertTrue(TestEVM.__np_all_close(tfevm0._ev_w_scale, tfevm1._ev_w_scale, atol=6e-6))

    def test_weibull_partial_fit_modes_against_fit_float32(self):
        tf.config.run_functions_eagerly(False)
        self.__test_weibull_partial_fit_modes_against_fit_float32()

    def test_weibull_partial_fit_modes_against_fit_float32_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_weibull_partial_fit_modes_against_fit_float32()

    def __test_weibull_partial_fit_modes_against_fit_float64(self):
        tail_size = 10
        mini_batch_size = 25

        x, y = load_iris(return_X_y=True)

        n_samples = len(y)
        perm = np.random.permutation(n_samples)
        x = x[perm]
        y = y[perm]

        # classic single fit
        tfevm0 = TFiEVM(tail_size=tail_size, batch_size=len(y), dtype="float64")
        tfevm0.fit(x, y)

        for mode in ["none", "max", "all"]:
            tfevm1 = TFiEVM(tail_size=tail_size, tail_track_mode=mode,
                            dtype="float64")
            for i, [cx, cy] in enumerate(my_h.batch_generator(x, y, batch_size=mini_batch_size)):
                tfevm1.partial_fit(cx, cy)

            self.assertTrue(TestEVM.__np_all_close(tfevm0._ev_w_shape, tfevm1._ev_w_shape, atol=8e-9))
            self.assertTrue(TestEVM.__np_all_close(tfevm0._ev_w_scale, tfevm1._ev_w_scale, atol=4e-12))

    def test_weibull_partial_fit_modes_against_fit_float64(self):
        tf.config.run_functions_eagerly(False)
        self.__test_weibull_partial_fit_modes_against_fit_float64()

    def test_weibull_partial_fit_modes_against_fit_float64_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_weibull_partial_fit_modes_against_fit_float64()

    def __test_acc_datasets(self, shape_init="ones"):
        scorer = skmetrics.make_scorer(skmetrics.accuracy_score)
        for data_fn, min_acc in zip([load_digits, load_breast_cancer, load_wine, load_iris], [.97, .95, .99, .93]):
            x, y = data_fn(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            param_grid = {"tail_size": [50, 100, 250, 500], "distance_multiplier": [.25, .5, .75, 1.]}
            gs = GridSearchCV(TFiEVM(newton_init_mode=shape_init),
                              param_grid=param_grid, scoring=scorer, n_jobs=1)
            gs.fit(x_train, y_train)

            evm = gs.best_estimator_
            y_pred = evm.predict(x_test)
            acc = skmetrics.accuracy_score(y_test, y_pred)
            self.assertGreater(acc, min_acc)

    def test_acc_datasets(self):
        tf.config.run_functions_eagerly(False)
        self.__test_acc_datasets()

    def test_acc_datasets_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_acc_datasets()

    def test_acc_datasets_mean_shape(self):
        tf.config.run_functions_eagerly(False)
        self.__test_acc_datasets(shape_init="mean")

    def test_acc_datasets_mean_shape_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_acc_datasets(shape_init="mean")

    def __test_cevm_acc_datasets(self):
        scorer = skmetrics.make_scorer(skmetrics.accuracy_score)
        for data_fn, min_acc in zip([load_digits, load_breast_cancer], [.97, .96]):
            x, y = data_fn(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            param_grid = {"tail_size": [50, 100, 250, 500], "distance_multiplier": [.25, .5, .75, 1.]}
            gs = GridSearchCV(TFiEVM(tail_size=1, dbscan_kwargs="normal"), cv=KFold(3),
                              param_grid=param_grid, scoring=scorer, n_jobs=1)
            gs.fit(x_train, y_train)

            evm = gs.best_estimator_
            y_pred = evm.predict(x_test)
            acc = skmetrics.accuracy_score(y_test, y_pred)
            self.assertGreater(acc, min_acc)

    def test_cevm_acc_datasets(self):
        tf.config.run_functions_eagerly(False)
        self.__test_cevm_acc_datasets()

    def test_cevm_acc_datasets_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_cevm_acc_datasets()

    def __test_bisection_set_cover_reduction(self):
        """
        Test model reduction by comparing to original implementation.
        """

        tail_size = 10
        distance_multiplier = 0.7
        x, y = load_iris(return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        for max_evs in [1, 5, 25, 100]:
            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier,
                           reduction_mode="bisection_set_cover", reduction_param=max_evs, reduce_after_fit=True)
            tfevm.fit(x_train, y_train)
            _, cnts = np.unique(tfevm._ev_y, return_counts=True)

            self.assertTrue(np.all(np.less_equal(cnts, max_evs)))

        tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier,
                       reduction_mode="bisection_set_cover", reduction_param=500)
        tfevm.fit(x_train, y_train)
        for max_evs in [100, 25, 5, 1]:
            tfevm.reduction_param = max_evs
            tfevm.reduce()
            _, cnts = np.unique(tfevm._ev_y, return_counts=True)
            self.assertTrue(np.all(np.less_equal(cnts, max_evs)))

    def test_bisection_set_cover_reduction(self):
        tf.config.run_functions_eagerly(False)
        self.__test_bisection_set_cover_reduction()

    def test_bisection_set_cover_reduction_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_bisection_set_cover_reduction()

    def __test_weighted_set_cover_reduction(self):
        tail_size = 10
        distance_multiplier = 0.7
        x, y = load_iris(return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        for max_evs in [1, 5, 25, 100]:
            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier,
                           reduction_mode="weighted_set_cover", reduction_param=max_evs, reduce_after_fit=False)
            tfevm.fit(x_train, y_train)
            tfevm.reduce()
            _, cnts = np.unique(tfevm._ev_y, return_counts=True)

            self.assertTrue(np.all(np.less_equal(cnts, max_evs)))

        tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier,
                       reduction_mode="weighted_set_cover",
                       reduction_param=500)
        tfevm.fit(x_train, y_train)
        for max_evs in [100, 25, 5, 1]:
            tfevm.reduction_param = max_evs
            tfevm.reduce()
            _, cnts = np.unique(tfevm._ev_y, return_counts=True)
            self.assertTrue(np.all(np.less_equal(cnts, max_evs)))

    def test_weighted_set_cover_reduction(self):
        tf.config.run_functions_eagerly(False)
        self.__test_weighted_set_cover_reduction()

    def test_weighted_set_cover_reduction_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_weighted_set_cover_reduction()

    # --------------------------------------
    # ----------- OPTIONAL TESTS -----------
    # --------------------------------------

    def __test_opt_reference_weibull(self):
        """
        Test Weibull results by comparing to original implementation over several distance multipliers.
        """

        tail_size = 10
        x, y = load_iris(return_X_y=True)
        x = x[y != 1]
        y = y[y != 1]

        for dm, (prec_shape, prec_scale) in zip([0.25, 0.5, 0.75, 1.],
                                                [(2e-2, 4e-6), (7e-3, 7e-6), (6e-3, 1e-5), (5e-3, 2e-5)]):
            evm = EVMWrapper(tail_size=tail_size, distance_multiplier=dm)
            evm.fit(x, y)
            w_shapes_evm = []
            w_scales_evm = []
            for i in range(len(evm.evm.evms)):
                for mw in evm.evm.evms[i].margin_weibulls:
                    params = mw.get_params()
                    w_shapes_evm.append(params[1])
                    w_scales_evm.append(params[0])
            w_shapes_evm = np.array(w_shapes_evm)
            w_scales_evm = np.array(w_scales_evm)

            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=dm)
            tfevm.fit(x, y)

            self.assertTrue(TestEVM.__np_all_close(w_shapes_evm, tfevm._ev_w_shape, atol=prec_shape))
            self.assertTrue(TestEVM.__np_all_close(w_scales_evm, tfevm._ev_w_scale, atol=prec_scale))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_weibull()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_weibull()

    def __test_opt_reference_probas(self):
        """
        Test estimated probabilities by comparing to original implementation over several distance multipliers.
        """

        tail_size = 10
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        for dm, prec in zip([0.25, 0.5, 0.75, 1.], [2e-4, 4e-4, 5e-5, 5e-5]):
            evm = EVMWrapper(tail_size=tail_size, distance_multiplier=dm)
            evm.fit(x_train, y_train)
            evm_probas = evm.predict_proba(x_test)

            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=dm)
            tfevm.fit(x_train, y_train)
            tfevm_probas = tfevm.predict_proba(x_test)

            self.assertTrue(TestEVM.__np_all_close(evm_probas, tfevm_probas, atol=prec))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_probas(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_probas()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_probas_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_probas()

    def __test_opt_reference_predict(self):
        """
        Test predictions by comparing to original implementation over several distance multipliers.
        """

        tail_size = 10
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        for dm in [0.25, 0.5, 0.75, 1.]:
            evm = EVMWrapper(tail_size=tail_size, distance_multiplier=dm)
            evm.fit(x_train, y_train)
            evm_pred = evm.predict(x_test)

            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=dm)
            tfevm.fit(x_train, y_train)
            tfevm_pred = tfevm.predict(x_test)

            self.assertTrue(np.all(np.equal(evm_pred, tfevm_pred)))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_predict(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_predict()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_predict_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_predict()

    def __test_opt_reference_predict_probas(self):
        tail_size = 10
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        for dm in [0.25, 0.5, 0.75, 1.]:
            evm = EVMWrapper(tail_size=tail_size, distance_multiplier=dm)
            evm.fit(x_train, y_train)
            evm_pred = evm.predict(x_test)

            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=dm)
            tfevm.fit(x_train, y_train)
            tfevm_pred = tfevm.predict(x_test)
            tfevm_probas = tfevm.predict_proba(x_test)
            tfevm_probas_pred = np.argmax(tfevm_probas, axis=1)
            tfevm_probas_pred = tfevm._label_encoder.inverse_transform(tfevm_probas_pred)

            self.assertTrue(np.all(np.equal(evm_pred, tfevm_pred)))
            self.assertTrue(np.all(np.equal(tfevm_pred, tfevm_probas_pred)))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_predict_probas(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_predict_probas()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_predict_probas_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_predict_probas()

    def __test_opt_reference_weibull_dtype(self):
        tail_size = 10
        x, y = load_iris(return_X_y=True)
        evm = EVMWrapper(tail_size=tail_size)
        evm.fit(x, y)
        w_shapes_evm = []
        w_scales_evm = []
        for i in range(len(evm.evm.evms)):
            for mw in evm.evm.evms[i].margin_weibulls:
                params = mw.get_params()
                w_shapes_evm.append(params[1])
                w_scales_evm.append(params[0])
        w_shapes_evm = np.array(w_shapes_evm)
        w_scales_evm = np.array(w_scales_evm)

        for dtype, (prec_shape, prec_scale) in zip(["float32", "float64"], [(8e-3, 7e-6), (7e-5, 1e-7)]):
            tfevm = TFiEVM(tail_size=tail_size, dtype=dtype)
            tfevm.fit(x, y)

            self.assertTrue(TestEVM.__np_all_close(w_shapes_evm, tfevm._ev_w_shape, atol=prec_shape))
            self.assertTrue(TestEVM.__np_all_close(w_scales_evm, tfevm._ev_w_scale, atol=prec_scale))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_weibull_opt_reference_dtype(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_weibull_dtype()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_dtype_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_weibull_dtype()

    def __test_opt_reference_probas_dtype(self):
        tail_size = 10
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        evm = EVMWrapper(tail_size=tail_size)
        evm.fit(x_train, y_train)
        evm_probas2 = evm.predict_proba(x_test)

        for dtype, prec in zip(["float32", "float64"], [3e-4, 8e-7]):
            tfevm = TFiEVM(tail_size=tail_size, dtype=dtype)
            tfevm.fit(x_train, y_train)
            tfevm_probas = tfevm.predict_proba(x_test)
            self.assertTrue(TestEVM.__np_all_close(evm_probas2, tfevm_probas, atol=prec))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_probas_dtype(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_probas_dtype()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_probas_dtype_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_probas_dtype()

    def __test_opt_reference_weibull_parallel(self):
        tail_size = 10
        x, y = load_iris(return_X_y=True)

        evm = EVMWrapper(tail_size=tail_size)
        evm.fit(x, y)
        w_shapes_evm = []
        w_scales_evm = []
        for i in range(len(evm.evm.evms)):
            for mw in evm.evm.evms[i].margin_weibulls:
                params = mw.get_params()
                w_shapes_evm.append(params[1])
                w_scales_evm.append(params[0])
        w_shapes_evm = np.array(w_shapes_evm)
        w_scales_evm = np.array(w_scales_evm)

        prec_shape_scale = [(1e-2, 7e-6), (1e-2, 7e-6), (7e-5, 2e-7), (7e-5, 1e-7)]
        i = 0
        for dtype in ["float32", "float64"]:
            # for vectorized in [True, False]:
            for vectorized in [False]:
                tfevm = TFiEVM(tail_size=tail_size, vectorized=vectorized, dtype=dtype)
                tfevm.fit(x, y)

                self.assertTrue(TestEVM.__np_all_close(w_shapes_evm, tfevm._ev_w_shape, atol=prec_shape_scale[i][0]))
                self.assertTrue(TestEVM.__np_all_close(w_scales_evm, tfevm._ev_w_scale, atol=prec_shape_scale[i][1]))
                i += 1

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_parallel(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_weibull_parallel()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_parallel_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_weibull_parallel()

    def __test_opt_reference_weibull_batchwise(self):
        tail_size = 10
        distance_multiplier = 0.5

        x, y = load_iris(return_X_y=True)

        evm = EVMWrapper(tail_size=tail_size, distance_multiplier=distance_multiplier)
        evm.fit(x, y)
        w_shapes_evm = []
        w_scales_evm = []
        for i in range(len(evm.evm.evms)):
            for mw in evm.evm.evms[i].margin_weibulls:
                params = mw.get_params()
                w_shapes_evm.append(params[1])
                w_scales_evm.append(params[0])
        w_shapes_evm = np.array(w_shapes_evm)
        w_scales_evm = np.array(w_scales_evm)

        for batch_size in [len(y), 10, 50, 5000]:
            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier, batch_size=batch_size)
            tfevm.fit(x, y)

            self.assertTrue(TestEVM.__np_all_close(w_shapes_evm, tfevm._ev_w_shape, atol=8e-3))
            self.assertTrue(TestEVM.__np_all_close(w_scales_evm, tfevm._ev_w_scale, atol=7e-6))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_batchwise_opt_reference(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_weibull_batchwise()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_batchwise_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_weibull_batchwise()

    def __test_opt_reference_weibull_batchwise_tf(self):
        tail_size = 105
        distance_multiplier = 0.5

        x, y = load_iris(return_X_y=True)

        evm = EVMWrapper(tail_size=tail_size, distance_multiplier=distance_multiplier)
        evm.fit(x, y)
        w_shapes_evm = []
        w_scales_evm = []
        for i in range(len(evm.evm.evms)):
            for mw in evm.evm.evms[i].margin_weibulls:
                params = mw.get_params()
                w_shapes_evm.append(params[1])
                w_scales_evm.append(params[0])
        w_shapes_evm = np.array(w_shapes_evm)
        w_scales_evm = np.array(w_scales_evm)

        for batch_size in [len(y), 10, 50, 5000]:
            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier, batch_size=batch_size)
            tfevm.fit(x, y)

            self.assertTrue(TestEVM.__np_all_close(w_shapes_evm, tfevm._ev_w_shape, atol=7e-5))
            self.assertTrue(TestEVM.__np_all_close(w_scales_evm, tfevm._ev_w_scale, atol=3e-6))

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_batchwise_tf(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_weibull_batchwise_tf()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_weibull_batchwise_tf_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_weibull_batchwise_tf()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_reduction_set_cover(self):
        tf.config.run_functions_eagerly(False)
        self.__test_opt_reference_reduction_set_cover()

    @unittest.skipIf(SKIP_OPT_EVM_TESTS, "optional evm package not available")
    def test_opt_reference_reduction_set_cover_eagerly(self):
        tf.config.run_functions_eagerly(True)
        self.__test_opt_reference_reduction_set_cover()

    def __test_opt_reference_reduction_set_cover(self):
        """
        Test model reduction by comparing to original implementation.
        """

        tail_size = 10
        distance_multiplier = 0.7
        x, y = load_iris(return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=self.seed)

        for cover_threshold in [0.5, 0.7, 0.9, 0.95]:
            evm = EVMWrapper(tail_size=tail_size, distance_multiplier=distance_multiplier,
                             cover_threshold=cover_threshold)
            evm.fit(x_train, y_train)
            evm_ev_x, _ = evm.extreme_vectors()

            tfevm = TFiEVM(tail_size=tail_size, distance_multiplier=distance_multiplier, reduction_mode="set_cover",
                           reduction_param=cover_threshold, reduce_after_fit=True)
            tfevm.fit(x_train, y_train)

            self.assertTrue(TestEVM.__np_all_close(evm_ev_x, tfevm._ev_x, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
