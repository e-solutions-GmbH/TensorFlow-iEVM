import copy
import logging
import math
from typing import Tuple, Union, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import DBSCAN
from sklearn.exceptions import NotFittedError
from tensorflow import keras
import tensorflow as tf

import tf_ievm.custom_tf_util as ctf_util
import tf_ievm.preprocessing as le
import tf_ievm.weibull as wb


class TFiEVM(BaseEstimator, ClassifierMixin):  # noqa

    def __init__(self, tail_size: int = 50, distance_multiplier: float = 0.5, distance: str = "euclidean",
                 reduction_mode: str = "set_cover", reduction_param: Union[float, int] = 1.0, reduction_tolerance=1e-6,
                 reduce_after_fit: bool = False, tail_track_mode: str = "none",
                 dbscan_kwargs: Union[str, dict] = None, max_iter=100, tolerance=1e-6,
                 precision=keras.backend.epsilon(), batch_size: int = 2048, vectorized=True, parallel=None,
                 newton_init_mode="ones", cpu_fallback: int = 0, dtype="float32", verbose=0,
                 statistics: Union[str, List[str]] = None):
        """
        Parameters
        ----------
        tail_size : int, default is 50. Has to be > 0. The tail size defines the number of negative samples used per
                    anchor sample to estimate the Weibull distribution.
        distance_multiplier : float, default is 0.5. Has to be in range (0, 1].
                              The (tail) distances are multiplied by this factor. With a constant tail size, an
                              increase in this factor leads to Weibull distributions with greater variance.
        distance : str, default is "euclidean". The distance metric between samples.
                   Choices = ['euclidean', 'squared_euclidean'].
        reduction_mode :    str, default is "set_cover". The model reduction mode.
                            Choices = ['set_cover', 'bisection_set_cover', 'weighted_set_cover'].
                            'set_cover':    The vanilla set cover model reduction requires a floating point value for
                                            <reduction_param> that defines the coverage threshold.
                            'bisection_set_cover':  Performs multiple set cover iterations (bisection) to find a
                                                    suitable coverage threshold. Requires an int value for
                                                    <reduction_param> that defines the maximum number of extreme
                                                    vectors allowed per class.
                            'weighted_set_cover':   Performs the weighted set cover model reduction. Requires an int
                                                    value for <reduction_param> that defines the maximum number of
                                                    extreme vectors allowed per class.
        reduction_param :   Union[float, int], default is 1.0.
                            The parameter required for the chosen model reduction mode.
                            float & 'set_cover':    The coverage threshold, in range [0, 1].
                            int & '*_set_cover':    The maximum number of extreme vectors allowed per class, greater 0.
        reduction_tolerance :   float, default is 1e-6. The convergence tolerance for the bisection set cover model
                                reduction.
        reduce_after_fit :  bool, default is False.
                            Whether a model reduction should be carried out after each (partial-)fit, otherwise the
                            reduction must be called explicitly.
        tail_track_mode : str, default is 'none'.
                          This parameter influences the update routine of the extreme vectors (EVs) during partial fit.
                          Without model reduction, all modes are the same in the resulting model representation, but the
                          training time is different.

                          Choices are ['none', 'max', 'all'].
                          Memory consumption (less -> more): 'none', 'max', 'all'.
                          Incremental computational complexity (less -> more): 'all', 'max', 'none'.

                          'none':   Do not track tail distances. The EVs are updated during partial fit using the
                                    currently available data, i.e. the EVs and the data given during partial fit.
                                    Without model reduction, this is equivalent to cyclic retraining of the EVM.

                          'max':    Track the maximum tail distance. The EVs are updated when one of the new samples is
                                    within the range of the maximum tail distance. The new tail is calculated based on
                                    the currently available data, i.e. the EVs and the data specified in partial fit.
                                    During model reduction, the tail distances of the EVs to removed samples are lost,
                                    which can lead to different Weibull distributions.

                          'all':    For each EV, the entire tail is stored. EVs are updated when the tail changes, i.e.
                                    when one of the new samples is in the range of the maximum tail distance. Old tail
                                    distances that are still smaller than the new maximum tail distance are reused.
                                    This differs from the 'max' mode as the tail distances from EVs to removed samples
                                    are not lost. In the case of large tail sizes and no model reduction, this can lead
                                    to a high memory footprint.
        dbscan_kwargs : Union[str, dict], default is None, i.e. no prior clustering.
                        If given, class-wise clustering is performed before each (partial-)fit with new samples and
                        EVs.

                        If dbscan_kwargs == "normal", DBSCAN will be initialized with proposed arguments:
                        {'eps': 0.3, 'min_samples': 1, 'n_jobs': -1}.
                        The dict needs to contain the initialization arguments for sklearn.cluster.DBSCAN.

                        Reference: Henrydoss, James, et al.
                        "Enhancing Open-Set Recognition using Clustering-based Extreme Value Machine (C-EVM)."
                        2020 IEEE International Conference on Big Data (Big Data). IEEE, 2020.

                        Additional notes regarding partial fit and the 'tail_track_mode':
                        'none':             Clustering is done with the new data and the EVs. Note that the EVs
                                            are the centroids from previous fits.
                        'max' and 'all':    Clustering is only carried out with the new data.
                                            Reason: Including the EV centroids in the clustering would likely lead
                                            to minor changes in the centroids themselves. This in turn would
                                            invalidate the tails tracked, lead to new Weibull estimates and negate the
                                            very purpose of tail tracking.
        max_iter : int, default is 100. Has to be greater 0.
                   The maximum number of Newton iterations to fit a Weibull distribution.
        tolerance : float, default is 1e-6. Has to be >= keras.backend.epsilon().
                    Parameter tolerance to check for convergence of the shape parameter during Weibull estimation.
        precision : float, default is keras.backend.epsilon(), which is 1e-7. Has to be >= than that.
                    Stop the Newton iterations in case the derivative of the objective w.r.t. the shape parameter
                    is <= 'precision'.
        batch_size : int, default is 2048.
                     A value > 0 trains the EVM in batches to avoid large matrices and memory explosion.
                     If batch_size is 0, the entire batch is trained at once.
                     Depending on the EVM arguments, training in batches is not the same as partial fit.
        vectorized : bool, default is True.
                     If True, Weibull distributions are estimated fully vectorized. This is usually the faster option.
                     If False, Weibull distributions are estimated in parallel (sample-wise) via tf.map_fn.
                     The number of parallel calculations can be set via the 'parallel' parameter.
        parallel : int, default is None, i.e. 1 in eager execution and 10 in graph execution.
                   This is the parallel parameter for the tf.map_fn function and determines the number of Weibull
                   estimates that will be run in parallel. This parameter is ignored if 'vectorized' is True.
        newton_init_mode : str, default is 'ones'. Choices = ['ones', 'mean'].
                           'ones':  Will initialize the Weibull shape parameter with 1.
                           'mean':  Will initialize the Weibull shape parameter with 1 and in following estimations
                                    with the running average from previous estimates. Note that the running average is
                                    only computed during/within a fit and not across multiple fits (e.g. across
                                    several following partial_fits).
        cpu_fallback : int, default is 0, i.e. CPU fallback is deactivated.
                       If greater 0, then some computations might be moved to CPU to prevent GPU OOM-Error.
                       Some computations will check whether involved tensors exceed the size of 'cpu_fallback'. If this
                       is the case then the computation will be moved to CPU.
                       To have a reference number: for a GPU with 16 GB, we use int(4e8).
        dtype : str, default is "float32", choices = ["float32", "float64"].
                The floating point type.
        verbose : int, default is 0. Verbosity level.
                  > 0: log some general information.
                  > 1: log statistics.
        statistics : Union[str, List[str]], default is None. A list of statistics that can be tracked.
                     If statistics == "all", all of the below will be added.
                     'update_ratio': Tracks the percentage of updated EVs after every partial_fit.
                     'cluster_diff': Tracks the percentage of removed samples after DBSCAN.
        """
        # parameters
        self.tail_size = tail_size
        self.distance_multiplier = distance_multiplier
        self.distance = distance
        self.reduction_mode = reduction_mode
        self.reduction_param = reduction_param
        self.reduction_tolerance = reduction_tolerance
        self.reduce_after_fit = reduce_after_fit
        self.tail_track_mode = tail_track_mode
        self.dbscan_kwargs = dbscan_kwargs
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.precision = precision
        self.batch_size = batch_size
        self.vectorized = vectorized
        self.parallel = parallel
        self.newton_init_mode = newton_init_mode
        self.cpu_fallback = cpu_fallback
        self.dtype = dtype
        self.verbose = verbose

        # model
        self._label_encoder = le.IncrementalLabelEncoder()
        self._ev_x = None
        self._ev_y = None
        self._ev_tail = None
        self._ev_w_shape = None
        self._ev_w_scale = None
        self._ev_intercepts = None
        self._dbscan = None

        self._statistic_dir = {}
        self.statistics = statistics

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Resets the current EVM model and fits a new one.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features).
            The data to fit.
        y : np.ndarray, shape (n_samples,).
            The labels to fit.
        """
        self.reset()
        self.__check_fit_samples(x, y)

        # Transform labels
        y = self._label_encoder.fit_transform(y)

        # Clustering (C-EVM)
        if self._dbscan is not None:
            x, y = self.__cluster(x, y)

        # Fit
        if x.shape[0] > 0:
            self.__fit_anchors(anchors_x=x, anchors_y=y)

        if self._verbose > 1:
            self.print_statistics()

        # Model reduction
        if self._reduce_after_fit:
            self.reduce()

    def partial_fit(self, x: np.ndarray, y: np.ndarray):
        """
        Incremental update of the current EVM model.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features).
            The data to fit.
        y : np.ndarray, shape (n_samples,).
            The labels to fit.
        """

        # If empty, do normal fit.
        if self.is_empty():
            self.fit(x, y)
            return

        self.__check_fit_samples(x, y)

        if x.shape[1] != self._ev_x.shape[1]:
            raise ValueError(
                "{} requires consistent features, the saved extreme vectors have {}, but the current "
                "batch has {}".format(self.__class__.__name__, self._ev_x.shape[1], x.shape[1]))

        # If not empty but the tail was not tracked for the EVs, do normal fit with EVs and new data.
        if self._tail_track_mode == "none":
            self.fit(np.concatenate([self._ev_x, x], axis=0),
                     np.concatenate([self._label_encoder.inverse_transform(self._ev_y), y], axis=0))
            return

        # The label encoder can change the internal class order during a partial fit, which invalidates the stored
        # extreme vector labels, so we have to invert them and transform them back later again.
        self._ev_y = self._label_encoder.inverse_transform(self._ev_y)
        y = self._label_encoder.partial_fit_transform(y)
        self._ev_y = self._label_encoder.transform(self._ev_y)

        # Clustering (C-EVM)
        if self._dbscan is not None:
            x, y, = self.__cluster(x, y)

        # Update EVs.
        if self._verbose > 0:
            self._logger.info("updating EVs...")

        self.__fit_anchors(anchors_x=self._ev_x, anchors_y=self._ev_y, x_all=x, y_all=y, is_ev_update=True)

        # Fit new samples on themselves and EVs.
        if self._verbose > 0:
            self._logger.info("fitting new samples...")

        # Fit new samples
        if x.shape[0] > 0:
            self.__fit_anchors(anchors_x=x, anchors_y=y,
                               x_all=np.concatenate([x, self._ev_x], axis=0),
                               y_all=np.concatenate([y, self._ev_y], axis=0))

        if self._verbose > 1:
            self.print_statistics()

        # Model reduction
        if self._reduce_after_fit:
            self.reduce()

    def reduce(self):
        """
        Do model reduction.
        """
        if self._reduction_mode == "set_cover" and not isinstance(self._reduction_param, float):
            raise RuntimeError(
                "{} requires that the model reduction mode '{}' has a reduction_param of type float in range "
                "of (0, 1], but type is '{}' and value is '{}'.".format(
                    self.__class__.__name__, self._reduction_mode, type(self._reduction_param),
                    self._reduction_param))

        if self._reduction_mode in ["bisection_set_cover", "weighted_set_cover"] \
                and not isinstance(self._reduction_param, int):
            raise RuntimeError(
                "{} requires that the model reduction mode '{}' has a reduction_param of type int greater 0, "
                " but type is '{}' and value is '{}'.".format(
                    self.__class__.__name__, self._reduction_mode, type(self._reduction_param),
                    self._reduction_param))

        if self.is_empty():
            return

        if self._verbose > 0:
            self._logger.info("performing model reduction...")

        all_classes = self._label_encoder.transform(self._label_encoder.classes_)

        idxs_to_keep = self._reduce(
            x_all=tf.convert_to_tensor(self._ev_x, dtype=self._tf_dtype),
            y_all=tf.convert_to_tensor(self._ev_y, dtype=tf.int32),
            w_scales_all=tf.convert_to_tensor(self._ev_w_scale[:, np.newaxis], dtype=self._tf_dtype),
            w_shapes_all=tf.convert_to_tensor(self._ev_w_shape[:, np.newaxis], dtype=self._tf_dtype),
            intercepts_all=tf.convert_to_tensor(self._ev_intercepts[:, np.newaxis], dtype=self._tf_dtype),
            classes=tf.convert_to_tensor(all_classes, dtype=tf.int32),
            reduction_param=tf.convert_to_tensor(
                self._reduction_param, dtype=self._tf_dtype if isinstance(self._reduction_param, float) else tf.int32))

        idxs_to_keep = idxs_to_keep.numpy()
        self._ev_x = self._ev_x[idxs_to_keep]
        self._ev_y = self._ev_y[idxs_to_keep]
        self._ev_w_scale = self._ev_w_scale[idxs_to_keep]
        self._ev_w_shape = self._ev_w_shape[idxs_to_keep]
        self._ev_intercepts = self._ev_intercepts[idxs_to_keep]

        if self._tail_track_mode != "none":
            self._ev_tail = self._ev_tail[idxs_to_keep]

    def predict(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features).

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,).
        """
        self.__check_predict_samples(x)

        n_samples = len(x)
        batch_size = n_samples
        if self._batch_size > 0:
            batch_size = self._batch_size

        cpu_fallback = self.__check_cpu_fallback(self._ev_x)

        idxs = self._batch_wise_predict(tf.convert_to_tensor(x, self._tf_dtype),
                                        tf.convert_to_tensor(batch_size, tf.int32),
                                        tf.convert_to_tensor(self._ev_x, self._tf_dtype),
                                        tf.convert_to_tensor(self._ev_intercepts, self._tf_dtype),
                                        tf.convert_to_tensor(self._ev_w_scale, self._tf_dtype),
                                        tf.convert_to_tensor(self._ev_w_shape, self._tf_dtype),
                                        cpu_fallback)
        return self._label_encoder.inverse_transform(self._ev_y[idxs.numpy()])

    def predict_proba(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features).

        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, n_classes).
                        Column corresponds to the class order in the internal label encoder self._enc.
        """
        self.__check_predict_samples(x)

        n_samples = len(x)
        batch_size = n_samples
        if self._batch_size > 0:
            batch_size = self._batch_size

        all_classes = self._label_encoder.transform(self._label_encoder.classes_)

        cpu_fallback = self.__check_cpu_fallback(self._ev_x)

        y_pred_t = self._batch_wise_predict_proba(
            tf.convert_to_tensor(x, self._tf_dtype), tf.convert_to_tensor(batch_size, tf.int32),
            tf.convert_to_tensor(self._ev_x, dtype=self._tf_dtype),
            tf.convert_to_tensor(self._ev_y, dtype=tf.int32),
            tf.convert_to_tensor(self._ev_intercepts, dtype=self._tf_dtype),
            tf.convert_to_tensor(self._ev_w_scale, dtype=self._tf_dtype),
            tf.convert_to_tensor(self._ev_w_shape, dtype=self._tf_dtype),
            tf.convert_to_tensor(all_classes, dtype=tf.int32),
            cpu_fallback)
        return y_pred_t.numpy()

    def predict_from_proba(self, proba: np.ndarray):
        """
        Takes the output of predict_proba to make predictions.

        Note that this may lead to different predictions than the direct predict function but only where all
        probabilities are 0.

        Parameters
        ----------
        proba

        Returns
        -------

        """
        argmax = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(argmax)

    def reset(self):
        """
        Reset the entire model.
        """
        self._ev_x = None
        self._ev_y = None
        self._ev_tail = None
        self._ev_w_scale = None
        self._ev_w_shape = None
        self._label_encoder.reset()
        self.__reset_dbscan()

        if self._statistic_dir is not None:
            for key in self._statistic_dir.keys():
                self._statistic_dir[key] = []

    def is_empty(self) -> bool:
        """
        Check whether the model is trained or empty.

        Returns
        -------
        is_empty : bool, returns True if the model is empty, otherwise False.
        """
        return self._ev_x is None or len(self._ev_x) == 0

    def n_classes(self) -> int:
        """
        Get the number of classes.

        Returns
        -------
        n_classes : int, the number of classes.
        """
        return len(self._label_encoder.classes_)

    def n_evs(self) -> int:
        """
        Get the number of extreme vectors.

        Returns
        -------
        n_evs : int, the number of extreme vectors.
        """
        if self.is_empty():
            return 0
        return len(self._ev_x)

    def print_statistics(self):
        """
        Log all tracked statistics.
        """
        if len(self._statistic_dir) > 0:
            empty = True
            to_print = "EVM statistics: "
            for key, val in self._statistic_dir.items():
                if len(val) > 0:
                    empty = False
                    to_print += "{}: {} over {} fits".format(key, np.around(np.mean(val), decimals=3), len(val))
            if not empty:
                self._logger.info(to_print)

    def __fit_anchors(self, anchors_x: np.ndarray, anchors_y: np.ndarray, x_all: np.ndarray = None,
                      y_all: np.ndarray = None, is_ev_update: bool = False):
        n_samples = len(anchors_y)
        batch_size = n_samples
        if self._batch_size > 0:
            batch_size = self._batch_size

        # This is a full tf-function that will compute the tail and estimate Weibull
        # distributions for the given anchors.
        shapes, scales, requires_update, tail_array, intercepts = self._batch_wise_fit(
            anchors_x=tf.convert_to_tensor(anchors_x, dtype=self._tf_dtype),
            anchors_y=tf.convert_to_tensor(anchors_y, dtype=tf.int32),
            batch_size=tf.convert_to_tensor(batch_size, tf.int32),
            distance_multiplier=tf.convert_to_tensor(self.distance_multiplier, self._tf_dtype),
            x_all=None if x_all is None else tf.convert_to_tensor(x_all, dtype=self._tf_dtype),
            y_all=None if y_all is None else tf.convert_to_tensor(y_all, dtype=tf.int32),
            is_ev_update=is_ev_update,
            ev_tail_t=tf.convert_to_tensor(self._ev_tail, self._tf_dtype) if is_ev_update else None)

        tail = None
        if self.tail_track_mode != "none":
            tail = ctf_util.convert_tf_tensor_array_2_numpy(tail_array)

        _requires_update = None
        if is_ev_update:
            _requires_update = ctf_util.convert_tf_tensor_array_2_numpy(requires_update)
            if _requires_update is not None and "update_ratio" in self._statistic_dir.keys():
                self._statistic_dir["update_ratio"].append(len(_requires_update) / float(len(self._ev_x)) * 100.)

        self.__update_model(shapes.numpy(), scales.numpy(), tail,
                            anchors_x=None if is_ev_update else anchors_x,
                            anchors_y=None if is_ev_update else anchors_y,
                            update_idxs=_requires_update, intercepts=intercepts.numpy())

    def __cluster(self, x, y):
        """
        Given samples are clustered class-wise. For each cluster a centroid (mean) will be generated.

        This is what the C-EVM does.
        Note that the DBSCAN object is reset directly at the end.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features).
            The data to cluster.
        y : np.ndarray, shape (n_samples,).
            The labels to cluster.

        Returns
        -------
        center_x : np.ndarray, shape (m_samples, n_features).
                   The generated centroids.
        center_y : np.ndarray, shape (m_samples,).
                   The labels of the centroids.
        """
        unique_classes = np.unique(y)

        center_x = []
        center_y = []

        # todo: parallelize this sometime
        # Perform clustering on every class independently.
        for class_label in unique_classes:
            cls_x = x[y == class_label]
            self._dbscan.fit(cls_x)

            cluster_labels = self._dbscan.labels_
            unique_clusters = np.unique(cluster_labels)

            # Remove "noisy" samples.
            unique_clusters = unique_clusters[unique_clusters != -1]

            n_clusters = len(unique_clusters)
            n_class_samples = len(cls_x)

            # If the amount of clusters is equal to the amount of class-samples
            # then the centroids are equal to the real samples.
            if n_clusters == n_class_samples:
                center_x.append(cls_x)
                center_y.append(np.full((n_class_samples,), class_label))
            else:
                centroids = np.empty((n_clusters, cls_x.shape[1]))
                for i, cluster_label in enumerate(unique_clusters):
                    cluster_x = self._dbscan.components_[cluster_labels == cluster_label]
                    centroids[i] = np.mean(cluster_x, axis=0, keepdims=True)

                center_x.append(centroids)
                center_y.append(np.full((len(centroids)), class_label))

        # We reset DBSCAN here as it stores samples in memory which we do not need.
        self.__reset_dbscan()

        center_x = np.concatenate(center_x, axis=0)
        center_y = np.concatenate(center_y, axis=0)

        if "cluster_diff" in self._statistic_dir.keys():
            self._statistic_dir["cluster_diff"].append((1. - (len(center_y) / float(len(y)))) * 100.)

        return center_x, center_y

    def __update_model(self, shapes, scales, tail, anchors_x=None, anchors_y=None, update_idxs=None, intercepts=None):
        # EVM is empty, adopt all anchors.
        if self._ev_x is None:
            self._ev_x = anchors_x
            self._ev_y = anchors_y
            self._ev_w_shape = shapes
            self._ev_w_scale = scales
            self._ev_intercepts = intercepts
            if self._tail_track_mode == "max":
                self._ev_tail = tail[:, -1][:, np.newaxis]
            elif self._tail_track_mode == "all":
                self._ev_tail = tail

        # Anchors are former EVs, they do not need to be adopted but their shapes/scales/tail changed.
        elif update_idxs is not None:
            self._ev_w_shape[update_idxs] = shapes[update_idxs]
            self._ev_w_scale[update_idxs] = scales[update_idxs]
            self._ev_intercepts[update_idxs] = intercepts[update_idxs]
            if self._tail_track_mode == "max":
                self._ev_tail[update_idxs] = tail[:, -1][:, np.newaxis]
            elif self._tail_track_mode == "all":
                new_tail = tail
                if new_tail.shape[1] != self._ev_tail.shape[1]:
                    self._ev_tail = np.resize(self._ev_tail, (self._ev_tail.shape[0], new_tail.shape[1]))
                self._ev_tail[update_idxs] = new_tail

        # EVM is not empty and new EVs arrived which need to be concatenated.
        else:
            self._ev_x = np.concatenate([self._ev_x, anchors_x], axis=0)
            self._ev_y = np.concatenate([self._ev_y, anchors_y], axis=0)
            self._ev_w_shape = np.concatenate([self._ev_w_shape, shapes], axis=0)
            self._ev_w_scale = np.concatenate([self._ev_w_scale, scales], axis=0)
            self._ev_intercepts = np.concatenate([self._ev_intercepts, intercepts], axis=0)
            if self._tail_track_mode == "max":
                self._ev_tail = np.concatenate([self._ev_tail, tail[:, -1][:, np.newaxis]], axis=0)
            elif self._tail_track_mode == "all":
                self._ev_tail = np.concatenate([self._ev_tail, tail], axis=0)

        if np.isnan(self._ev_w_shape).any() or np.isnan(self._ev_w_scale).any():
            raise RuntimeError(
                "{} contains bad estimated (NaN) Weibull shapes or scales".format(self.__class__.__name__, ))

    @tf.function
    def _batch_wise_fit(self, anchors_x: tf.Tensor, anchors_y: tf.Tensor,
                        batch_size: tf.Tensor,
                        distance_multiplier: tf.Tensor,
                        x_all: tf.Tensor = None, y_all: tf.Tensor = None,
                        is_ev_update: bool = False, ev_tail_t: tf.Tensor = None):
        """
        Parameters
        ----------
        anchors_x : tf.Tensor, shape (n_samples, n_features).
                    The samples for which Weibull distributions are estimated.
        anchors_y : tf.Tensor, shape (n_samples,). The associated labels of anchors_x.
        batch_size : tf.Tensor, shape (). The batch size that is used to split the anchors into batches.
        distance_multiplier : tf.Tensor, shape (). The distance multiplier.
        x_all : tf.Tensor or None, default is None. All samples that may be used to build the tail.
                If None, it equals anchors_x.
        y_all : tf.Tensor or None, default is None. The related labels of x_all.
                If None, it equals anchors_y.
        is_ev_update : bool, default is False. Flag denoting that the anchors are extreme vectors.
        ev_tail_t : tf.Tensor, shape depends on the tail_track_mode, default is None. The stored tail.
        """
        if x_all is None and y_all is None:
            x_all = anchors_x
            y_all = anchors_y

        n_samples = tf.size(anchors_y)
        n_batches = ctf_util.compute_batch_count(n_samples, batch_size)

        def condition(_i, _1, _2, _3, _4, _5, _6, _7):
            return tf.less(_i, n_batches)

        def body(_i, _shapes, _scales, _requires_update: tf.TensorArray, _tail_array: tf.TensorArray, _intercepts,
                 _shape_incr_mean, _cnt):
            start_idx = _i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, n_samples)

            # The samples for which the tail and Weibull distributions are calculated.
            # This can be a batch of new data or EVs.
            cx_t = anchors_x[start_idx:end_idx]
            cy_t = anchors_y[start_idx:end_idx]

            # In case of partial fit and some tail_track_mode we need to update the existing EVs.
            max_td_t = None
            old_tail_t = None
            if is_ev_update and self._tail_track_mode != "none":
                max_td_t = ev_tail_t[start_idx:end_idx, -1]
                if self._tail_track_mode == "all":
                    old_tail_t = ev_tail_t[start_idx:end_idx]

            # Make tail.
            c_tail_t, tail_sizes, c_requires_update = self._construct_tail(
                cx_t, cy_t, x_all, y_all, max_td_t=max_td_t, ev_tail=old_tail_t, ev_x_t=anchors_x, ev_y_t=anchors_y,
                is_ev_update=is_ev_update)

            tail_tensor_size = tf.size(c_tail_t)

            # Tail may be empty in case of EV update (e.g. all EV anchors and y_all share the same class).
            # But for non-EV anchors this should not happen, as the user needs to fit with a single class which is
            # already checked previously.
            if not is_ev_update:
                tf.assert_greater(tail_tensor_size, tf.constant(0, dtype=tf.int32),
                                  "TFEVM: constructed tail has size 0, this should not happen.")

            def empty_tail_fn(__tail_array):
                return _shapes, _scales, _requires_update, __tail_array, _intercepts, _shape_incr_mean, _cnt

            def non_empty_tail_fn(
                    __shapes, __scales, __requires_update, __tail_array, __intercepts, __shape_incr_mean, __cnt):
                arange_start_to_end = tf.range(start_idx, end_idx)[:, tf.newaxis]

                if self.tail_track_mode != "none":
                    __tail_array = __tail_array.write(_i, c_tail_t)
                if c_requires_update is not None:
                    __requires_update = __requires_update.write(_i, c_requires_update + start_idx)
                    arange_start_to_end = tf.gather(arange_start_to_end, c_requires_update)

                # Apply distance multiplier and normalization before Weibull fitting.
                tail_for_weibull = tf.negative(distance_multiplier) * c_tail_t
                c_intercepts = self._gather_intercepts(tail_for_weibull, tail_sizes)
                tail_for_weibull = self._normalize_fn(tail_for_weibull, c_intercepts)
                __intercepts = tf.tensor_scatter_nd_update(
                    __intercepts, arange_start_to_end, tf.squeeze(c_intercepts, axis=1))

                # Fit Weibull distributions per sample.
                if self._vectorized:
                    c_shapes, c_scales = wb.fit_vectorized(
                        tails=tail_for_weibull, tail_sizes=tf.cast(tail_sizes, self._tf_dtype), max_iter=self._max_iter,
                        tolerance=self._tolerance, shape_init=__shape_incr_mean, eps=self._precision,
                        dtype=self._tf_dtype)
                else:
                    c_shapes, c_scales = self._weibull_fit_map_fn(tail_for_weibull, shape_init=__shape_incr_mean)

                if self.newton_init_mode == "mean":
                    __shape_incr_mean = (tf.cast(__cnt, self._tf_dtype) * __shape_incr_mean) + tf.reduce_sum(c_shapes)
                    __cnt = __cnt + (end_idx - start_idx)
                    __shape_incr_mean = __shape_incr_mean / tf.cast(__cnt, self._tf_dtype)

                __shapes = tf.tensor_scatter_nd_update(__shapes, arange_start_to_end, tf.squeeze(c_shapes, axis=1))
                __scales = tf.tensor_scatter_nd_update(__scales, arange_start_to_end, tf.squeeze(c_scales, axis=1))
                return __shapes, __scales, __requires_update, __tail_array, __intercepts, __shape_incr_mean, __cnt

            _shapes, _scales, _requires_update, _tail_array, _intercepts, _shape_incr_mean, _cnt = tf.cond(
                tf.equal(tail_tensor_size, 0),
                true_fn=lambda: empty_tail_fn(_tail_array),
                false_fn=lambda: non_empty_tail_fn(
                    _shapes, _scales, _requires_update, _tail_array, _intercepts, _shape_incr_mean, _cnt),
                name="if_tail_not_empty_fit_weibull")

            return tf.add(_i, 1), _shapes, _scales, _requires_update, _tail_array, _intercepts, _shape_incr_mean, _cnt

        shapes = tf.zeros((n_samples,), self._tf_dtype)
        scales = tf.zeros((n_samples,), self._tf_dtype)
        requires_update_array = tf.TensorArray(
            tf.int32, size=0, dynamic_size=True, infer_shape=False, element_shape=(None,))

        # tail_array is only used if some tail_track_mode is enabled.
        if self.tail_track_mode == "none":
            tail_array = tf.TensorArray(self._tf_dtype, size=0, dynamic_size=False, infer_shape=True,
                                        element_shape=(0,))
        else:
            tail_array = tf.TensorArray(
                self._tf_dtype, size=0, dynamic_size=True, infer_shape=False, element_shape=(None, None))

        intercepts = tf.zeros((n_samples,), self._tf_dtype)
        shape_incr_mean = tf.ones([], self._tf_dtype)
        cnt = tf.zeros([], tf.int32)

        # This while-loop should prevent memory explosions: parallel_iterations=1
        _, shapes, scales, requires_update_array, tail_array, intercepts, _, _ = tf.while_loop(
            condition, body,
            (tf.constant(0, tf.int32), shapes, scales, requires_update_array, tail_array, intercepts,
             shape_incr_mean, cnt),
            parallel_iterations=1, name="for_all_batches_fit")

        # TensorArray behaves different in (non-)eager execution.
        if not tf.executing_eagerly():
            requires_update_array = requires_update_array.concat()
            tail_array = tail_array.concat()

        return shapes, scales, requires_update_array, tail_array, intercepts

    @tf.function
    def _reduce(self, x_all: tf.Tensor, y_all: tf.Tensor, w_scales_all: tf.Tensor, w_shapes_all: tf.Tensor,
                intercepts_all: tf.Tensor, classes: tf.Tensor, reduction_param: tf.Tensor):
        # Loop over classes: Reduce each class.
        n_classes = tf.shape(classes)[0]

        def cond(_i, _):
            return tf.less(_i, n_classes)

        def body(_i, _idxs_to_keep):
            class_idxs = tf.cast(tf.squeeze(tf.where(tf.equal(y_all, tf.gather(classes, _i))), axis=1), tf.int32)

            def _do_reduction():
                # todo: should we do this batch-wise?
                dists = self._distance_fn(tf.gather(x_all, class_idxs, axis=0))
                dists = self._normalize_fn(tf.negative(dists), tf.gather(intercepts_all, class_idxs, axis=0))
                dists = tf.maximum(dists, tf.constant(0., dtype=self._tf_dtype))

                # Compute probabilities.
                dists = wb.cdf(dists, tf.gather(w_shapes_all, class_idxs, axis=0),
                                    tf.gather(w_scales_all, class_idxs, axis=0), dtype=self._tf_dtype)

                # Set diagonal to one, i.e. each sample has a coverage of itself of 1.
                diag = tf.ones((1, tf.shape(dists)[0]), dtype=dists.dtype)
                dists = tf.linalg.set_diag(dists[tf.newaxis, :], diag)
                dists = tf.squeeze(dists, axis=0)

                # The probabilities are row-wise, i.e.
                # probas[0, 0] is the inclusion probability of sample_0 to sample_0.
                # probas[0, 1] is the inclusion probability of sample_1 to sample_0.
                # probas[1, 0] is the inclusion probability of sample_0 to sample_1.

                if self._reduction_mode == "set_cover":
                    cover_idxs = self._reduce_set_cover(dists, threshold=reduction_param)
                elif self._reduction_mode == "bisection_set_cover":
                    cover_idxs = self._reduce_bisection_set_cover(
                        dists, n_max=reduction_param,
                        eps=tf.convert_to_tensor(self._reduction_tolerance, dtype=self._tf_dtype))
                elif self._reduction_mode == "weighted_set_cover":
                    cover_idxs = self._reduce_weighted_set_cover(dists, n_max=reduction_param)
                else:
                    # should not happen, checked previously
                    raise NotImplementedError(
                        "{} does not implement model reduction mode '{}'".format(
                            self.__class__.__name__, self._reduction_mode))

                return tf.gather(class_idxs, cover_idxs, axis=0)

            if self._reduction_mode == "set_cover":
                class_idxs_reduced = _do_reduction()
            else:
                # skip reduction if the class has <= samples than the desired amount
                skip_reduction = tf.less_equal(tf.size(class_idxs), reduction_param)
                class_idxs_reduced = tf.cond(skip_reduction, true_fn=lambda: class_idxs, false_fn=_do_reduction)

            idxs_to_keep_updated = tf.concat([_idxs_to_keep, class_idxs_reduced], axis=0)
            return tf.add(_i, 1), idxs_to_keep_updated

        _, idxs_to_keep = tf.while_loop(
            cond, body, loop_vars=(tf.constant(0, tf.int32), tf.constant(0, dtype=tf.int32, shape=(0,))),
            shape_invariants=(tf.TensorShape((1,)), tf.TensorShape((None,))),
            name="for_each_class_reduce")
        return idxs_to_keep

    @tf.function
    def _reduce_set_cover(self, probas_cls: tf.Tensor, threshold: tf.Tensor):
        def cond(_cover_idxs, _samples_covering_samples):
            # set-cover problem is solved when all samples are covered
            return tf.logical_not(tf.equal(tf.shape(_samples_covering_samples)[1], 0))

        def body(_cover_idxs, _samples_covering_samples):
            # Count for each sample the amount of covered samples and choose the one that covers most
            cur_max_covering_idx = tf.argmax(
                tf.math.count_nonzero(_samples_covering_samples, axis=1, dtype=np.int32), output_type=tf.int32)

            # keep only not covered samples
            not_covered_idxs = tf.boolean_mask(tf.range(tf.shape(_samples_covering_samples)[1]),
                                               tf.equal(_samples_covering_samples[cur_max_covering_idx], 0))
            _samples_covering_samples_updated = tf.gather(_samples_covering_samples, not_covered_idxs, axis=1)

            cover_idxs_updated = tf.concat([_cover_idxs, [cur_max_covering_idx]], axis=0)
            return cover_idxs_updated, _samples_covering_samples_updated

        # each row contains the idxs that are covered by the "row-sample", note that the idxs have +1 offset,
        # i.e. a zero means that the sample is not covered
        samples_covering_samples = tf.where(tf.greater_equal(probas_cls, threshold),
                                            tf.range(tf.shape(probas_cls)[0]) + 1, 0)
        cover_idxs, _ = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0, dtype=tf.int32, shape=(0,)), samples_covering_samples),
            shape_invariants=(tf.TensorShape((None,)), tf.TensorShape((samples_covering_samples.get_shape()[0], None))),
            parallel_iterations=1)  # no parallel run, order matters

        return cover_idxs

    @tf.function
    def _reduce_bisection_set_cover(self, probas_cls: tf.Tensor, n_max: tf.Tensor, eps):
        n_nmax = tf.shape(probas_cls)[0] - n_max

        def cond(not_finished, t_min, t_max, t_old, _cover_idxs):
            return not_finished

        def body(not_finished, t_min, t_max, t_old, _cover_idxs):
            t_new = (t_min + t_max) / tf.constant(2., dtype=self._tf_dtype)
            cover_idxs_new = self._reduce_set_cover(probas_cls, t_new)
            m = tf.size(cover_idxs_new)

            def finished_fn():
                return cover_idxs_new[:n_max], t_max, t_min, t_new

            def not_finished_fn():
                m_nmax = m - n_max
                n_nmax_ge_m_nmax = tf.greater_equal(n_nmax, m_nmax)
                m_nmax_g_0 = tf.greater(m_nmax, 0)

                def c1_false():
                    # we do not have to exclude M == N_max in check "c2"
                    # since it is checked previously in the "c4"-check
                    c2 = tf.logical_and(n_nmax_ge_m_nmax, tf.logical_not(m_nmax_g_0))
                    c3 = tf.less(n_nmax, m_nmax)
                    __t_min_new = tf.cond(tf.logical_or(c2, c3), lambda: t_new, lambda: t_min)
                    return t_max, __t_min_new

                c1 = tf.logical_and(n_nmax_ge_m_nmax, m_nmax_g_0)
                _t_max_new, _t_min_new = tf.cond(c1, true_fn=lambda: (t_new, t_min), false_fn=c1_false)

                return cover_idxs_new, _t_max_new, _t_min_new, t_new

            c4 = tf.logical_not(tf.logical_or(tf.equal(m, n_max),
                                              tf.logical_not(tf.greater(tf.abs(t_new - t_old), eps))))
            cover_idxs_new, t_max_new, t_min_new, t_old_new = tf.cond(c4, true_fn=not_finished_fn, false_fn=finished_fn)

            return c4, t_min_new, t_max_new, t_old_new, cover_idxs_new

        _, _, _, _, cover_idxs = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(True, dtype=tf.bool),  # not_finished
                       tf.constant(0., dtype=self._tf_dtype),  # t_min
                       tf.constant(1., dtype=self._tf_dtype),  # t_max
                       tf.constant(1., dtype=self._tf_dtype),  # t_old
                       tf.constant(0, dtype=tf.int32, shape=(0,))),  # cover_idxs
            shape_invariants=(tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1,)),
                              tf.TensorShape((1,)), tf.TensorShape((None,))),
            parallel_iterations=1)  # no parallel run, order matters

        return cover_idxs

    @tf.function
    def _reduce_weighted_set_cover(self, probas_cls: tf.Tensor, n_max: tf.Tensor):
        """
        This implementation of the weighted maximum K-set cover model reduction has several limitations:
        1)  The sum of inclusion probabilities for each sample is not held across partial_fits,
            i.e. it needs to be fully recomputed here.
        2)  We do not remove selected samples from the p-vector instead we set the value to -1. This is suboptimal
            since argmax always has to iterate through the whole vector. However, to do it otherwise we need to slice,
            shrink, and concatenate several other tensors, which might be even slower (not tested).
        3)  Since we only set the sum of inclusion probabilities for the selected sample to -1, it gets further
            reduced in following iterations. However, we do not believe that this becomes an issue.
        """

        def cond(_cover_idxs, _p):
            return tf.not_equal(tf.size(_cover_idxs), n_max)

        def body(_cover_idxs, _p):
            # get sample that has the highest sum of inclusion probabilities
            cur_max_covering_idx = tf.argmax(_p, output_type=tf.int32)
            # update probability sum
            _p_new = _p - tf.gather(probas_cls, cur_max_covering_idx, axis=1)
            # set selected sum to -1 to prevent the idxs reselection
            _p_new = tf.tensor_scatter_nd_update(_p_new, [[cur_max_covering_idx]], [-1.])
            cover_idxs_updated = tf.concat([_cover_idxs, [cur_max_covering_idx]], axis=0)
            return cover_idxs_updated, _p_new

        p = tf.reduce_sum(probas_cls, axis=1)
        cover_idxs, _ = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0, dtype=tf.int32, shape=(0,)), p),
            shape_invariants=(tf.TensorShape((None,)), p.get_shape()),
            parallel_iterations=1)  # no parallel run, order matters

        return cover_idxs

    @tf.function
    def _batch_wise_predict_proba(self, x: tf.Tensor, batch_size: tf.Tensor, ev_x_t: tf.Tensor,
                                  ev_y_t: tf.Tensor, intercepts_t: tf.Tensor, scales_t: tf.Tensor, shapes_t: tf.Tensor,
                                  classes: tf.Tensor, cpu_fallback: bool):
        n_samples = tf.shape(x)[0]
        n_batches = ctf_util.compute_batch_count(n_samples, batch_size)
        n_classes = tf.shape(classes)[0]

        def condition(_i, _max_probas):
            return tf.less(_i, n_batches)

        def body(_i, _max_probas):
            # Make batch.
            start_idx = _i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, n_samples)
            cx_t = x[start_idx:end_idx]

            # Auxiliary variables.
            cur_batch_size = end_idx - start_idx
            arange_to_batch_size = tf.range(0, cur_batch_size)[:, tf.newaxis]

            if cpu_fallback:  # If condition is only used during graph building.
                with tf.device("/device:CPU:0"):
                    dists = self._distance_fn(cx_t, ev_x_t)
            else:
                dists = self._distance_fn(cx_t, ev_x_t)

            dists = self._normalize_fn(tf.negative(dists), intercepts_t)
            dists = tf.maximum(dists, tf.constant(0., dtype=self._tf_dtype))

            # Compute probabilities.
            dists = wb.cdf(dists, shapes_t, scales_t, dtype=self._tf_dtype)

            # The probabilities are column-wise, i.e.
            # probas[0, 0] is the inclusion probability of sample_0 to sample_0.
            # probas[0, 1] is the inclusion probability of sample_0 to sample_1.
            # probas[1, 0] is the inclusion probability of sample_1 to sample_0.

            # Loop over classes:
            # Find for each sample the maximum probability of each class.

            def class_condition(_i, _):
                return tf.less(_i, n_classes)

            def class_body(_i, _class_max):
                class_idxs = tf.cast(tf.squeeze(tf.where(tf.equal(ev_y_t, classes[_i])), axis=1), tf.int32)
                cur_cls_max = tf.reduce_max(tf.gather(dists, class_idxs, axis=1), axis=1)

                # idxs to do: _class_max[:, cls] = cmax, where cls is the current class-column
                arange_to_batch_size_cls = tf.concat(
                    [arange_to_batch_size, tf.repeat(_i, cur_batch_size, axis=0)[:, tf.newaxis]], axis=1)

                _class_max = tf.tensor_scatter_nd_update(_class_max, arange_to_batch_size_cls, cur_cls_max)
                return tf.add(_i, 1), _class_max

            class_max = tf.zeros((cur_batch_size, n_classes), self._tf_dtype)
            _, class_max = tf.while_loop(class_condition, class_body, (tf.constant(0, tf.int32), class_max),
                                         name="for_all_samples_in_batch_find_all_class_maxima")

            # End loop over classes.

            arange_start_to_end = tf.range(start_idx, end_idx)[:, tf.newaxis]
            _max_probas = tf.tensor_scatter_nd_update(_max_probas, arange_start_to_end, class_max)
            return tf.add(_i, 1), _max_probas

        # This while-loop should prevent memory explosions: parallel_iterations=1
        max_probas = tf.zeros((n_samples, n_classes), self._tf_dtype)
        _, max_probas = tf.while_loop(condition, body, (tf.constant(0, tf.int32), max_probas),
                                      parallel_iterations=1, name="for_all_batches_predict")
        return max_probas

    @tf.function
    def _batch_wise_predict(self, x: tf.Tensor, batch_size: tf.Tensor, ev_x_t: tf.Tensor,
                            intercepts_t: tf.Tensor, scales_t: tf.Tensor, shapes_t: tf.Tensor, cpu_fallback: bool):
        n_samples = tf.shape(x)[0]
        n_batches = ctf_util.compute_batch_count(n_samples, batch_size)

        def condition(_i, _):
            return tf.less(_i, n_batches)

        def body(_i, _max_idxs):
            # Make batch.
            start_idx = _i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, n_samples)
            cx_t = x[start_idx:end_idx]

            # Compute distances to EVs.
            if cpu_fallback:  # If condition is only used during graph building.
                with tf.device("/device:CPU:0"):
                    dists = self._distance_fn(cx_t, ev_x_t)
            else:
                dists = self._distance_fn(cx_t, ev_x_t)

            dists = self._normalize_fn(tf.negative(dists), intercepts_t)
            dists = tf.maximum(dists, tf.constant(0., dtype=self._tf_dtype))

            # Compute probabilities
            dists = wb.cdf(dists, shapes_t, scales_t, dtype=self._tf_dtype)

            # The probabilities are column-wise, i.e.
            # probas[0, 0] is the inclusion probability of sample_0 to sample_0.
            # probas[0, 1] is the inclusion probability of sample_0 to sample_1.
            # probas[1, 0] is the inclusion probability of sample_1 to sample_0.

            arange_start_to_end = tf.range(start_idx, end_idx)[:, tf.newaxis]
            argmax = tf.cast(tf.argmax(dists, axis=1), tf.int32)

            _max_idxs = tf.tensor_scatter_nd_update(_max_idxs, arange_start_to_end, argmax)
            return tf.add(_i, 1), _max_idxs

        # This while-loop should prevent memory explosions: parallel_iterations=1
        max_idxs = tf.zeros((n_samples,), tf.int32)
        _, max_idxs = tf.while_loop(condition, body, (tf.constant(0, tf.int32), max_idxs),
                                    parallel_iterations=1, name="for_all_batches_predict")

        return max_idxs

    @tf.function
    def _construct_tail(self, cx_t: tf.Tensor, cy_t: tf.Tensor, x_t: tf.Tensor = None, y_t: tf.Tensor = None,
                        max_td_t: tf.Tensor = None, ev_tail: tf.Tensor = None,
                        ev_x_t: tf.Tensor = None, ev_y_t: tf.Tensor = None,
                        is_ev_update: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Construct the tail matrix.

        Parameters
        ----------
        cx_t : tf.Tensor, shape (n_samples, n_features).
               The samples for which the tails are built.
        cy_t : tf.Tensor, shape (n_samples,).
               The labels of the samples for which the tails are built.
        x_t : tf.Tensor, shape (m_samples, n_features), default is None, i.e. 'cx_t' is used.
              The samples that are searched for the tail.
        y_t : tf.Tensor, shape (m_samples,), default is None, i.e. 'cy_t' is used.
              The labels of the samples that are searched for the tail.
        max_td_t : tf.Tensor, shape (n_samples,), default is None.
                   If given, the 'cx_t' and 'cy_t' are part of the EVs. 'max_td_t' is the maximum tail distance per
                   sample in 'cx_t' and is used to check whether this EV requires an update.
        ev_tail : tf.Tensor, shape (n_samples, some_tail_size), default is None.
                  Given if the 'self._tail_track_mode' equals 'all'. This is the whole tail of the current EV samples.
        ev_x_t : tf.Tensor, shape (k_samples, n_features), default is None.
                 In case there are too many EVs such that they need to be updated batch-wise, 'cx_t' will contain a
                 subset of EVs and 'ev_x_t' all of them.

        Returns
        -------
        tail_tensor : tf.Tensor, shape (n_samples, self.tail_size).
                      The tail matrix, for each sample the 'self.tail_size' nearest negative samples.
                      If there are not enough negatives, the entries are filled with inf.
        tail_size_per_samples : tf.Tensor, shape (n_samples,).
                                The amount of negatives per sample (excluding the inf entries).
        requires_update : tf.Tensor, shape (n_updates,).
        """
        dists, n_negatives_per_sample = self._compute_distances(cx_t, cy_t, x_t, y_t, is_ev_update=is_ev_update)

        requires_update = None
        if max_td_t is not None and self._tail_track_mode != "none":
            dists, n_negatives_per_sample, requires_update = self._tail_tracking(
                cx_t, cy_t, dists, n_negatives_per_sample, max_td_t, ev_tail, ev_x_t, ev_y_t)

        def _false_branch():
            tail_size = tf.minimum(tf.shape(dists)[1], self._tail_size)
            _tail_size_per_sample = tf.minimum(n_negatives_per_sample, self._tail_size)[:, tf.newaxis]
            _tail_tensor = tf.math.top_k(tf.math.negative(dists), k=tail_size, sorted=True).values
            _tail_tensor = tf.math.negative(_tail_tensor)  # undo negative
            return [_tail_tensor, _tail_size_per_sample]

        empty_dists = tf.equal(tf.size(dists), 0)
        tail_tensor, tail_size_per_sample = tf.cond(
            empty_dists, true_fn=lambda: [tf.zeros((0, 0), dtype=self._tf_dtype), tf.zeros((0, 0), dtype=tf.int32)],
            false_fn=_false_branch, name="if_not_empty_dists_make_tail")

        return tail_tensor, tail_size_per_sample, requires_update

    @tf.function
    def _compute_distances(self, anchors_x: tf.Tensor, anchors_y: tf.Tensor, x_all: tf.Tensor = None,
                           y_all: tf.Tensor = None, is_ev_update: bool = False):
        if y_all is None:
            neq_mask = tf.not_equal(anchors_y[:, tf.newaxis], anchors_y[tf.newaxis, :])
        else:
            neq_mask = tf.not_equal(anchors_y[:, tf.newaxis], y_all[tf.newaxis, :])

        n_negatives_per_sample = tf.reduce_sum(tf.cast(neq_mask, tf.int32), axis=-1)

        # EVs may have no new negatives.
        if not is_ev_update:
            # Raise exception if there is a sample with no negatives.
            with tf.control_dependencies([anchors_x]):
                tf.assert_greater(n_negatives_per_sample, tf.constant(0, dtype=tf.int32))

        # x_all can be None, distance function has to deal with this.
        dists = self._distance_fn(anchors_x, x_all)

        # Set intra-class distances to inf.
        inter_class_dists = tf.where(neq_mask, dists, math.inf)
        return inter_class_dists, n_negatives_per_sample

    @tf.function
    def _tail_tracking(self, cx_t, cy_t, dists, n_negatives_per_sample, max_td_t, ev_tail=None,
                       ev_x_t: tf.Tensor = None, ev_y_t: tf.Tensor = None):
        cur_min_td = tf.reduce_min(dists, axis=1)
        requires_update = tf.cast(tf.squeeze(tf.where(tf.less(cur_min_td, max_td_t)), axis=1), tf.int32)
        dists = tf.gather(dists, requires_update)
        n_negatives_per_sample = tf.gather(n_negatives_per_sample, requires_update)

        if self._tail_track_mode == "max":
            ev_tail, ev_n_negatives = self._compute_distances(tf.gather(cx_t, requires_update),
                                                              tf.gather(cy_t, requires_update),
                                                              ev_x_t, ev_y_t)
        elif self._tail_track_mode == "all":
            ev_tail = tf.gather(ev_tail, requires_update)
            ev_n_negatives = tf.reduce_sum(
                tf.cast(tf.math.logical_not(tf.math.is_inf(ev_tail)), dtype=tf.int32), axis=1)
        else:
            # This should never happen as the tail_track_mode is checked in the setter.
            raise NotImplementedError("{} does not know given tail_track_mode '{}'.".format(
                self.__class__.__name__, self._tail_track_mode))

        n_negatives_per_sample = n_negatives_per_sample + ev_n_negatives
        dists = tf.concat([dists, ev_tail], axis=1)
        return dists, n_negatives_per_sample, requires_update

    @tf.function
    def _gather_intercepts(self, tail_tensor: tf.Tensor, tail_size_per_samples: tf.Tensor):
        arange = tf.range(0, tf.shape(tail_tensor)[0])[:, tf.newaxis]
        max_tail_idx = tf.concat([arange, tf.subtract(tail_size_per_samples, 1)], axis=1)
        return tf.gather_nd(tail_tensor, max_tail_idx)[:, tf.newaxis]

    @tf.function
    def _normalize_fn(self, tail_tensor, intercepts):
        return tail_tensor + tf.constant(1., dtype=self._tf_dtype) - intercepts

    @tf.function
    def _weibull_fit_map_fn(self, tail_t: tf.Tensor, shape_init):
        return tf.map_fn(
            lambda _tail: wb.fit(
                ctf_util.remove_infs(_tail),
                max_iter=self._max_iter, tolerance=self._tolerance, shape_init=shape_init, eps=self._precision,
                dtype=self._tf_dtype),
            elems=tail_t, fn_output_signature=(self._tf_dtype, self._tf_dtype), parallel_iterations=self._parallel)

    def __reset_dbscan(self):
        self.dbscan_kwargs = self._dbscan_kwargs  # resets the DBSCAN object

    def __check_fit_samples(self, x: np.ndarray, y: np.ndarray):
        if len(x.shape) != 2:
            raise ValueError(
                "{} requires a 2d (n_samples, n_features) sample input, "
                "but got {}d".format(self.__class__.__name__, len(x.shape)))

        if len(y.shape) != 1:
            raise ValueError(
                "{} requires a 1d (n_samples,) label input, but got {}d".format(
                    self.__class__.__name__, len(y.shape)))

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "{} requires the sample and label inputs to have the same size in the first dimension, "
                "but got {} and {}.".format(self.__class__.__name__, x.shape[0], y.shape[0]))

    def __check_predict_samples(self, x: np.ndarray):
        if self.is_empty():
            raise NotFittedError("{} is not fitted yet".format(self.__class__.__name__))

        if len(x.shape) != 2:
            raise ValueError(
                "{} requires a 2d (n_samples, n_features) sample input, "
                "but got {}d".format(self.__class__.__name__, len(x.shape)))

        if x.shape[1] != self._ev_x.shape[1]:
            raise ValueError(
                "{} requires consistent features, the saved extreme vectors have {}, but the prediction data "
                "has {}".format(self.__class__.__name__, self._ev_x.shape[1], x.shape[1]))

    def __check_cpu_fallback(self, arr: np.ndarray):
        """
        Check if 'cpu_fallback' is greater 0 and the size of the given array is also greater than 'cpu_fallback'.

        Parameters
        ----------
        arr: np.ndarray, the array to be checked.

        Returns
        -------
        True, if the computation with 'arr' should be moved to CPU.
        """
        return 0 < self._cpu_fallback < np.size(arr)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        klass = self.__class__
        new_obj = klass.__new__(klass)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == "_distance_fn":
                continue

            setattr(new_obj, k, copy.deepcopy(v, memo))

        new_obj._distance_fn = new_obj.__get_distance_fn(new_obj._distance)
        return new_obj

    @property
    def tail_size(self):
        return self._tail_size

    @tail_size.setter
    def tail_size(self, tail_size: int):
        if tail_size < 1:
            raise ValueError(
                "{} requires a tail_size > 0, but is {}".format(self.__class__.__name__, tail_size))
        self._tail_size = tail_size

    @property
    def distance_multiplier(self):
        return self._distance_multiplier

    @distance_multiplier.setter
    def distance_multiplier(self, distance_multiplier: float):
        if not distance_multiplier > 0. or distance_multiplier > 1.:
            raise ValueError(
                "{} requires a distance_multiplier in range (0, 1], but is {}".format(
                    self.__class__.__name__, distance_multiplier))
        self._distance_multiplier = distance_multiplier

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance: str):
        self._distance_fn = self.__get_distance_fn(distance)
        self._distance = distance

    def __get_distance_fn(self, distance: str):
        _distance = distance.lower()
        if _distance == "euclidean":
            return ctf_util.euclidean_distances
        elif _distance == "squared_euclidean":
            return lambda _a, _b=None: ctf_util.euclidean_distances(_a, _b, squared=True)
        else:
            raise NotImplementedError(
                "{} does not support given distance '{}'.".format(self.__class__.__name__, distance))

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        if max_iter < 1:
            raise ValueError("Number of Newton iterations has be >0, but is {}".format(max_iter))
        self._max_iter = max_iter

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        if tolerance < keras.backend.epsilon():
            raise ValueError(
                "{} given parameter tolerance is {}, but should not be less than the actual Keras precision {}.".format(
                    self.__class__.__name__, tolerance, keras.backend.epsilon()))
        self._tolerance = tolerance

    @property
    def reduction_mode(self):
        return self._reduction_mode

    @reduction_mode.setter
    def reduction_mode(self, reduction_mode: str):
        _reduction_mode = reduction_mode.lower()
        if _reduction_mode not in ["set_cover", "bisection_set_cover", "weighted_set_cover"]:
            raise ValueError(
                "{} does not know given reduction_mode '{}'.".format(
                    self.__class__.__name__, reduction_mode))
        self._reduction_mode = reduction_mode

    @property
    def reduction_param(self):
        return self._reduction_param

    @reduction_param.setter
    def reduction_param(self, reduction_param: Union[float, int]):
        if not isinstance(reduction_param, (float, int)):
            raise TypeError(
                "{} requires reduction_param to be of type float or int, but type is '{}'.".format(
                    self.__class__.__name__, type(reduction_param)))

        if isinstance(reduction_param, int) and reduction_param < 1:
            raise ValueError(
                "{} reduction_param is of type int and has to be greater 0, but is '{}'.".format(
                    self.__class__.__name__, reduction_param))

        if isinstance(reduction_param, float) and (reduction_param < 0.0 or reduction_param > 1.0):
            raise ValueError(
                "{} reduction_param is of type float and has to be in range [0, 1], but is '{}'.".format(
                    self.__class__.__name__, reduction_param))

        self._reduction_param = reduction_param

    @property
    def reduce_after_fit(self):
        return self._reduce_after_fit

    @reduce_after_fit.setter
    def reduce_after_fit(self, reduce_after_fit):
        self._reduce_after_fit = reduce_after_fit

    @property
    def reduction_tolerance(self):
        return self._reduction_tolerance

    @reduction_tolerance.setter
    def reduction_tolerance(self, reduction_tolerance):
        if reduction_tolerance < keras.backend.epsilon():
            raise ValueError(
                "{} given parameter reduction_tolerance is {}, but should not be less than the actual Keras "
                "precision {}.".format(self.__class__.__name__, reduction_tolerance, keras.backend.epsilon()))
        self._reduction_tolerance = reduction_tolerance

    @property
    def tail_track_mode(self):
        return self._tail_track_mode

    @tail_track_mode.setter
    def tail_track_mode(self, tail_track_mode: str):
        _tail_track_mode = tail_track_mode.lower()
        if _tail_track_mode not in ["none", "max", "all"]:
            raise ValueError(
                "{} does not know given tail_track_mode '{}'.".format(
                    self.__class__.__name__, tail_track_mode))
        self._tail_track_mode = tail_track_mode

    @property
    def dbscan_kwargs(self):
        return self._dbscan_kwargs

    @dbscan_kwargs.setter
    def dbscan_kwargs(self, dbscan_kwargs):
        _dbscan_kwargs = dbscan_kwargs
        if dbscan_kwargs == "normal":
            _dbscan_kwargs = {"eps": 0.3, "min_samples": 1, "n_jobs": -1}

        # Map n_jobs=0 to n_jobs=-1 to be compatible with sklearn and to use all processors.
        if _dbscan_kwargs is not None and _dbscan_kwargs.get("n_jobs", None) == 0:
            _dbscan_kwargs["n_jobs"] = -1

        self._dbscan_kwargs = dbscan_kwargs
        self._dbscan = None
        if _dbscan_kwargs is not None:
            self._dbscan = DBSCAN(**_dbscan_kwargs)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        if batch_size < 0:
            raise ValueError(
                "{} requires a batch_size >= 0, but is {}".format(self.__class__.__name__, batch_size))
        self._batch_size = batch_size

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        if precision < keras.backend.epsilon():
            raise ValueError(
                "{} given precision is {}, but should not be less than the actual Keras precision {}.".format(
                    self.__class__.__name__, precision, keras.backend.epsilon()))
        self._precision = precision

    @property
    def vectorized(self):
        return self._vectorized

    @vectorized.setter
    def vectorized(self, vectorized):
        self._vectorized = vectorized

    @property
    def parallel(self):
        return self._parallel

    @parallel.setter
    def parallel(self, parallel):
        self._parallel = parallel

    @property
    def newton_init_mode(self):
        return self._newton_init_mode

    @newton_init_mode.setter
    def newton_init_mode(self, newton_init_mode):
        _newton_init_mode = newton_init_mode.lower()
        if _newton_init_mode not in ["ones", "mean"]:
            raise ValueError(
                "{} does not know given unknown_mode '{}'.".format(
                    self.__class__.__name__, newton_init_mode))
        self._newton_init_mode = newton_init_mode

    @property
    def cpu_fallback(self):
        return self._cpu_fallback

    @cpu_fallback.setter
    def cpu_fallback(self, cpu_fallback):
        self._cpu_fallback = cpu_fallback

    @property
    def dtype(self) -> str:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: str):
        if not isinstance(dtype, str):
            raise ValueError(
                "{} requires dtype in str format but type is '{}'".format(self.__class__.__name__, type(dtype)))

        _dtype = dtype.lower()

        if _dtype == "float32":
            tf_dtype = tf.float32
        elif _dtype == "float64":
            tf_dtype = tf.float64
        else:
            raise ValueError("{} does not support given dtype '{}'".format(self.__class__.__name__, dtype))

        self._dtype = dtype
        # hide tf.dtype in extra attribute since scikit-learn can have
        # issues with the representation when used in meta-estimators
        self._tf_dtype = tf_dtype

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        if self._verbose > 0:
            handler = logging.StreamHandler()
            handler.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
            self._logger = logging.getLogger("TF-EVM")
            if not self._logger.hasHandlers():
                self._logger.addHandler(handler)

    @property
    def statistics(self):
        return self._statistics

    @statistics.setter
    def statistics(self, statistics: Union[str, List[str]]):
        self._statistics = statistics

        if statistics is None:
            return

        _statistics = statistics
        if isinstance(statistics, str) and statistics == "all":
            _statistics = ["update_ratio", "cluster_diff"]

        for stat in _statistics:
            self.add_statistic(stat, append=False)

    def add_statistic(self, stat: str, append=True):
        if stat not in ["update_ratio", "cluster_diff"]:
            raise ValueError("{} does not now given statistic '{}'".format(self.__class__.__name__, stat))

        if append:
            self._statistics.append(stat)

        if stat not in self._statistic_dir.keys():
            self._statistic_dir[stat] = []

    @staticmethod
    def dbscan_defaults():
        return {'eps': 0.3, 'min_samples': 1, 'n_jobs': -1}

    @property
    def classes_(self):
        return self._label_encoder.classes_
