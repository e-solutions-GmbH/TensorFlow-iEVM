import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable TensorFlow warnings

import logging
import random
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
import tensorflow as tf
from tensorflow.python.framework import random_seed

from tf_ievm.ievm import TFiEVM


def main():
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    random_seed.set_seed(seed)
    tf.get_logger().setLevel(logging.ERROR)
    tf.config.run_functions_eagerly(True)

    # Load dataset, make train/test split
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # Setup classification pipeline
    pipe = Pipeline([("scale", None),
                     ("reduce", None),
                     ("clf", TFiEVM())])

    # Define parameter grid
    param_grid = {"scale": [None, StandardScaler(), Normalizer()],
                  "reduce": [None, PCA(16), PCA(32)],
                  "clf__tail_size": [100, 200, 300],
                  "clf__distance_multiplier": [0.3, 0.5, 0.7]}

    # Run gridsearch
    cv = GridSearchCV(pipe, param_grid=param_grid,
                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed), verbose=4)
    start = time.time()
    cv.fit(x_train, y_train)
    print("Training took {} min".format(np.around((time.time() - start) / 60., decimals=3)))

    # Check test set
    acc = accuracy_score(y_test, cv.predict(x_test))
    print("Achieved {}% accuracy with best parameters: {}".format(np.around(acc * 100., decimals=2), cv.best_params_))

    # Do model reduction
    pipe = cv.best_estimator_
    pipe.set_params(clf__reduction_param=0.9)
    start = time.time()
    pipe.named_steps.clf.reduce()
    print("Reduction took {} sec".format(np.around(time.time() - start, decimals=3)))

    # Check test set after model reduction
    acc = accuracy_score(y_test, pipe.predict(x_test))
    print("Achieved {}% accuracy after set cover model reduction and coverage threshold of {}".format(
        np.around(acc * 100., decimals=2), pipe.named_steps.clf.reduction_param))


if __name__ == '__main__':
    main()
