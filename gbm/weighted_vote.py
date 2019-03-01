# coding: utf-8

import numpy
import time

from gbm import Data


class WeightedVote(object):
    def __init__(self, knn, boosting, logistic_regression, learning_rate=0.1, eps=1e-10, max_iter=100):
        self.knn = knn
        self.boosting = boosting
        self.logistic_regression = logistic_regression

        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter

        self.weights = numpy.ndarray(shape=(3, ))

    def fit_models(self, data):
        # type: (Data) -> None

        print('fit knn')
        _now = time.time()
        self.knn.fit(data.data, data.target)
        print('total time: {}'.format(time.time() - _now))

        print('fit logistic regression')
        _now = time.time()
        self.logistic_regression.fit(data.data, data.target)
        print('total time: {}'.format(time.time() - _now))

        print('fit boosting')
        _now = time.time()
        self.boosting.fit(data)
        print('total time: {}'.format(time.time() - _now))

    def fit(self, data):
        # type: (Data) -> None

        self.fit_models(data)

        models_predict = numpy.ndarray(shape=(3, data.shape[0]))
        for i in range(self.max_iter):







