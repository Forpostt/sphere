# coding: utf-8

import numpy
import time
import tqdm

from gbm.utils import negative_gradient, log_loss, sigmoid


class WeightedVote(object):
    def __init__(self, knn, boosting, logistic_regression, learning_rate=0.1, eps=1e-10, max_iter=100):
        self.knn = knn
        self.boosting = boosting
        self.logistic_regression = logistic_regression

        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter

        self.models_prepared = False
        self.weights = numpy.ones(shape=(3, 1))

    def fit_models(self, data, target):
        # type: (numpy.ndarray, numpy.ndarray) -> None

        print('fit knn')
        _now = time.time()
        self.knn.fit(data, target)
        print('total time: {}'.format(time.time() - _now))

        print('fit logistic regression')
        _now = time.time()
        self.logistic_regression.fit(data, target)
        print('total time: {}'.format(time.time() - _now))

        print('fit boosting')
        _now = time.time()
        self.boosting.fit(data, target)
        print('total time: {}'.format(time.time() - _now))

        self.models_prepared = True

    def predict_models(self, data):
        # type: (numpy.ndarray) -> numpy.ndarray

        models_predict = numpy.ndarray(shape=(3, data.shape[0]))
        models_predict[0, :] = self.knn.predict(data)
        models_predict[1, :] = self.logistic_regression.predict(data)
        models_predict[2, :] = self.boosting.predict(data)
        return models_predict

    def fit(self, data, target):
        # type: (numpy.ndarray, numpy.ndarray) -> WeightedVote

        if not self.models_prepared:
            self.fit_models(data, target)

        models_predict = self.predict_models(data)

        for _ in tqdm.tqdm(range(self.max_iter)):
            predict = (models_predict * self.weights).sum(axis=0)
            negative_gradient_ = negative_gradient(target, predict).reshape((1, -1))
            self.weights += (negative_gradient_ * models_predict).sum(axis=1).reshape(3, 1)

            if log_loss(target, (sigmoid(models_predict * self.weights) > 0.5) * 1) < self.eps:
                print('model is ready')
                break

        return self

    def l2(self):
        return numpy.power(self.weights, 2).sum()

    def predict(self, data):
        # type: (numpy.ndarray) -> numpy.ndarray

        models_predict = self.predict_models(data)
        return ((models_predict * self.weights).sum(axis=0) > 0) * 1

    def staged_predict_proba(self, data):
        # type: (numpy.ndarray) -> numpy.ndarray

        knn_predict = self.knn.predict(data) * self.weights[0][0]
        log_reg_predict = self.logistic_regression.predict(data) * self.weights[1][0]

        for predict in self.boosting.staged_predict(data):
            print(numpy.unique(predict * self.weights[2][0] + knn_predict + log_reg_predict))
            yield sigmoid(predict * self.weights[2][0] + knn_predict + log_reg_predict)

    def reset(self, with_models=False):
        self.weights = numpy.ones(shape=(3, 1))
