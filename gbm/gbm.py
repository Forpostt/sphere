# coding: utf-8

import tqdm
import numpy

from gbm import Data, DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor as SkTree


class Estimator(object):
    def __init__(self, model, b):
        self.model = model
        self.b = b

    def predict(self, data):
        return self.model.predict(data) * self.b


class GradientBoostingClassifier(object):
    @staticmethod
    def _sigmoid(target):
        # type: (numpy.ndarray) -> numpy.ndarray
        return 1 / (1 + numpy.exp(-target))
    
    def __init__(self, n_estimators, max_depth=None, init_model=None, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.init_model = init_model
        self.trained_estimators = []

    def _loss(self, target, predict):
        # type: (numpy.ndarray, numpy.ndarray) -> float
        p = self._sigmoid(predict) + 1e-7
        return (-target * numpy.log(p) - (1 - target) * numpy.log(1 - p)).sum()

    def _anti_gradient(self, target, previous_predict):
        # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
        return target - self._sigmoid(previous_predict)

    def _best_b(self, target, model_predict, previous_predict):
        # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray) -> float
        def g(x):
            res = ((self._sigmoid(previous_predict + x * model_predict) - target) * model_predict)
            return res.sum()

        def dg(x):
            res = (
                self._sigmoid(previous_predict + x * model_predict) *
                (1 - self._sigmoid(previous_predict + x * model_predict)) *
                numpy.power(model_predict, 2)
            )
            return res.sum()

        b = 0
        while abs(g(b)) > 1e-6:
            b -= g(b) / dg(b)
        return b

    def fit(self, data, eps=1e-7):
        # type: (Data) -> GradientBoostingClassifier

        previous_predict = numpy.ones(shape=(data.shape[0], ))
        if self.init_model is not None:
            self.init_model.fit(data)
            previous_predict = self.init_model.predict(data)

        for _ in range(1, self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            # tree = SkTree(max_depth=self.max_depth)
            anti_gradient = self._anti_gradient(data.target, previous_predict)
            tree_data = Data(data=data.data, target=anti_gradient)

            tree.fit(tree_data)
            # tree.fit(tree_data.data, tree_data.target)
            tree_predict = tree.predict(tree_data)
            # tree_predict = tree.predict(tree_data.data)
            b = self._best_b(data.target.reshape((-1, )), tree_predict, previous_predict)

            self.trained_estimators.append(Estimator(model=tree, b=b))
            previous_predict += b * tree_predict * self.learning_rate

            if abs(self._loss(data.target, (previous_predict > 0) * 1)) < eps:
                print('model is ready')
                return self

        return self

    def predict(self, data):
        # type: (Data) -> numpy.ndarray

        predict = numpy.ones(shape=(data.shape[0], ))
        if self.init_model is not None:
            predict = self.init_model.predict(data)
            
        for estimator in self.trained_estimators:
            predict += estimator.model.predict(data) * estimator.b * self.learning_rate
            # predict += estimator.model.predict(data.data) * estimator.b * self.learning_rate

        return (predict > 0) * 1

    def staged_predict(self, data):
        # type: (Data) -> None

        current_predict = numpy.ones(shape=(data.shape[0],))
        if self.init_model is not None:
            current_predict = self.init_model.predict(data.data)
        yield (current_predict > 0) * 1

        for i, estimator in enumerate(self.trained_estimators):
            # current_predict += estimator.model.predict(data.data) * estimator.b * self.learning_rate
            current_predict += estimator.model.predict(data) * estimator.b * self.learning_rate
            yield (current_predict > 0) * 1

    def staged_predict_proba(self, data):
        # type: (Data) -> None

        current_predict = numpy.ones(shape=(data.shape[0],))
        if self.init_model is not None:
            current_predict = self.init_model.predict(data.data)
        yield self._sigmoid(current_predict)

        for i, estimator in enumerate(self.trained_estimators):
            # current_predict += estimator.model.predict(data.data) * estimator.b * self.learning_rate
            current_predict += estimator.model.predict(data) * estimator.b * self.learning_rate
            yield self._sigmoid(current_predict)

    def reset(self):
        self.trained_estimators = []
