# coding: utf-8

import tqdm
import numpy

from gbm import DecisionTreeRegressor
from gbm.utils import sigmoid


class Estimator(object):
    def __init__(self, model, b):
        self.model = model
        self.b = b

    def predict(self, data):
        return self.model.predict(data) * self.b


class GradientBoostingClassifier(object):    
    @staticmethod
    def _loss(target, predict):
        # type: (numpy.ndarray, numpy.ndarray) -> float
        p = sigmoid(predict) + 1e-7
        return (-target * numpy.log(p) - (1 - target) * numpy.log(1 - p)).sum()

    @staticmethod
    def negative_gradient(target, previous_predict):
        # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
        return target - sigmoid(previous_predict)

    def __init__(self, n_estimators, max_depth=None, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.trained_estimators = []
        self.init_scale = None

    def fit(self, data, target, eps=1e-7):
        # type: (numpy.ndarray, numpy.ndarray) -> GradientBoostingClassifier

        pos = numpy.sum(target)
        neg = target.shape[0] - pos
        self.init_scale = numpy.log(pos / neg)
        previous_predict = numpy.ones(shape=(data.shape[0], )) * self.init_scale

        for _ in tqdm.tqdm(range(self.n_estimators)):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            anti_gradient = self.negative_gradient(target, previous_predict)

            tree.fit(data, anti_gradient)
            previous_predict = self.update_tree(tree, data, target, previous_predict)
            self.trained_estimators.append(tree)

            if abs(self._loss(target, (previous_predict > 0) * 1)) < eps:
                print('model is ready')
                return self

        return self

    def update_tree(self, tree, data, target, previous_predict):
        # type: (DecisionTreeRegressor, numpy.ndarray, numpy.ndarray) -> numpy.ndarray

        for node in tree.leafs():
            node_target = target[node.mask]
            node_prob = sigmoid(previous_predict[node.mask])

            numerator = numpy.sum(node_prob - node_target)
            denominator = numpy.sum(node_prob * (1 - node_prob))

            if abs(denominator) < 1e-17:
                node.leaf_value = 0.0
            else:
                node.leaf_value = -numerator / denominator * self.learning_rate

        return previous_predict + tree.predict(data)

    def predict(self, data):
        # type: (numpy.ndarray) -> numpy.ndarray

        predict = numpy.ones(shape=(data.shape[0],)) * self.init_scale

        for estimator in self.trained_estimators:
            predict += estimator.predict(data)

        return (sigmoid(predict) > 0.5) * 1

    def staged_predict(self, data):
        # type: (numpy.ndarray) -> None

        current_predict = numpy.ones(shape=(data.shape[0],)) * self.init_scale

        for i, estimator in enumerate(self.trained_estimators):
            current_predict += estimator.predict(data)
            yield (sigmoid(current_predict) > 0.5) * 1

    def staged_predict_proba(self, data):
        # type: (numpy.ndarray) -> None

        current_predict = numpy.ones(shape=(data.shape[0],)) * self.init_scale

        for i, estimator in enumerate(self.trained_estimators):
            current_predict += estimator.predict(data)
            yield sigmoid(current_predict)

    def reset(self):
        self.trained_estimators = []
