# coding: utf-8

import numpy


def sigmoid(target):
    # type: (numpy.ndarray) -> numpy.ndarray
    return 1 / (1 + numpy.exp(-target))


def log_loss(target, predict):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    p = sigmoid(predict) + 1e-7
    return (-target * numpy.log(p) - (1 - target) * numpy.log(1 - p)).sum()


def negative_gradient(target, previous_predict):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    return target - sigmoid(previous_predict)

