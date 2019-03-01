# coding: utf-8

import numpy


def sigmoid(target):
    # type: (numpy.ndarray) -> numpy.ndarray
    return 1 / (1 + numpy.exp(-target))


def log_loss(target, predict):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    return numpy.log(1 + numpy.exp(-target * predict))
