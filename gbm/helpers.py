# coding: utf-8

import numpy


class Data(object):
    def __init__(self, data, target):
        # type: (numpy.ndarray, numpy.ndarray) -> None

        self.data = data
        self.target = target

        self._prepared = False

    @property
    def prepared(self):
        return self._prepared

    @property
    def shape(self):
        return self.data.shape

    def setup(self):
        # type: () -> None

        if self._prepared:
            return

        if not isinstance(self.data, numpy.ndarray):
            raise ValueError('data is not numpy array')

        self._prepared = True


class Predicate(object):
    def __init__(self, feature_id, value):
        self._feature_id = feature_id
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def feature_id(self):
        return self._feature_id

    def __call__(self, sample):
        # type: (numpy.ndarray) -> bool
        return sample[self.feature_id] > self.value


class TreeNode(object):
    def __init__(self, mask=None):
        self.predicate = None
        self.leaf_value = None
        self.depth = None

        self.left = None
        self.right = None

        self.mask = mask


class Tree(object):
    def __init__(self):
        self.root = TreeNode()
