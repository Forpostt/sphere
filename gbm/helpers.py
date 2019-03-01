# coding: utf-8

import numpy


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
        self.is_leaf = False

        self.mask = mask


class Tree(object):
    def __init__(self):
        self.root = TreeNode()
