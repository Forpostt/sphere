# coding: utf-8

import numpy
from gbm.helpers import Data, Tree, TreeNode, Predicate


class DecisionTreeRegressor(object):
    def _fit_node(self, node, data, eps):
        # type: (TreeNode, Data, float) -> None

        # print('node mask', node.mask)

        node.leaf_value = data.target[node.mask].mean()
        if node.mask.shape[0] == 1:
            return

        if node.mask is None:
            raise RuntimeError('tree node has empty mask')

        # Sort data for this node
        sorted_args = numpy.argsort(data.data[node.mask], axis=0)
        sorted_data = data.data[node.mask][sorted_args, numpy.arange(0, sorted_args.shape[1])]
        sorted_target = numpy.tile(data.target[node.mask], (sorted_args.shape[1], 1)).transpose()[sorted_args, numpy.arange(0, sorted_args.shape[1])]

        target_cumsum = sorted_target.cumsum(axis=0)
        cumsum_range = numpy.arange(1, target_cumsum.shape[0]).reshape((-1, 1))

        # Find losses for all splits
        losses = (
            numpy.power(target_cumsum[:-1], 2) / cumsum_range +
            numpy.power(target_cumsum[-1] - target_cumsum[:-1], 2) / (sorted_target.shape[0] - cumsum_range)
        )

        repeated_mask = sorted_data[1:] == sorted_data[:-1]
        losses = numpy.ma.array(losses, mask=repeated_mask)
        losses[losses.mask] = 0.0

        # Find best split
        best_feature_splits = losses.argmax(axis=0)
        best_loss = losses[best_feature_splits, numpy.arange(0, best_feature_splits.shape[0])].max()
        best_feature = numpy.random.choice(numpy.where(losses[best_feature_splits, numpy.arange(0, best_feature_splits.shape[0])] == best_loss)[0])

        if self.max_depth is not None and node.depth >= self.max_depth:
            return

        # Fixme: change to true eps
        if losses[best_feature_splits[best_feature], best_feature] > eps:
            # Samples with best split on best_feature
            split_sample_1 = node.mask[sorted_args[best_feature_splits[best_feature], best_feature]]
            split_sample_2 = node.mask[sorted_args[best_feature_splits[best_feature] + 1, best_feature]]

            # Create predicate
            split_value = (data.data[split_sample_1, best_feature] + data.data[split_sample_2, best_feature]) / 2
            node.predicate = Predicate(feature_id=best_feature, value=split_value)

            # Split data
            node.left = TreeNode(mask=node.mask[sorted_args[:best_feature_splits[best_feature] + 1, best_feature]])
            node.right = TreeNode(mask=node.mask[sorted_args[best_feature_splits[best_feature] + 1:, best_feature]])
            node.left.depth = node.right.depth = node.depth + 1

            self._fit_node(node.left, data, eps)
            self._fit_node(node.right, data, eps)

    def __init__(self, max_depth=None):
        # type: () -> None
        self.tree = Tree()
        self.max_depth = max_depth

    def reset(self):
        self.tree = Tree()

    def fit(self, data, eps=1e-7):
        # type: (Data) -> DecisionTreeRegressor

        if not isinstance(data, Data):
            raise ValueError('wrong data type, got: {}, expected: {}'.format(type(data), Data))

        if not data.prepared:
            data.setup()

        self.tree.root.mask = numpy.arange(0, data.shape[0], dtype=numpy.uint64)
        self.tree.root.leaf_value = data.target.mean()
        self.tree.root.depth = 0

        self._fit_node(self.tree.root, data, eps)
        return self

    def predict(self, data):
        # type: (Data) -> numpy.ndarray

        return self._predict(self.tree.root, data.data)

    def _predict(self, node, data):
        # type: (TreeNode, numpy.ndarray) -> numpy.ndarray

        if node.predicate is not None:
            right_mask = numpy.where(data[:, node.predicate.feature_id] > node.predicate.value)[0]
            left_mask = numpy.setdiff1d(numpy.arange(0, data.shape[0]), right_mask)

            res = numpy.ndarray((data.shape[0], ))
            res[left_mask] = self._predict(node.left, data[left_mask])
            res[right_mask] = self._predict(node.right, data[right_mask])
            return res
        else:
            return numpy.ones(data.shape[0]) * node.leaf_value

    def leafs(self, node=None):
        if node is None:
            return self.leafs(self.tree.root.right) + self.leafs(self.tree.root.left)
        else:
            if node.left is None and node.right is None:
                return [node]
            else:
                return self.leafs(node.left) + self.leafs(node.right)
