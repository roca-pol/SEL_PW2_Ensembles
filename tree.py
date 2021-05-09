import numpy as np
from numpy.random import RandomState
import pandas as pd
import itertools


def binary_split_categ(seq):
    v = len(seq)
    all_partitions = (itertools.combinations(seq, m) for m in range(1, v))
    all_partitions = itertools.chain.from_iterable(all_partitions)
    
    for m in range(2**(v-1)-1):
        yield next(all_partitions)

def binary_split_num(seq):
    seq = np.sort(seq)

    for i in range(len(seq)-1):
        if seq[i] != seq[i+1]:
            yield (seq[i] + seq[i+1]) / 2.


def best_split(data: pd.DataFrame, attr, target, is_numeric=False):
    attr_col = data[attr].values
    targ_col = data[target].values
    n = len(data)
    
    split_scores = []
    # add worst score in case we cannot split
    split_scores.append((None, np.inf, None, None))

    if is_numeric:
        for midpoint in binary_split_num(attr_col):
            l = targ_col[attr_col <= midpoint]
            r = targ_col[attr_col > midpoint]
            lscore, llabels = gini_index(l)
            rscore, rlabels = gini_index(r)
            score = (len(l) / n) * lscore + (len(r) / n) * rscore
            split_scores.append((midpoint, score, llabels, rlabels))

    else:
        values = np.unique(attr_col)
        for subset in binary_split_categ(values):
            mask = np.isin(attr_col, subset)
            l = targ_col[mask]
            r = targ_col[~mask]
            lscore, llabels = gini_index(l)
            rscore, rlabels = gini_index(r)
            score = (len(l) / n) * lscore + (len(r) / n) * rscore
            split_scores.append((subset, score, llabels, rlabels))

    return min(split_scores, key=lambda x: x[1])


def gini_index(data):
    labels, counts = np.unique(data, return_counts=True)
    pr = counts / len(data)
    return 1. - np.square(pr).sum(), labels


class Node:
    left = None
    right = None
    is_leaf = False
    label = None
    attr = None
    val = None
    is_numeric = False

    def next(self, x):
        return self.left if self.condition(x) else self.right

    def condition(self, x):
        if self.is_numeric:
            return x[self.attr] <= self.val
        else:
            return np.isin(x[self.attr], self.val)


class DecisionTree:
    def fit(self, data: pd.DataFrame, target='class'):
        # precompute attribute types because it seems to be slow
        attr_cols = data.drop(target, axis=1)
        self._num_attrs = set(attr_cols.select_dtypes(include=['number']))

        self.feature_importance_ = pd.Series(index=attr_cols.columns, 
                                             data=0.)
        self.root_ = self._build_tree(data.copy(), target)
        self.n_nodes_ = int(self.feature_importance_.sum())
        self.feature_importance_ /= self.n_nodes_
        return self

    def predict(self, data: pd.DataFrame):
        labels = []
        for _, ins in data.iterrows():
            node = self.root_
            while not node.is_leaf:
                node = node.next(ins)
            labels.append(node.label)

        return np.array(labels)

    def _build_tree(self, data, target):
        root = Node()
        nodes_to_split = [(root, data)]

        while len(nodes_to_split) > 0:
            node, data = nodes_to_split.pop()
            attrs = self._get_attr_subset(data.columns.drop(target))

            if len(attrs) == 0:
                node.is_leaf = True
                node.label = data[target].mode()[0]
                continue
                            
            # find the best splitting attribute
            attr, val, llabels, rlabels = \
                self._find_best_split(data, attrs, target)

            # in case all attrs had just 1 value (no split)
            if attr is None:
                # try again without considering any of these attrs
                cols = data.columns.difference(attrs)
                nodes_to_split.append((node, data[cols]))
                continue

            self.feature_importance_[attr] += 1

            node.is_numeric = attr in self._num_attrs
            node.attr = attr
            node.val = val

            # left
            if len(llabels) == 1:
                node.left = Node()
                node.left.is_leaf = True
                node.left.label = llabels[0]
            else:
                node.left = Node()
                lsplit = data[node.condition(data)]
                nodes_to_split.append((node.left, lsplit))

            # right
            if len(rlabels) == 1:
                node.right = Node()
                node.right.is_leaf = True
                node.right.label = rlabels[0]
            else:
                node.right = Node()
                rsplit = data[~node.condition(data)]
                nodes_to_split.append((node.right, rsplit))
            
        return root

    def _get_attr_subset(self, attrs):
        return attrs

    def _find_best_split(self, data, attrs, target):
        best_result = (None, None, None, None)
        best_score = np.inf

        for attr in attrs:
            is_numeric = attr in self._num_attrs
            val, score, llabels, rlabels = \
                best_split(data, attr, target, is_numeric) 

            if score < best_score:
                best_result = attr, val, llabels, rlabels
                best_score = score

        return best_result


class RandomTree(DecisionTree):
    def __init__(self, n_attr, random_state=None):
        super().__init__()
        self.n_attr = n_attr
        if isinstance(random_state, RandomState):
            self.random = random_state
        else:
            self.random = RandomState(seed=random_state)

    def _get_attr_subset(self, attrs):
        if len(attrs) <= self.n_attr:
            return attrs
        else:
            return self.random.choice(attrs, self.n_attr, replace=False)
        