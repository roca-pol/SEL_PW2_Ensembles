import numpy as np
from numpy.random import RandomState
import pandas as pd
import itertools


def binary_split_categ(seq):
    """
    Given a sequence of elements, computes all the possible
    binary partitions of it but yields only one of the subsets
    (without the complementary).
    """
    v = len(seq)
    all_partitions = (itertools.combinations(seq, m) for m in range(1, v))
    all_partitions = itertools.chain.from_iterable(all_partitions)
    
    for m in range(2**(v-1)-1):
        yield next(all_partitions)


def binary_split_num(seq):
    """
    Given a sequence of numbers yields all midpoints
    between every pair of sorted values.
    """
    seq = np.sort(seq)
    for i in range(len(seq)-1):
        if seq[i] != seq[i+1]:
            yield (seq[i] + seq[i+1]) / 2.


def best_split(data: pd.DataFrame, attr, target, is_numeric=False):
    """
    Given a DataFrame and a specific attribute,
    computes the best possible way to split data
    minimizing the Gini impurity index. Also returns
    the labels left on each of the two splits.

    Parameters
    ----------
    data : DataFrame
        Supervised DataFrame of instances to split 
        with features and a target class label.

    attr : str
        Name of the attribute to perform the split.

    target : str
        Name of the column of the class label.

    is_numeric : bool, default=False
        If True then it means attr is a numeric 
        attribute/feature.

    Returns
    -------
    midpoint or subset : float, int, set
        Value of the point or set of categorical values
        to perform the split on the attribute.

    score : float
        Gini impurity index of the partition (weighted
        average).

    llabels : any
        Class labels left on one of the partitions (left).

    rlabels : any
        Class labels left on the other partition (right).
    """
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
    """
    Computes the Gini impurity index of a given structure
    of data. Also returns the unique values found in it.

    Parameters
    ----------
    data : ndarray, DataFrame, list
        A single column/row structure containing the labels
        of some instances.

    Returns
    -------
    index : float
        The Gini impurity index

    labels : any
        The unique values found in data.
    """
    labels, counts = np.unique(data, return_counts=True)
    pr = counts / len(data)
    return 1. - np.square(pr).sum(), labels


class Node:
    """
    This class represents a binary node in a DecisionTree
    built with the CART algorithm.
    """
    left = None
    right = None
    is_leaf = False
    label = None
    attr = None
    val = None
    is_numeric = False

    def next(self, x):
        """
        Used to travese a tree. Pass an instance to obtain
        the following node depending on the features.

        Parameters
        ----------
        x : any
            An instance represented in attribue-value pairs.

        Returns
        -------
        next : Node
            The node on the left if x satisfies the splitting 
            condition, otherwise the node on the right.
        """
        return self.left if self.condition(x) else self.right

    def condition(self, x):
        if self.is_numeric:
            return x[self.attr] <= self.val
        else:
            return np.isin(x[self.attr], self.val)


class DecisionTree:
    """
    This class represents a decision tree classifier induced
    from a given a supervised dataset following the CART algorithm.
    It implements the sklearn's Estimator API.
    """
    def fit(self, data: pd.DataFrame, target='class'):
        """
        Induce a decision tree classifier from the given dataset
        by the CART algorithm.

        Parameters
        ----------
        data : DataFrame
            A supervised dataset.

        target : str, default='class'
            Name of the attribute containing class labels.

        Attributes
        ----------
        root_ : Node
            Root node of the tree, from which it can be traversed.

        n_nodes_ : int
            Total amount of nodes of the induced tree.

        feature_importance_ : Series
            A Series with attribues and their normalized importance
            (frequency) computed from the amount of times they were
            chosen to perform a split.
        """
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
        """
        Use the tree to classify the given instances.

        Parameters
        ----------
        data : DataFrame
            Instances to classify. Must have the same attributes
            as the data used to induce the tree.

        Returns
        -------
        labels : ndarray
            The predicted class labels for the provided instances.
        """
        labels = []
        for _, ins in data.iterrows():
            node = self.root_
            while not node.is_leaf:
                node = node.next(ins)
            labels.append(node.label)

        return np.array(labels)

    def _build_tree(self, data, target):
        """
        Internal method to induce the tree. The algorithm followed
        is CART. The tree is constructed iteratively (not recursive).
        """
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

            # in the extreme case where all attrs
            # had just 1 value (no possible split)
            if attr is None:
                # try again without considering any of these attrs
                others = data.columns.difference(attrs)
                nodes_to_split.append((node, data[others]))
                continue

            self.feature_importance_[attr] += 1

            node.is_numeric = attr in self._num_attrs
            node.attr = attr
            node.val = val

            # left
            node.left = Node()
            if len(llabels) == 1:
                node.left.is_leaf = True
                node.left.label = llabels[0]
            else:
                lsplit = data[node.condition(data)]
                nodes_to_split.append((node.left, lsplit))

            # right
            node.right = Node()
            if len(rlabels) == 1:
                node.right.is_leaf = True
                node.right.label = rlabels[0]
            else:
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
    """
    This class represents a random decision tree classifier induced
    from a given a supervised dataset following the CART algorithm.
    It implements the sklearn's Estimator API.

    See `DecisionTree` class of the same module.

    Parameters
    ----------
    n_attr : int
        Number of attributes to subsample at each node when evaluating
        which one to use to split.

    random_state : int, RandomState, default=None
        Seed or random number generator to be used. By default a new
        system seed will be generated each time (done by numpy).
    """
    def __init__(self, n_attr, random_state=None):
        super().__init__()
        self.n_attr = n_attr
        if isinstance(random_state, RandomState):
            self.random = random_state
        else:
            self.random = RandomState(seed=random_state)

    def _get_attr_subset(self, attrs):
        """
        This internal method makes the difference between a standard
        decision tree and a random decision tree. In this case, a 
        random subset of attributes is returned.
        """
        if len(attrs) <= self.n_attr:
            return attrs
        else:
            return self.random.choice(attrs, self.n_attr, replace=False)
        