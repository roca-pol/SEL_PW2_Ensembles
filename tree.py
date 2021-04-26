import warnings
import numpy as np
import pandas as pd
import itertools

# try to load progress bars module to show a simple display
# with the current progress of the algorithm (OPTIONAL)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class Node:
    condition = None
    left = None
    right = None
    is_leaf = False
    label = None

    def next(self, x):
        return self.left if self.condition(x) else self.right


def binary_split(seq):
    v = len(seq)
    all_partitions = (itertools.combinations(seq, m) for m in range(1, v))
    all_partitions = itertools.chain.from_iterable(all_partitions)
    
    for m in range(2**(v-1) - 1):
        yield next(all_partitions)


def best_split(data: pd.DataFrame, attr, values, target):
    attr_col = data[attr]
    n = len(data)
    
    split_scores = []
    for subset in binary_split(values):
        mask = attr_col.isin(subset)
        l = data[mask]
        r = data[~mask]
        lscore, llabels = gini_index(l[target])
        rscore, rlabels = gini_index(r[target])
        score = (len(l) / n) * lscore + (len(r) / n) * rscore
        split_scores.append((subset, score, llabels, rlabels))

    return min(split_scores, key=lambda x: x[1])


def gini_index(data: pd.DataFrame):
    labels, counts = np.unique(data.values, return_counts=True)
    pr = counts / len(data)
    return 1 - np.square(pr).sum(), labels


class DecisionTree:

    def fit(self, data: pd.DataFrame, target='class'):
        # precompute attribute types because it seems to be slow
        attr_cols = data.drop(target, axis=1)
        self._categ_attrs = set(attr_cols.select_dtypes(exclude=['number']))
        self._num_attrs = set(attr_cols.select_dtypes(include=['number']))
        
        self.root_ = self._build_tree(data.copy(), target)
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

            data, attrs, vals = self._get_splittable_data(data, target)

            if len(attrs) == 0:
                node.is_leaf = True
                node.label = data[target].mode()[0]
                continue

            attrs = self._get_attr_subset(attrs)

            # binarize numerical attributes
            num_attrs = list(self._num_attrs.intersection(attrs))
            
            if len(num_attrs) > 0:
                data_categ, means = self._bin_numerical_attrs(data, num_attrs)
            else:
                data_categ = data
                
            # find the best splitting attribute
            attr, val_subset, llabels, rlabels = \
                self._find_best_split(data_categ, attrs, vals, target)

            if attr in num_attrs:
                mean = means[attr]
                node.condition = lambda x: x[attr] <= mean
            else:
                node.condition = lambda x: np.isin(x[attr], val_subset)

            if len(llabels) == 1:
                node.left = Node()
                node.left.is_leaf = True
                node.left.label = llabels[0]
            else:
                node.left = Node()
                lsplit = data[node.condition(data)]
                nodes_to_split.append((node.left, lsplit))

            if len(rlabels) == 1:
                node.right = Node()
                node.right.is_leaf = True
                node.right.label = rlabels[0]
            else:
                node.right = Node()
                rsplit = data[~node.condition(data)]
                nodes_to_split.append((node.right, rsplit))
            
        return root

    def _get_splittable_data(self, data, target):
        categ_attrs = self._categ_attrs.intersection(data.columns)
        num_attrs = self._num_attrs.intersection(data.columns)
        
        # include only attributes with multiple values
        vals = {}
        for attr in categ_attrs:
            unique_vals = pd.unique(data[attr])
            if len(unique_vals) > 1:
                vals[attr] = tuple(unique_vals)

        for attr in num_attrs:
            unique_vals = pd.unique(data[attr])
            if len(unique_vals) > 1:
                vals[attr] = (True, False)  # will be binarized

        useful_attrs = list(vals.keys())
        useful_cols = useful_attrs + [target]
        return data[useful_cols], useful_attrs, vals

    def _get_attr_subset(self, attrs):
        return attrs

    def _find_best_split(self, data, attrs, vals, target):
        splits = [(attr,) + best_split(data, attr, vals[attr], target) 
                  for attr in attrs]
        best_attr, val_subset, score, llabels, rlabels \
             = max(splits, key=lambda x: x[2])
        return best_attr, val_subset, llabels, rlabels

    def _bin_numerical_attrs(sefl, data, num_attrs):
        means = data[num_attrs].mean()
        data = data.copy()
        data[num_attrs] = data[num_attrs] <= means
        return data, means



class RandomTree(DecisionTree):
    def __init__(self, n_attr):
        super().__init__()
        self.n_attr = n_attr

    def _get_attr_subset(self, attrs):
        if len(attrs) <= self.n_attr:
            return attrs
        else:
            return np.random.choice(attrs, self.n_attr, replace=False)
        


        # data, attrs, vals = self._get_splittable_data(data, target)

        # if len(attrs) == 0:
        #     node = Node()
        #     node.is_leaf = True
        #     node.label = data[target].mode()[0]
        #     return node

        # attrs = self._get_attr_subset(attrs)

        # # binarize numerical attributes
        # num_attrs = data[attrs].select_dtypes(include=['number'])
        # if not num_attrs.empty:
        #     data_categ, means = self._bin_numerical_attrs(data, num_attrs)
        # else:
        #     data_categ = data
            
        # # find the best splitting attribute
        # attr, val_subset, llabels, rlabels = \
        #     self._find_best_split(data_categ, attrs, vals, target)

        # node = Node()
        # if attr in num_attrs.columns:
        #     mean = means[attr]
        #     node.condition = lambda x: x[attr] <= mean
        # else:
        #     node.condition = lambda x: np.isin(x[attr], val_subset)

        # if len(llabels) == 1:
        #     node.left = Node()
        #     node.left.is_leaf = True
        #     node.left.label = llabels[0]
        # else:
        #     left_split = data[node.condition(data)]
        #     node.left = self._build_tree(left_split, target)

        # if len(rlabels) == 1:
        #     node.right = Node()
        #     node.right.is_leaf = True
        #     node.right.label = rlabels[0]
        # else:
        #     right_split = data[~node.condition(data)]
        #     node.right = self._build_tree(right_split, target)
            
        # return node