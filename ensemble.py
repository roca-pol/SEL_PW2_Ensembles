import numpy as np
from numpy.random import RandomState
import pandas as pd
from tree import DecisionTree, RandomTree

# try to load progress bars module to show a simple display
# with the current progress of the algorithm (OPTIONAL)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def mode(x):
    uniques, counts = np.unique(x, return_counts=True)
    return uniques[np.argmax(counts)]


class Forest:
    def __init__(self, n_trees, n_feat, random_state=None):
        self.n_trees = n_trees
        self.n_feat = n_feat
        if isinstance(random_state, RandomState):
            self.random = random_state
        else:
            self.random = RandomState(seed=random_state)
        
    def predict(self, data: pd.DataFrame):
        res = [tree.predict(data) for tree in tqdm(self.trees_, desc='Inference')]
        return np.apply_along_axis(mode, 0, np.array(res))


class RandomForest(Forest):
    def fit(self, data: pd.DataFrame, target='class'):
        trees = []
        f_imp = pd.Series(index=data.columns.drop(target),
                          dtype=int)
        for _ in tqdm(range(self.n_trees), desc='Training'):
            # generate bootstrapped dataset (indices)
            bootstrap_idx = self.random.choice(len(data), len(data), replace=True)

            # train tree
            tree = RandomTree(self.n_feat, random_state=self.random)
            tree.fit(data.iloc[bootstrap_idx], target)
            trees.append(tree)
            
            # accumulate feature importances
            for attr, count in tree.feature_importance_.iteritems():
                f_imp[attr] += count

        self.trees_ = trees
        self.feature_importance_ = f_imp
        return self


class DecisionForest(Forest):
    def fit(self, data: pd.DataFrame, target='class'):
        trees = []
        attrs = data.columns.drop(target)
        f_imp = pd.Series(index=attrs, dtype=int)
        
        # prepare number of features for each tree
        if self.n_feat == 'unif':
            n_feat_seq = self.random.randint(1, len(attrs)+1, self.n_trees)
        else:
            n_feat_seq = [self.n_feat] * self.n_trees

        for n_feat in tqdm(n_feat_seq, desc='Training'):
            # extract subset of features
            feat_subset = self.random.choice(attrs, n_feat, replace=False)
            feat_subset = list(feat_subset) + [target]

            # train tree
            tree = DecisionTree()
            tree.fit(data[feat_subset], target)
            trees.append(tree)

            # accumulate feature importances
            for attr, count in tree.feature_importance_.iteritems():
                f_imp[attr] += count

        self.trees_ = trees
        self.feature_importance_ = f_imp
        return self

