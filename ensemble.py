import numpy as np
from numpy.random import RandomState
import pandas as pd
from tree import DecisionTree, RandomTree

# try to load progress bars module to show a simple display
# with the current progress of the algorithm (OPTIONAL)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # does nothing


def mode(x):
    uniques, counts = np.unique(x, return_counts=True)
    return uniques[np.argmax(counts)]


class Forest:
    """
    This base (abstract) class represents a classifier forest.
    It implements the Estimator API from sklearn.
    """
    def __init__(self, n_trees, n_feat, random_state=None):
        self.n_trees = n_trees
        self.n_feat = n_feat
        if isinstance(random_state, RandomState):
            self.random = random_state
        else:
            self.random = RandomState(seed=random_state)
        
    def fit(self, *args, **kwargs):
        raise NotImplementedError('This method must be overwritten by'
            'the implemented forest (decision or random).')

    def predict(self, data: pd.DataFrame):
        """
        Use the forest to classify the given instances. Classification
        is performed with a voting scheme, returning the majority vote for
        each instance.

        Parameters
        ----------
        data : DataFrame
            Instances to classify. Must have the same attributes
            as the data used to induce the forest.

        Returns
        -------
        labels : ndarray
            The predicted class labels for the provided instances.
        """
        res = [tree.predict(data) for tree in tqdm(self.trees_, desc='Inference')]
        return np.apply_along_axis(mode, 0, np.array(res))


class DecisionForest(Forest):
    """
    This class represents a decision forest classifier.
    See `tree.DecisionTree`
    
    Parameters
    ----------
    n_trees : int
        Number of decision trees to induce.

    n_feat : int, 'unif'
        Size of the subset of features that each tree will use.
        If 'unif' then this value will be generated randomly per
        tree between 1 and the total number of features.

    random_state : int, RandomState, default=None
        Seed or random number generator to be used. By default a new
        system seed will be generated each time (done by numpy).
    """
    def fit(self, data: pd.DataFrame, target='class'):
        """
        Induce the decision trees of the forest from the given dataset.
        Each tree will receive a random subset of n_feat features.

        Parameters
        ----------
        data : DataFrame
            A supervised dataset.

        target : str, default='class'
            Name of the attribute containing class labels.

        Attributes
        ----------
        trees_ : list
            List of DecisionTree's induced.

        n_nodes_ : int
            Total amount of nodes of among all induced trees.

        feature_importance_ : Series
            A Series with attribues and their normalized importance
            (frequency) computed from the amount of times they were
            chosen to perform a split.
        """
        trees = []
        n_nodes = 0
        attrs = data.columns.drop(target)
        f_imp = pd.Series(index=attrs, data=0.)
        
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
            n_nodes += tree.n_nodes_

            # accumulate feature importances
            for attr, freq in tree.feature_importance_.iteritems():
                f_imp[attr] += freq * tree.n_nodes_

        self.trees_ = trees
        self.n_nodes_ = n_nodes
        self.feature_importance_ = f_imp / n_nodes
        return self


class RandomForest(Forest):
    """
    This class represents a random decision forest classifier.
    See `tree.RandomTree`

    Parameters
    ----------
    n_trees : int
        Number of random trees to induce.

    n_feat : int
        Size of the subset of features that each tree will use.

    random_state : int, RandomState, default=None
        Seed or random number generator to be used. By default a new
        system seed will be generated each time (done by numpy).
    """
    def fit(self, data: pd.DataFrame, target='class'):
        """
        Induce the random trees of the forest from the given dataset.
        Each tree will receive a bootstrapped version of the dataset.

        Parameters
        ----------
        data : DataFrame
            A supervised dataset.

        target : str, default='class'
            Name of the attribute containing class labels.

        Attributes
        ----------
        trees_ : list
            List of RandomTree's induced.

        n_nodes_ : int
            Total amount of nodes of among all induced trees.

        feature_importance_ : Series
            A Series with attribues and their normalized importance
            (frequency) computed from the amount of times they were
            chosen to perform a split.
        """
        trees = []
        n_nodes = 0
        f_imp = pd.Series(index=data.columns.drop(target), data=0.)

        for _ in tqdm(range(self.n_trees), desc='Training'):
            # generate bootstrapped dataset (indices)
            bootstrap_idx = self.random.choice(len(data), len(data), replace=True)

            # train tree
            tree = RandomTree(self.n_feat, random_state=self.random)
            tree.fit(data.iloc[bootstrap_idx], target)
            trees.append(tree)
            n_nodes += tree.n_nodes_
            
            # accumulate feature importances
            for attr, freq in tree.feature_importance_.iteritems():
                f_imp[attr] += freq * tree.n_nodes_

        self.trees_ = trees
        self.n_nodes_ = n_nodes
        self.feature_importance_ = f_imp / n_nodes
        return self

