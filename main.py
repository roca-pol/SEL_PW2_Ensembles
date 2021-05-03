import os
import time
import numpy as np
from numpy.random import RandomState
import pandas as pd

from tree import DecisionTree, RandomTree
from ensemble import DecisionForest, RandomForest

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import click
from click import Choice


def load_hepatitis():
    path = os.path.join('data', 'hepatitis.csv')
    df = pd.read_csv(path)
    return df


def load_cmc():
    path = os.path.join('data', 'cmc.csv')
    df = pd.read_csv(path)

    df = df.astype(str)
    num_attr = ['Wifes_age', 'Number_of_children_ever_born']
    df[num_attr] = df[num_attr].astype(int)
    return df


def load_nursery():
    path = os.path.join('data', 'nursery.csv')
    df = pd.read_csv(path)
    return df


@click.command()
@click.option('-d', '--dataset', type=Choice(['hepatitis', 'cmc', 'nursery']), required=True,
              help='Dataset name.')
@click.option('-c', '--classifier-type', type=Choice(['decision', 'random']), required=True,
              help='Type of classifier to train: *decision* forest or *random* forest.')
@click.option('-t', '--n-trees', default=1, show_default=True, help='Number of trees'
              ' (weak classifiers) for the forest (ensemble).')
@click.option('-f', '--n-features', type=int, default=None, help='Size of the'
              ' subset of features that each tree or node will use. A value of 0 will'
              ' make this number to be drawn from an uniform distribution for each tree'
              ' (only works with decision forests). By default all features will be used.')
@click.option('-s', '--seed', type=int, default=None, help='Random seed used'
              ' for sampling, shuffling, or any op involving randomness.')
def main(dataset, classifier_type, n_trees, n_features, seed):
    # load the corresponding dataset into a dataframe
    if dataset == 'hepatitis':
        df = load_hepatitis()
        target = 'class'

    elif dataset == 'cmc':
        df = load_cmc()
        target = 'Contraceptive_method_used'

    elif dataset == 'nursery':
        df = load_nursery()
        target = 'class'

    random_state = RandomState(seed=seed)

    if n_features is None:
        n_features = len(df.columns) - 1

    if classifier_type == 'decision':
        n_features = 'unif' if n_features == 0 else n_features
        clf = DecisionForest(n_trees=n_trees, n_feat=n_features, 
                             random_state=random_state)
    else:
        if n_features < 1:
            raise ValueError('Option "--n-features" must be higher than 0'
                             ' for a random forest classifier.')
        clf = RandomForest(n_trees=n_trees, n_feat=n_features, 
                           random_state=random_state)
        
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[target],
                                         shuffle=True, random_state=random_state)
    df_train = df_train.copy()
    df_test = df_test.copy()

    # preprocessing - fill missing values
    num_attr = df.select_dtypes(include=['number']).columns
    means = df_train[num_attr].mean()
    df_train.fillna(means, inplace=True)
    df_test.fillna(means, inplace=True)

    categ_attr = df.select_dtypes(exclude=['number']).columns
    df_train.replace('?', np.nan, inplace=True)
    df_test.replace('?', np.nan, inplace=True)
    modes = df_train[categ_attr].mode().iloc[0]
    df_train.fillna(modes, inplace=True)
    df_test.fillna(modes, inplace=True)

    # induce forest
    t0 = time.time()
    clf.fit(df_train, target=target)
    t1 = time.time()
    print(f'Training time: {round(t1 - t0, 2)}s')
    print('\nFeature importance:')
    print(clf.feature_importance_.sort_values().to_string())
    print(f'Total: {clf.feature_importance_.sum()}\n')

    # classify using forest
    t0 = time.time()
    y_pred = clf.predict(df_test)
    t1 = time.time()
    print(f'Inference time: {round(t1 - t0, 2)}s')

    y_true = df_test[target]
    print('\nAccuracy:', round(accuracy_score(y_true, y_pred), 4))


if __name__ == '__main__':
    main()