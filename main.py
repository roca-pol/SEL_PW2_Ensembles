import os
import json
import time
import click
from click import Choice
import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ensemble import DecisionForest, RandomForest


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


@click.command(help='Train and test a decision or random forest classifier with the'
               ' specified parameters on a given dataset (80/20, train/test).')
@click.option('-d', '--dataset', type=Choice(['hepatitis', 'cmc', 'nursery']), required=True,
              help='Dataset name.')
@click.option('-c', '--classifier-type', type=Choice(['decision', 'random']), required=True,
              help='Type of classifier to train: *decision* forest or *random* forest.')
@click.option('-t', '--n-trees', default=1, show_default=True, help='Number of trees'
              ' (weak classifiers) for the forest (ensemble).')
@click.option('-f', '--n-features', type=int, default=None, help='Size of the'
              ' subset of features that each tree or node will use. A value of 0 will'
              ' make this number be drawn from a uniform distribution for each tree'
              ' (only works with decision forests). By default all features are used.')
@click.option('-s', '--seed', type=int, default=None, help='Random seed used'
              ' for sampling, shuffling, or any op involving randomness.')
@click.option('-o', '--out', type=str, default=None, help='File to save output to as JSON.')
def main(dataset, classifier_type, n_trees, n_features, seed, out):
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

    if n_trees < 1:
        raise ValueError('Option "--n-trees" must be higher than 0.')

    if n_features is None:
        n_features = len(df.columns) - 1
    elif n_features < 0:
        raise ValueError('Option "--n-features" must be higher than 1'
                         ' (or 0 only if it is a decision forest).')

    random_state = RandomState(seed=seed)

    if classifier_type == 'decision':
        n_features = 'unif' if n_features == 0 else n_features
        clf = DecisionForest(n_trees=n_trees, n_feat=n_features, 
                             random_state=random_state)
    else:
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
    train_time = time.time() - t0
    print(f'Training time: {round(train_time, 2)}s')
    print('\nFeature importance:')
    print(clf.feature_importance_.sort_values().round(4).to_string())
    print('Nodes:', clf.n_nodes_, '\n')

    # classify using forest
    t0 = time.time()
    y_pred = clf.predict(df_test)
    infer_time = time.time() - t0
    print(f'Inference time: {round(infer_time, 2)}s')

    y_true = df_test[target]
    accuracy = accuracy_score(y_true, y_pred)
    print('\nAccuracy:', round(accuracy, 4))

    # append result to output JSON file
    if out is not None:
        result = {
            'dataset': dataset,
            'classifier_type': classifier_type,
            'n_trees': n_trees,
            'n_features':n_features,
            'train_time': train_time,
            'infer_time': infer_time,
            'n_nodes': clf.n_nodes_,
            'accuracy': accuracy,
            'feature_importance': clf.feature_importance_.to_dict()
        }
        with open(out, 'a') as f:
            f.write(json.dumps(result, indent=4) + ',\n')


if __name__ == '__main__':
    main()