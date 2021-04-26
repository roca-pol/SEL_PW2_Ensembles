import os
import time
import numpy as np
import pandas as pd

from tree import DecisionTree, RandomTree

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
@click.option('-s', '--seed', type=int, default=101, show_default=True, help='Random seed used when'
              ' shuffling data before splitting it into train and test sets.')
def main(dataset, seed):
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

    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)
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

    # induce rules
    clf = DecisionTree() # RandomTree(n_attr=5) # 
    t0 = time.time()
    clf.fit(df_train, target=target)
    t1 = time.time()

    print(f'Compute time: {round(t1 - t0, 2)}s')

    # apply rules
    y_pred = clf.predict(df_test)
    y_true = df_test[target]
    print('Accuracy:', round(accuracy_score(y_true, y_pred), 4))


if __name__ == '__main__':
    main()