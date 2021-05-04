# Random forests and Decision forests induced with CART

## How to run

See options:
```console
$ python3 main.py --help
Usage: main.py [OPTIONS]

  Train and test a decision or random forest classifier with the specified
  parameters on a given dataset (80/20, train/test).

Options:
  -d, --dataset [hepatitis|cmc|nursery]
                                  Dataset name.  [required]
  -c, --classifier-type [decision|random]
                                  Type of classifier to train: *decision*
                                  forest or *random* forest.  [required]
  -t, --n-trees INTEGER           Number of trees (weak classifiers) for the
                                  forest (ensemble).  [default: 1]
  -f, --n-features INTEGER        Size of the subset of features that each
                                  tree or node will use. A value of 0 will
                                  make this number be drawn from a uniform
                                  distribution for each tree (only works with
                                  decision forests). By default all features
                                  are used.
  -s, --seed INTEGER              Random seed used for sampling, shuffling, or
                                  any op involving randomness.
  -o, --out TEXT                  File to save output to as JSON.
  --help                          Show this message and exit.
```

Train and test a classifier forest on a dataset:
```console
$ python3 main.py -d DATASET_NAME -c CLASSIFIER -t N_TREES -f N_FEATURES
$ python3 main.py -d nursery -c random -t 5 -f 6
$ python3 main.py -d hepatitis -c decision -t 100
$ python3 main.py -d cmc -c decision -t 2 -f 0
```
This command splits the dataset into 2 (train and test sets), then induces the forest on the train set, and finally evaluates the accuracy on the test set.

\
\
\
Datasets credits:\
DuBois, Christopher L. & Smyth, P. (2008). [UCI Network Data Repository](http://networkdata.ics.uci.edu). Irvine, CA: University of California, School of Information and Computer Sciences.