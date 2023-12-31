#!/bin/python3

'''
Hyperparamters:
    - k:
        number of allowable neighbors
    - distance metric:
        method to quantify neighbor proximity
    - weights:
        assigned importance to neighbor based on similarity
'''

# > Priority:
# todo: figure out how to score each test case
# todo: work on notebook
# todo: figure out weighted knn
# todo: figure out how to plot performances (WCSS vs k?)

import time
from datetime import datetime
import numpy as np
from numba import jit, cuda

# note: added this to suppress warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from rich.progress import track
from common import *

ground_truth = 5

'''
    CONFIGURATION -------------------------------
'''

def configure_testcases(metrics, k_values):
    # create a list of all possible parameter combinations
    test_cases = []
    for metric in metrics:
        for k in k_values:
            test_case = {
                'parameters' :  {
                    'metric' : metric,
                    'k'      : k
                },
            }
            test_cases.append(test_case)

    return test_cases

@jit(target_backend='cuda')
def preprocess(data_set, preprocess_method):
    if preprocess_method == 'normalizing': #> pre-processing data | y = y / |y|
        data_set = (data_set - data_set.min()) / (data_set.max() - data_set.min())
    elif preprocess_method == 'logarithms': # > pre-processing data | y = log(x + 1)
        data_set = np.log1p(data_set)
    elif preprocess_method == 'clipping': #> pre-processing data | y < 10
        data_set = np.clip(data_set, a_min = None, a_max = 10)
    else:
        print(f'warning: selected method {preprocess_method} NOT implemented!')

    return data_set

'''
    KNN ALGORITHM -------------------------------
'''

@jit(target_backend='cuda')
def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(np.subtract(x, y))))

@jit(target_backend='cuda')
def cosine_distance(x, y):
    a = np.sum(np.multiply(x, y))
    b = np.multiply( np.sqrt(np.sum( np.square(x))), np.sqrt( np.sum( np.square(y))))
    return 1 - np.divide(a, b)

@jit(target_backend='cuda')
def jaccard_distance(x, y):
    a = np.sum(np.multiply(x, y))
    b = np.add(np.sum(np.square(x)), np.sum(np.square(y)))
    return np.divide(a, np.subtract(b, a))

@jit(target_backend='cuda')
def KNN(train_set, test_sample, metric, k):
    neighbor_distances = np.zeros(train_set['width'])
    nearest_neighbors  = np.zeros(train_set['width'], dtype = int)

    for i in train_set['columns']:
        if metric == 'euclidean':
            neighbor_distances[i] = euclidean_distance(train_set['data'][:,i], test_sample)
            nearest_index = np.argpartition(neighbor_distances, k)[:k] #> get the smallest
        if metric == 'cosine':
            neighbor_distances[i] = cosine_distance(train_set['data'][:,i], test_sample)
            nearest_index = np.argpartition(neighbor_distances, -k)[-k:] #> get the largest
        if metric == 'jaccard':
            neighbor_distances[i] = jaccard_distance(train_set['data'][:,i], test_sample)
            nearest_index = np.argpartition(neighbor_distances, -k)[-k:] #> get the largest

    for i in nearest_index:
        nearest_neighbors[i] = 1

    # note: these two prints are the same for arg sort and arg partition
    # print(f'arg partition: {neighbor_distances[nearest_neighbors == 1]}')
    # print(f'arg sort: {neighbor_distances[np.argsort(neighbor_distances)[-k:]]}')

    assert np.sum(nearest_neighbors) == k

    return nearest_neighbors

'''
    CROSS-VALIDATION ----------------------------
'''

# split will randomize the columns and split the trainset into five subsets.
def split(grid):
    grid = np.transpose(grid)
    np.random.shuffle(grid)
    grid = np.transpose(grid)
    res = [grid[:, 0:2966], grid[:, 2966:5932], grid[:, 5932: 8899],
            grid[:, 8899:11866], grid[:, 11866:14832]]

    return res

def produceTestSet(grid):
    test1 = np.concatenate((grid[0], grid[1], grid[2], grid[3]), 1)
    test2 = np.concatenate((grid[0], grid[1], grid[2], grid[4]), 1)
    test3 = np.concatenate((grid[0], grid[1], grid[3], grid[4]), 1)
    test4 = np.concatenate((grid[0], grid[2], grid[3], grid[4]), 1)
    test5 = np.concatenate((grid[1], grid[2], grid[3], grid[4]), 1)

    res = [test1, test2, test3, test4, test5]
    index = 1
    for test in res:
        with open(f'../test/{index}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(test)
        index += 1

def cross_validate(test_cases):
    raise NotImplementedError

'''
    TEST CASE RUN -------------------------------
'''

def run_testcase(test_case, train_set, test_set, knn_graph):

    print(f"\nrunning test case {test_case['parameters']['metric']} distance with {test_case['parameters']['k']} neighbors...")

    test_case['recommendation'] = np.zeros(test_set['data'].shape, dtype = int)

    #> for all test_set samples/columns, choose top 5 in knn_centroids that are also 0 in test_set sample/column
    for column in track(test_set['columns']):
        knn_graph[:,column] = KNN(train_set, test_set['data'][:,column], test_case['parameters']['metric'],test_case['parameters']['k'])
        knn_centroids = np.zeros(train_set['height'])

        #> find our index of all k 1's in `knn_graph` column
        knn_index = np.argwhere(knn_graph[:,column] == 1)

        #> find our centroids for all `train_set` rows
        for row in train_set['rows']:
            knn_centroids[row] = np.mean(train_set['data'][row, knn_index])

        #> find our index of zero-values in a `test_set` column
        test_sample_zero_index = np.where(test_set['data'][:,column] == 0)[0]

        #> find index of largest values in our centroids AND that are zero in the `test_set` column
        recommend_set_index = test_sample_zero_index[np.argpartition(-knn_centroids[test_sample_zero_index], ground_truth)[:ground_truth]]

        for i in recommend_set_index:
            test_case['recommendation'][i, column] = 1

        assert np.sum(test_case['recommendation'][:,column]) == ground_truth

    # todo: cross-validate and get our best testcase to then write to a csv file
    # todo: if we split and save our seperate files in folders `cv_train` `cv_test`, we can pull from these now...
    # test_case['score'] = cross_validate(test_case, cv_test_set, cv_test_)
    test_case['score'] = 1.0 # note: dummy value

    return test_case
