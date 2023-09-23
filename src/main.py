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

import time
from datetime import datetime
import numpy as np

from rich.progress import track
from common import *

# pre-process to normalize data to variance stable transformations

def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(np.subtract(x, y))))

def cosine_distance(x, y):
    a = np.sum(np.multiply(x, y))
    b = np.multiply( np.sqrt(np.sum( np.square(x))), np.sqrt( np.sum( np.square(y))))
    return 1 - np.divide(a, b)

def jaccard_distance(x, y):
    a = np.sum(np.multiply(x, y))
    b = np.add(np.sum(np.square(x)), np.sum(np.square(y)))
    return np.divide(a, np.subtract(b, a))

def absolute_difference(x, y):
    return np.subtract(x, y)

def KNN(train_set, test_sample, k):
    neighbor_distances = np.zeros(train_set['width'])
    nearest_neighbors  = np.zeros(train_set['width'], dtype = int)

    for column in train_set['columns']:
        neighbor_distances[column] = euclidean_distance(train_set['data'][:,column], test_sample)

    nearest_index = np.argpartition(neighbor_distances, k)[:k]

    # for i in range(len(neighbor_distances)):
        # nearest_neighbors[i] = 1 if neighbor_distances[i] in neighbor_distances[nearest_index] else 0
    for i in nearest_index:
        nearest_neighbors[i] = 1

    # print(f'nearest neighbors: {nearest_neighbors[nearest_index]}')
    # print(f'assert sum is k = {np.sum(nearest_neighbors)}')
    assert np.sum(nearest_neighbors) == k

    return nearest_neighbors

def main():
    print(f'readinng in our train and test sets...')
    train_set = read_input('../train/train_set.csv')
    test_set  = read_input('../test/test_set.csv')

    # todo: pre-process
    #> clipping our sets
    train_set['data'] = np.clip(train_set['data'], a_min = None, a_max = 10)
    test_set['data'] = np.clip(test_set['data'], a_min = None, a_max = 10)

    recommend_set = np.zeros(test_set['data'].shape, dtype = int)

    k = 20
    ground_truth = 5

    # exit_value = 1000 - 1
    exit_value = test_set['width'] - 1

    knn_graph     = np.zeros((train_set['width'], test_set['width']), dtype = int)
    # knn_index     = np.zeros((train_set['width'], k), dtype = int)
    # knn_centroids = np.zeros(train_set['height'])

    print(f'finding k nearest neighbors...') # note: 5 tc's costs 0.7856 seconds, 770 seconds for all tc's
    start = time.time()
    for column in track(test_set['columns']):
        knn_graph[:,column] = KNN(train_set, test_set['data'][:,column], k)
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
            recommend_set[i, column] = 1

        assert np.sum(recommend_set[:,column]) == ground_truth

        if column == exit_value:
            break

    end = time.time()
    print(f'timelapse: {time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(end - start))}')

    # todo: cross-validate and get our best testcase to then write to a csv file

    #> write to our .csv file
    write_output(f'../out/final/recommend_set_{datetime.now().strftime("%m_%d_%H_%M_%S")}.csv', recommend_set)

if __name__ == '__main__':
    main()
else:
    print(f'cannot be imported, run as script!')