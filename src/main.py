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

import numpy as np
from numba import jit, cuda

from rich.progress import track
from common import *

# note: added this to suppress numba deprecation warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#> cross-validation settings
cross_validation_data_sets_exists = True
use_cross_validation_data_sets    = False

#> recommend 5 birds that were likely spotted but not tallied
ground_truth = 5

'''
    CONFIGURATION -------------------------------
'''

'''
create a list of all possible parameter combinations
:param
    metrics = type of metric: euclidean, cosine, or jaccard
    k_values = # of k
'''
def configure_testcases(metrics, k_values):
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

# @jit(target_backend='cuda') # note: turned off due to non-optimal solutions
def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(np.subtract(x, y))))

@jit(target_backend='cuda')
def cosine_distance(x, y):
    a = np.sum(np.multiply(x, y))
    b = np.multiply( np.sqrt(np.sum( np.square(x))), np.sqrt( np.sum( np.square(y))))
    return 1 - np.divide(a, b)

# @jit(target_backend='cuda') # note: turned off due to non-optimal solutions
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
        if metric == 'cosine':
            neighbor_distances[i] = cosine_distance(train_set['data'][:,i], test_sample)
        if metric == 'jaccard':
            neighbor_distances[i] = jaccard_distance(train_set['data'][:,i], test_sample)

    nearest_index = np.argpartition(neighbor_distances, k)[:k]

    for i in nearest_index:
        nearest_neighbors[i] = 1

    assert np.sum(nearest_neighbors) == k

    return nearest_neighbors


'''
    CROSS-VALIDATION ----------------------------
    this section was used to for cross-validation of our hyper parameters
    using k-fold methods
'''

'''
split will randomize the columns and split the train_set into five subsets.
:param
    grid = 2D matrix (train_set to be test_set)
'''
def split(grid):
    grid = np.transpose(grid)
    np.random.shuffle(grid)
    grid = np.transpose(grid)
    res = [grid[:, 0:2966], grid[:, 2966:5932], grid[:, 5932: 8899],
            grid[:, 8899:11866], grid[:, 11866:14832]]

    return res

'''
This function was used to create tests for cross validation
:param
    grid = array of matrix (subsets of shuffled train_set)
'''
def produceTestSet(grid):
    test1 = np.concatenate((grid[0], grid[1], grid[2], grid[3]), 1)
    test2 = np.concatenate((grid[0], grid[1], grid[2], grid[4]), 1)
    test3 = np.concatenate((grid[0], grid[1], grid[3], grid[4]), 1)
    test4 = np.concatenate((grid[0], grid[2], grid[3], grid[4]), 1)
    test5 = np.concatenate((grid[1], grid[2], grid[3], grid[4]), 1)

    res = [test1, test2, test3, test4, test5]
    index = 1
    for test in res:
        with open(f'../train/cv_train/train_cvset_{index}.csv', mode='w', newline='') as train:
            writer = csv.writer(train)
            writer.writerows(test)

        with open(f'../test/cv_test/test_cvset_{index}.csv', mode='w', newline='') as test:
            writer = csv.writer(test)
            writer.writerows(grid[index - 1])
        index += 1

'''
    TEST CASE RUN -------------------------------
    main driver of the knn methods
:param
    test_case: metric & k
    train_set: matrix
    test_set:  matrix
    non_changed_test_set: matrix = this is the unchanged test set, that we'll use to evaluate our hyper parameters
    knn_graph: matrix of zeros at start
'''
def run_testcase(test_case, train_set, test_set, non_changed_test_set, knn_graph):

    print(f"\nrunning test case {test_case['parameters']['metric']} distance with {test_case['parameters']['k']} neighbors...")

    test_case['recommendation'] = np.zeros(test_set['data'].shape, dtype = int)

    knn_index = np.zeros((train_set['width'], test_case['parameters']['k']), dtype=int)
    knn_centroids = np.zeros(train_set['height'])
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

    #> cross-validate our test sets
    if use_cross_validation_data_sets:
        test_case['score'] = getScore(test_case['recommendation'], non_changed_test_set['data'])
        print(f'test case score: {test_case["score"]}')

    return test_case

'''
    this was used to create test batch for cross-validation
'''
def createTests():

    for i in range(1, 6):
        test = read_input(f'../test/cv_test/test_cvset_{i}.csv')
        test = randomize(test)

        with open(f'../test/rand_test/test_randset_{i}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(test['data'])

def main():

    #> tuning hyper-parameter settings
    all_preprocess_methods = ['normalizing', 'logarithms', 'clipping']
    selected_preprocess_method = all_preprocess_methods[1]
    metrics  = ['euclidean', 'cosine', 'jaccard']
    metrics  = ['cosine']
    k_values = [20, 21, 22, 23, 24, 25, 26, 27]
    k_values = [25]

    if cross_validation_data_sets_exists:
        if use_cross_validation_data_sets:
            print(f'reading in our cross-validation train and test sets...')
            numberOfSets = 3
            map = readInFiles(numberOfSets)
            assert 0 < numberOfSets and numberOfSets < 5
        else:
            print(f'reading in our train and test sets...')
            train_set = read_input('../train/train_set.csv', trim_header = True)
            test_set  = read_input('../test/test_set.csv', trim_header = True)
            non_changed_test_set = read_input('../test/cv_test/test_cvset_2.csv')
            numberOfSets = 1
            assert 0 < numberOfSets and numberOfSets == 1
    else:
        train_set = read_input('../train/train_set.csv', trim_header = True)
        test_set  = read_input('../test/test_set.csv', trim_header = True)
        shuffled_set = split(train_set['data'])
        produceTestSet(shuffled_set)

    for ind in range(numberOfSets):

        if use_cross_validation_data_sets:
            print(f'\ntesting cross validation set {ind + 1} of {numberOfSets}...')
            train_set = map['train'][ind]
            test_set  = map['test'][ind]
            non_changed_test_set = map['answer'][ind]

        knn_graph = np.zeros((train_set['width'], test_set['width']), dtype = int)

        print(f'pre-processing our data to use {selected_preprocess_method} on our train and test sets...')
        train_set['data'] = preprocess(train_set['data'], selected_preprocess_method)
        test_set['data']  = preprocess(test_set['data'],  selected_preprocess_method)

        #> create a list of all test cases to keep track of optimized parameters
        test_cases = configure_testcases(metrics, k_values)
        all_test_cases = range(len(test_cases))

        #> run each test case in our list and
        print(f'running all test cases with unique hyper-parameters...\n')
        for i in all_test_cases:
            test_cases[i] = run_testcase(test_cases[i], train_set, test_set, non_changed_test_set, knn_graph)

        best_score_index = 0

        #> write our optimal result to a .csv file for Kaggle submission
        file_name = f"k{test_cases[best_score_index]['parameters']['k']}_{test_cases[best_score_index]['parameters']['metric']}_{selected_preprocess_method}.csv"
        write_output(f'../out/final/{file_name}', test_cases[best_score_index]['recommendation'])

if __name__ == '__main__':
    main()
else:
    print(f'cannot be imported, run as script!')