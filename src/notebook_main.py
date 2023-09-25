#!/bin/python3

'''
    NOTE: TO MAKE IT EASIER TO CONVERT FROM A SCRIPT WITH MULTIPLE FUNCTIONS,
        WE SPLIT OUR MAIN AND NON-MAIN FUNCTIONS TO TWO FILES: MAIN AND HELPER.
        NOW, WE JUST CALL `from notebook_helper import *` to use all the functions
'''


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
from notebook_helper import *

def main():

    #> hyper-parameter lists
    all_preprocess_methods = ['normalizing', 'logarithms', 'clipping']
    selected_preprocess_method = all_preprocess_methods[0]

    # metrics  = ['euclidean', 'cosine', 'jaccard']
    # k_values = [20, 40, 60, 80, 100]
    metrics  = ['euclidean']
    k_values = [20] # scored a 0.11010

    print(f'\nreadinng in our train and test sets...\n')
    train_set = read_input('../train/train_set.csv')
    test_set  = read_input('../test/test_set.csv')
    knn_graph = np.zeros((train_set['width'], test_set['width']), dtype = int)

    print(f'\npre-processing our data to use {selected_preprocess_method} on our train and test sets...\n')
    train_set['data'] = preprocess(train_set['data'], selected_preprocess_method)
    test_set['data']  = preprocess(test_set['data'], selected_preprocess_method)

    print(f'\ncreating cross-validation sets to assess test case performance...\n')
    # shuffled_set = split(train_set['data'])
    # produceTestSet(shuffled_set) # note: holding off from producing these until we have our score function

    # exit()

    #> create a list of all test cases to keep track of optimized parameters
    test_cases = configure_testcases(metrics, k_values)
    all_test_cases = range(len(test_cases))

    #> run each test case in our list and
    print(f'\nrunning all test cases with unique hyper-parameters...\n')
    for i in all_test_cases:
        test_cases[i] = run_testcase(test_cases[i], train_set, test_set, knn_graph)

    # todo: find our `best_score_index` a.k.a the lowest score
    # best_score_index = np.argpartition(test_cases)
    best_score_index = 0 # note: dummy value
    # test_cases[best_score_index]['score'] = 0.0

    # note: debug purposes
    print(f"debug: printing the following chosen best score: {test_cases[best_score_index]['score']}")

    #> write to our .csv file
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    file_name = f"recommend_k{test_cases[best_score_index]['parameters']['k']}_{test_cases[best_score_index]['parameters']['metric']}_{now}.csv"
    # file_name = f"recommend_debug.csv" # note: keep overwriting this file instead of making so many copies of debug output
    write_output(f'../out/final/{file_name}', test_cases[best_score_index]['recommendation'])
