import os
import csv
import random

import numpy as np

from numpy import genfromtxt

def read_input(path : str, trim_header = False):

    if trim_header:
        data = np.delete(genfromtxt(path, delimiter = ',', skip_header = 1, dtype = int), obj = 0, axis = 1)
    else:
        data = genfromtxt(path, delimiter = ',', dtype = int)

    input = {
        'data'   : data,
        'height' : data.shape[0],
        'width'  : data.shape[1],
        'rows'   : range(data.shape[0]),
        'columns': range(data.shape[1])
    }
    return input

def readInFiles(count):
    map = {}
    test = []
    train = []
    answer = []
    map['test'] = test
    map['train'] = train
    map['answer'] = answer
    for i in range(1, 1+count):
        train.append(read_input(f'../train/cv_train/train_cvset_{i}.csv'))
        test.append(read_input(f'../test/rand_test/test_randset_{i}.csv'))
        answer.append(read_input(f'../test/cv_test/test_cvset_{i}.csv'))

    return map

def write_output(path : str, data):
    #> transpose and flatten our recommend set
    data = np.transpose(data)
    data = data.flatten('C')
    data = np.atleast_2d(data).T

    #> write our flattened KNN vector to a .csv file
    with open(path, mode = 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',', lineterminator = '\r\n', quotechar = "'")
        writer.writerow(['\"Id\"', '\"Expected\"'])

        for i in range(len(data)):
            writer.writerow([f'\"row_{i + 1}\"', f'{data[i][0]}'])

# shuffling columns for k-fold
def randomize(data):

    res = data
    row = data['height']
    col = data['width']
    for i in range(col):
        temp = []
        for j in range(row):
            a = res['data'][j][i]
            if a != 0:
                temp.append(j)

        for j in range(5):
            if len(temp) == 0:
                break
            choose = random.choice(temp)
            res['data'][choose][i] = 0
            temp.remove(choose)

    return res

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

# suggestion would be in form of matrix of 1s and 0s,
# main evaluation function, implementing MAE evaluation metric
def getScore(suggestion, actual):

    rows = len(suggestion)
    cols = len(suggestion[0])
    res = 0.0

    for i in range(cols):

        mismatch = 0.0
        for j in range(rows):

            if suggestion[j][i] != 0 and actual[j][i] == 0:
                mismatch += 1

        res += (mismatch * 2) / 85
    res = res / cols

    return res

def write_plot(path : str, data):
    # todo: this function is called after performing cross-validation to see how our tc performs
    raise NotImplementedError