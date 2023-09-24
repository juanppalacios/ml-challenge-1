import os
import csv
import numpy as np

from numpy import genfromtxt

def read_input(path : str):
    data = np.delete(genfromtxt(path, delimiter = ',', skip_header = 1, dtype = int), obj = 0, axis = 1)
    input = {
        'data'   : data,
        'height' : data.shape[0],
        'width'  : data.shape[1],
        'rows'   : range(data.shape[0]),
        'columns': range(data.shape[1])
    }
    return input

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

def write_plot(path : str, data):
    # todo: this function is called after performing cross-validation to see how our tc performs
    raise NotImplementedError

def defunct_main():
    ######? below this line is the old version
    k = 20
    # exit_value = 1000 - 1
    # exit_value = test_set['width'] - 1

    exit()
    knn_graph     = np.zeros((train_set['width'], test_set['width']), dtype = int)
    knn_index     = np.zeros((train_set['width'], k), dtype = int)
    knn_centroids = np.zeros(train_set['height'])

    print(f'finding k nearest neighbors...') # note: 5 tc's costs 0.7856 seconds, 770 seconds for all tc's
    start = time.time()
    for column in track(test_set['columns']):
        # knn_graph[:,column] = KNN(train_set, test_set['data'][:,column], k)
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