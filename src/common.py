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